import os
import time
import hashlib
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore

# NOTE: no torch / sentence-transformers here. Production uses ONNX (fastembed)
# or a hosted reranker API, so the process fits in 512 MB.
from light_models import LightEmbedder, build_reranker

load_dotenv(override=True)

log = logging.getLogger("ganesh-rag.retriever")


class GaneshRetriever:
    """
    Latency-optimised retriever.

    Changes vs. the original:
      1. Query is embedded ONCE (was: once per collection = 6x).
      2. All 6 collections are searched IN PARALLEL (was: sequential).
      3. Only the best `rerank_per_collection` hits per collection go to the
         cross-encoder (was: 6 x retrieve_k = 60 pairs).
      4. Cross-encoder inputs are truncated (`rerank_max_length`) - bge-reranker-v2-m3
         otherwise accepts very long inputs, which is slow on CPU.
      5. fp16 on GPU if available; model warmed at startup.
      6. Small in-memory cache so identical queries (e.g. after a rewrite that
         returns the same text) are not recomputed.
    """

    def __init__(
        self,
        puranas_collection="puranas",
        research_collection="research",
        iconography_collection="iconography",
        rahasya_collection="rahasya",
        sahastranaam_collection="sahastranaam",
        upanishad_collection="upanishad",
        retrieve_k=10,
        top_k=6,
        rerank_per_collection=5,
        cache_size=128,
    ):
        self.retrieve_k = retrieve_k
        self.top_k = top_k
        self.rerank_per_collection = rerank_per_collection

        # ---------------- embedding model (ONNX, same all-MiniLM-L6-v2 vectors) ----
        self.embedder = LightEmbedder()

        # ---------------- vector stores ----------------
        token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")

        def make_store(collection_name):
            return AstraDBVectorStore(
                collection_name=collection_name,
                embedding=self.embedder,
                token=token,
                api_endpoint=endpoint,
            )

        # label (used in metadata["collection"]) -> store
        self.stores = {
            "puranas": make_store(puranas_collection),
            "research": make_store(research_collection),
            "iconography": make_store(iconography_collection),
            "rahasya": make_store(rahasya_collection),
            "sahastranaam": make_store(sahastranaam_collection),
            "upanishad": make_store(upanishad_collection),
        }

        self._pool = ThreadPoolExecutor(max_workers=len(self.stores))

        # ---------------- reranker (hosted API or small ONNX model) ----------------
        self.reranker = build_reranker()

        # ---------------- cache ----------------
        self._cache = OrderedDict()
        self._cache_size = cache_size

        # ---------------- warm-up ----------------
        self.embedder.embed_query("warmup")
        try:
            self.reranker.score("warmup", ["warmup"])
        except Exception as e:  # e.g. API key missing / network - don't block startup
            log.warning("Reranker warm-up failed: %s", e)

    # =========================================================
    # SINGLE COLLECTION (vector already computed)
    # =========================================================

    def retrieve_from_collection(self, vector_store, vector, collection_name):
        try:
            results = vector_store.similarity_search_with_score_by_vector(
                vector, k=self.retrieve_k
            )

            candidates = []
            for rank, (document, vector_score) in enumerate(results, start=1):
                text = document.page_content or ""
                if not text.strip():
                    continue

                metadata = document.metadata or {}
                metadata["collection"] = collection_name
                document.metadata = metadata

                candidates.append(
                    {
                        "document": document,
                        "text": text,
                        "metadata": metadata,
                        "vector_score": vector_score,
                        "vector_rank": rank,
                        "collection": collection_name,
                    }
                )
            return candidates

        except Exception as e:
            log.warning("Error retrieving from %s: %s", collection_name, e)
            return []

    # =========================================================
    # AUTHORITY (unchanged; note it is not used in the sort)
    # =========================================================

    def get_authority_score(self, metadata):
        source_type = (metadata.get("source_type") or "").lower()
        authority_scores = {
            "purana": 1.0,
            "upanishad": 0.95,
            "traditional_text": 0.90,
            "sahasranama": 0.89,
            "iconography": 0.88,
            "rahasya": 0.87,
            "research": 0.60,
            "etic": 0.50,
        }
        return authority_scores.get(source_type, 0.40)

    # =========================================================
    # RETRIEVE + RERANK
    # =========================================================

    def retrieve(self, query: str):
        key = query.strip()

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        t0 = time.perf_counter()

        # 1. embed once
        vector = self.embedder.embed_query(key)
        t_embed = time.perf_counter()

        # 2. search all collections in parallel
        futures = {
            label: self._pool.submit(
                self.retrieve_from_collection, store, vector, label
            )
            for label, store in self.stores.items()
        }

        candidates, seen = [], set()
        counts = {}
        for label, fut in futures.items():
            hits = fut.result()
            counts[label] = len(hits)
            # hits are already vector-ranked; keep the best N per collection
            for c in hits[: self.rerank_per_collection]:
                h = hashlib.md5(c["text"].encode("utf-8")).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                candidates.append(c)
        t_search = time.perf_counter()

        if not candidates:
            return []

        # 3. one cross-encoder pass
        pairs = candidates  # (kept for the log line below)
        try:
            scores = self.reranker.score(key, [c["text"] for c in candidates])
            for c, s in zip(candidates, scores):
                c["reranker_score"] = float(s)
            sort_key = lambda x: x["reranker_score"]
        except Exception as e:
            # Reranker API down / rate-limited: fall back to vector order instead of failing.
            # reranker_score stays None, so grade_documents will use the LLM grading path.
            log.warning("Rerank failed, using vector order: %s", e)
            for c in candidates:
                c["reranker_score"] = None
            sort_key = lambda x: x["vector_score"]

        for c in candidates:
            c["authority_score"] = self.get_authority_score(c["metadata"])

        candidates.sort(key=sort_key, reverse=True)
        final_candidates = candidates[: self.top_k]
        t_rerank = time.perf_counter()

        log.info(
            "retrieve: embed=%dms search=%dms rerank=%dms (%d pairs) total=%dms hits=%s",
            (t_embed - t0) * 1000,
            (t_search - t_embed) * 1000,
            (t_rerank - t_search) * 1000,
            len(pairs),
            (t_rerank - t0) * 1000,
            counts,
        )

        # 4. cache
        self._cache[key] = final_candidates
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return final_candidates
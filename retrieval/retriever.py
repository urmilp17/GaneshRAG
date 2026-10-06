import os
import time
import hashlib
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore

from embedder import SentenceTransformerEmbeddings

load_dotenv(override=True)

log = logging.getLogger("ganesh-rag.retriever")

OPENROUTER_RERANK_URL = "https://openrouter.ai/api/v1/rerank"

# Tried in order. Override with e.g. RERANK_MODELS="voyageai/rerank-2.5-lite"
DEFAULT_RERANK_MODELS = [
    m.strip()
    for m in os.getenv(
        "RERANK_MODELS", "cohere/rerank-v3.5,voyageai/rerank-2.5-lite,qwen/qwen3-reranker-8b,"
    ).split(",")
    if m.strip()
]


class GaneshRetriever:
    """
    Memory-light retriever for small hosts (e.g. Render 512 MB).

      * Embeddings : fastembed / ONNX all-MiniLM-L6-v2 (no PyTorch). Same 384-d vectors,
                     so existing Astra collections keep working.
      * Reranking  : OpenRouter /rerank API (no local cross-encoder weights in RAM).
      * Search     : query embedded once, 6 collections searched in parallel.
      * Fallback   : if the rerank API fails, results fall back to vector order and
                     `reranker_score` is None (so graders treat them as "unsure").
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
        rerank_models=None,
        rerank_per_collection=5,
        rerank_doc_chars=1500,
        rerank_timeout=int(os.getenv("RERANK_TIMEOUT", "12")),
        cache_size=128,
    ):
        self.retrieve_k = retrieve_k
        self.top_k = top_k
        self.rerank_per_collection = rerank_per_collection
        self.rerank_doc_chars = rerank_doc_chars
        self.rerank_timeout = rerank_timeout
        self.rerank_models = rerank_models or DEFAULT_RERANK_MODELS

        # ---------------- embedding model (ONNX) ----------------
        self.embedder = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2",
            cache_dir=os.getenv("FASTEMBED_CACHE_DIR"),
        )

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

        # label (written to metadata["collection"]) -> store
        self.stores = {
            "puranas": make_store(puranas_collection),
            "research": make_store(research_collection),
            "iconography": make_store(iconography_collection),
            "rahasya": make_store(rahasya_collection),
            "sahastranaam": make_store(sahastranaam_collection),
            "upanishad": make_store(upanishad_collection),
        }
        self._pool = ThreadPoolExecutor(max_workers=len(self.stores))

        # ---------------- HTTP session for rerank API ----------------
        self._http = requests.Session()
        self._http.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=8))

        # ---------------- cache ----------------
        self._cache = OrderedDict()
        self._cache_size = cache_size

    # =========================================================
    # LIFECYCLE
    # =========================================================

    def warmup(self):
        """Force the ONNX session to initialise so the first user request isn't slow."""
        self.embedder.embed_query("warmup")

    def clear_cache(self):
        self._cache.clear()

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
    # RERANK via OpenRouter
    # =========================================================

    def _rerank_api(self, query, candidates):
        """
        Returns ([(candidate_index, relevance_score), ...] best-first, model_used)
        or (None, None) if every model failed.
        """
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            log.warning("OPENROUTER_API_KEY missing; skipping rerank")
            return None, None

        documents = [c["text"][: self.rerank_doc_chars] for c in candidates]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        last_error = None
        for model in self.rerank_models:
            payload = {
                "model": model,
                "query": query,
                "documents": documents,
                "top_n": min(self.top_k, len(documents)),
            }

            for attempt in range(2):
                try:
                    r = self._http.post(
                        OPENROUTER_RERANK_URL,
                        headers=headers,
                        json=payload,
                        timeout=self.rerank_timeout,
                    )
                except requests.exceptions.RequestException as e:
                    last_error = f"{model}: {e}"
                    break  # timeout / network: go straight to the next model

                if r.status_code == 200:
                    results = r.json().get("results") or []
                    if results:
                        ranked = [
                            (int(x["index"]), float(x["relevance_score"]))
                            for x in results
                        ]
                        ranked.sort(key=lambda t: t[1], reverse=True)
                        return ranked, model
                    last_error = f"{model}: empty results"
                    break

                if r.status_code in (429, 500, 502, 503, 524, 529) and attempt == 0:
                    last_error = f"{model}: HTTP {r.status_code}"
                    time.sleep(0.4)
                    continue  # one quick retry for transient errors

                last_error = f"{model}: HTTP {r.status_code} {r.text[:200]}"
                break  # 4xx etc: try next model

        log.warning("Rerank failed on all models (%s); using vector order", last_error)
        return None, None

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
            label: self._pool.submit(self.retrieve_from_collection, store, vector, label)
            for label, store in self.stores.items()
        }

        candidates, seen, counts = [], set(), {}
        for label, fut in futures.items():
            hits = fut.result()
            counts[label] = len(hits)
            for c in hits[: self.rerank_per_collection]:  # hits are already vector-ranked
                h = hashlib.md5(c["text"].encode("utf-8")).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                candidates.append(c)
        t_search = time.perf_counter()

        if not candidates:
            return []

        # 3. rerank via API (single call)
        ranked, model_used = self._rerank_api(key, candidates)

        if ranked is not None:
            final_candidates = []
            for idx, score in ranked[: self.top_k]:
                c = candidates[idx]
                c["reranker_score"] = score
                c["authority_score"] = self.get_authority_score(c["metadata"])
                final_candidates.append(c)
        else:
            # graceful degradation: best vector hits, no reranker score
            candidates.sort(key=lambda c: c["vector_score"], reverse=True)
            final_candidates = candidates[: self.top_k]
            for c in final_candidates:
                c["reranker_score"] = None
                c["authority_score"] = self.get_authority_score(c["metadata"])
            model_used = "vector-fallback"
        t_rerank = time.perf_counter()

        log.info(
            "retrieve: embed=%dms search=%dms rerank=%dms (%d docs, %s) total=%dms hits=%s",
            (t_embed - t0) * 1000,
            (t_search - t_embed) * 1000,
            (t_rerank - t_search) * 1000,
            len(candidates),
            model_used,
            (t_rerank - t0) * 1000,
            counts,
        )

        # 4. cache (only when reranking worked, so failures aren't remembered)
        if ranked is not None:
            self._cache[key] = final_candidates
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        return final_candidates
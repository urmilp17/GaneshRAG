import os
import re
import time
import hashlib
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from dotenv import load_dotenv
from langchain_core.documents import Document
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

# ---------------------------------------------------------------------------
# Chapter expansion ("small-to-big" retrieval)
# ---------------------------------------------------------------------------
# Collections whose chunks belong to chapters (label = metadata["collection"]).
EXPAND_COLLECTIONS = {
    c.strip()
    for c in os.getenv("EXPAND_COLLECTIONS", "puranas").split(",")
    if c.strip()
}

# Question wording that usually wants a whole story/chapter, not a snippet.
NARRATIVE_INTENT = re.compile(
    r"\b(narrative|story|stories|tale|legend|katha|retell|recount|chapter)\b", re.I
)
# A chunk that starts with "Chapter 28: ..." is the OPENING of a chapter.
CHAPTER_OPENING = re.compile(r"^\s*chapter\s+\d+\s*:", re.I)
CHUNK_NO = re.compile(r"chunk[_-]?(\d+)", re.I)

CHAPTER_MODES = ("off", "auto", "always")


class GaneshRetriever:
    """
    Memory-light retriever for small hosts (e.g. Render 512 MB).

      * Embeddings : fastembed / ONNX all-MiniLM-L6-v2 (no PyTorch). Same 384-d vectors,
                     so existing Astra collections keep working.
      * Reranking  : OpenRouter /rerank API (no local cross-encoder weights in RAM).
      * Search     : query embedded once, 6 collections searched in parallel.
      * Fallback   : if the rerank API fails, results fall back to vector order and
                     `reranker_score` is None (so graders treat them as "unsure").
      * Chapters   : after reranking, the best hit's whole chapter can be loaded with a
                     metadata filter (source + section + chapter_number) so long
                     narratives are no longer cut off after their opening chunk.
                     chapter_mode = "off" | "auto" (default) | "always".
    """

    def __init__(
        self,
        puranas_collection="puranas",
        research_collection="research",
        iconography_collection="iconography",
        rahasya_collection="rahasya",
        sahastranaam_collection="sahastranaam",
        upanishad_collection="upanishads",
        retrieve_k=10,
        top_k=6,
        rerank_models=None,
        rerank_per_collection=5,
        rerank_doc_chars=1500,
        rerank_timeout=int(os.getenv("RERANK_TIMEOUT", "12")),
        cache_size=128,
        expand_max_chunks=int(os.getenv("EXPAND_MAX_CHUNKS", "80")),
        expand_max_chars=int(os.getenv("EXPAND_MAX_CHARS", "24000")),
    ):
        self.retrieve_k = retrieve_k
        self.top_k = top_k
        self.rerank_per_collection = rerank_per_collection
        self.rerank_doc_chars = rerank_doc_chars
        self.rerank_timeout = rerank_timeout
        self.rerank_models = rerank_models or DEFAULT_RERANK_MODELS
        self.expand_max_chunks = expand_max_chunks
        self.expand_max_chars = expand_max_chars
        self.expand_collections = EXPAND_COLLECTIONS

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
    # CHAPTER EXPANSION  (small-to-big)
    # =========================================================

    @staticmethod
    def _chunk_index(doc):
        """Position of a chunk inside its chapter (metadata, else parsed from the id)."""
        meta = doc.metadata or {}
        raw = meta.get("chunk_index")
        if raw is not None and not isinstance(raw, bool):
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
        for candidate in (meta.get("chunk_id"), getattr(doc, "id", None)):
            if candidate:
                m = CHUNK_NO.search(str(candidate))
                if m:
                    return int(m.group(1))
        return None

    @staticmethod
    def _chapter_filter(meta):
        """Metadata filter that identifies ONE chapter, or None if it can't be built."""
        def has(k):
            return meta.get(k) not in (None, "", -1)

        if has("source") and has("chapter_number"):
            keys = ["source", "chapter_number"]
        elif has("source") and has("chapter"):
            keys = ["source", "chapter"]
        else:
            return None

        if has("section"):
            keys.insert(1, "section")  # e.g. "Khanda 4" - chapter numbers repeat per khanda
        return {k: meta[k] for k in keys}

    def _chapter_key(self, candidate):
        flt = self._chapter_filter(candidate["metadata"])
        if not flt:
            return None
        return (candidate["collection"], tuple(sorted((k, str(v)) for k, v in flt.items())))

    def _pick_expansion_target(self, mode, final_candidates, query):
        if mode == "off" or not final_candidates:
            return None

        if mode == "always":
            # best-ranked hit that lives in a chaptered collection
            return next(
                (c for c in final_candidates if c["collection"] in self.expand_collections),
                None,
            )

        # auto: only the overall top hit, and only with a clear signal
        top = final_candidates[0]
        if top["collection"] not in self.expand_collections:
            return None

        idx = self._chunk_index(top["document"])
        is_opening = (idx == 0) or bool(CHAPTER_OPENING.match(top["text"]))
        wants_narrative = bool(NARRATIVE_INTENT.search(query))
        return top if (is_opening or wants_narrative) else None

    def _expand_chapter(self, target, vector, query):
        """Load every chunk of the target's chapter and merge them in reading order."""
        flt = self._chapter_filter(target["metadata"])
        if not flt:
            return None

        store = self.stores[target["collection"]]
        try:
            try:
                results = store.similarity_search_with_score_by_vector(
                    vector, k=self.expand_max_chunks, filter=flt
                )
            except TypeError:
                # older langchain-astradb without `filter` on the by_vector call
                results = store.similarity_search_with_score(
                    query, k=self.expand_max_chunks, filter=flt
                )
        except Exception as e:
            log.warning("Chapter expansion failed for %s: %s", flt, e)
            return None

        docs = [d for d, _ in results if (d.page_content or "").strip()]
        if len(docs) <= 1:
            return None  # nothing more to add

        def sort_key(d):
            idx = self._chunk_index(d)
            return (idx if idx is not None else 10**9, str(getattr(d, "id", "") or ""))

        docs.sort(key=sort_key)

        parts, seen_text, total, truncated = [], set(), 0, False
        for d in docs:
            text = d.page_content.strip()
            if text in seen_text:
                continue
            seen_text.add(text)
            if total + len(text) > self.expand_max_chars:
                truncated = True
                break
            parts.append(text)
            total += len(text) + 2

        if not parts:
            return None

        if len(results) >= self.expand_max_chunks:
            truncated = True  # chapter may have more chunks than we asked for

        merged_text = "\n\n".join(parts)
        if truncated:
            merged_text += "\n\n[The chapter continues; this excerpt was shortened for length.]"

        meta = dict(target["metadata"])
        meta.pop("chunk_index", None)
        meta["chunk_id"] = f"full chapter ({len(parts)} chunks)"
        meta["expanded"] = True
        meta["chunks_merged"] = len(parts)

        merged_doc = Document(page_content=merged_text, metadata=meta)

        expanded = dict(target)  # keeps reranker_score, vector_rank, authority_score ...
        expanded.update(
            {
                "document": merged_doc,
                "text": merged_text,
                "metadata": meta,
                "expanded": True,
                "chunks_merged": len(parts),
                "truncated": truncated,
            }
        )
        return expanded

    # =========================================================
    # RETRIEVE + RERANK (+ chapter expansion)
    # =========================================================

    def retrieve(self, query: str, chapter_mode: str = "auto"):
        key = query.strip()

        mode = (chapter_mode or "auto").lower()
        if mode not in CHAPTER_MODES:
            mode = "auto"

        cache_key = (key, mode)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

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

        # 4. chapter expansion: swap the best hit for its whole chapter
        expanded_info = "no"
        target = self._pick_expansion_target(mode, final_candidates, key)
        if target is not None:
            expanded = self._expand_chapter(target, vector, key)
            if expanded is not None:
                chapter = self._chapter_key(target)
                final_candidates = [
                    expanded if c is target else c
                    for c in final_candidates
                    if c is target or self._chapter_key(c) != chapter
                ]
                expanded_info = f"{expanded['chunks_merged']} chunks"
        t_expand = time.perf_counter()

        log.info(
            "retrieve: embed=%dms search=%dms rerank=%dms (%d docs, %s) expand=%dms [%s, mode=%s] total=%dms hits=%s",
            (t_embed - t0) * 1000,
            (t_search - t_embed) * 1000,
            (t_rerank - t_search) * 1000,
            len(candidates),
            model_used,
            (t_expand - t_rerank) * 1000,
            expanded_info,
            mode,
            (t_expand - t0) * 1000,
            counts,
        )

        # 5. cache (only when reranking worked, so failures aren't remembered)
        if ranked is not None:
            self._cache[cache_key] = final_candidates
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        return final_candidates
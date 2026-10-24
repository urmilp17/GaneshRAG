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
# Context expansion ("small-to-big" retrieval)
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

      * Embeddings : fastembed / ONNX all-MiniLM-L6-v2 (no PyTorch).
      * Reranking  : OpenRouter /rerank API.
      * Search     : query embedded once, 6 collections searched in parallel.
      * Fallback   : if the rerank API fails, results fall back to vector order.

    Context expansion (chapter_mode = "off" | "auto" (default) | "always"):
      * FULL CHAPTER - when the question asks for a story/chapter, or the best hit is a
        chapter-opening chunk, the whole chapter is loaded (up to EXPAND_MAX_CHAPTERS).
      * WINDOW       - otherwise (auto mode), the best hits get their neighbouring chunks
        (WINDOW_BEFORE before, WINDOW_AFTER after). This fixes dialogue-style Puranas where a
        question ("Janaka said: How did Ganesha become embodied?") sits in one chunk and the
        answer ("Antariksha said: ...") in the next. Hits that end with a "?" are windowed
        even if they rank low.
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
        # full-chapter expansion
        expand_max_chunks=int(os.getenv("EXPAND_MAX_CHUNKS", "80")),
        expand_max_chars=int(os.getenv("EXPAND_MAX_CHARS", "24000")),
        expand_max_chapters=int(os.getenv("EXPAND_MAX_CHAPTERS", "2")),
        expand_total_chars=int(os.getenv("EXPAND_TOTAL_CHARS", "40000")),
        # neighbour-chunk windows
        window_hits=int(os.getenv("WINDOW_HITS", "3")),
        window_before=int(os.getenv("WINDOW_BEFORE", "1")),
        window_after=int(os.getenv("WINDOW_AFTER", "2")),
        window_max_chars=int(os.getenv("WINDOW_MAX_CHARS", "6000")),
        window_max_groups=int(os.getenv("WINDOW_MAX_GROUPS", "4")),
    ):
        self.retrieve_k = retrieve_k
        self.top_k = top_k
        self.rerank_per_collection = rerank_per_collection
        self.rerank_doc_chars = rerank_doc_chars
        self.rerank_timeout = rerank_timeout
        self.rerank_models = rerank_models or DEFAULT_RERANK_MODELS

        self.expand_max_chunks = expand_max_chunks
        self.expand_max_chars = expand_max_chars
        self.expand_max_chapters = expand_max_chapters
        self.expand_total_chars = expand_total_chars
        self.expand_collections = EXPAND_COLLECTIONS

        self.window_hits = window_hits
        self.window_before = window_before
        self.window_after = window_after
        self.window_max_chars = window_max_chars
        self.window_max_groups = window_max_groups

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
    # CHUNK / CHAPTER HELPERS
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
            keys.insert(1, "section")  # chapter numbers repeat in every Khanda
        return {k: meta[k] for k in keys}

    def _chapter_key(self, candidate):
        flt = self._chapter_filter(candidate["metadata"])
        if not flt:
            return None
        return (candidate["collection"], tuple(sorted((k, str(v)) for k, v in flt.items())))

    @staticmethod
    def _trim_overlap(prev, nxt, min_overlap=25, max_overlap=600):
        """Chunks overlap, so a chunk often starts with the end of the previous one. Drop that."""
        limit = min(len(prev), len(nxt), max_overlap)
        for k in range(limit, min_overlap - 1, -1):
            if prev.endswith(nxt[:k]):
                return nxt[k:].lstrip()
        return nxt

    @staticmethod
    def _join_parts(parts):
        """Join chunks; a chunk that ends mid-sentence is continued with a space, not a blank line."""
        out = parts[0]
        for part in parts[1:]:
            ends_sentence = re.search(r'[.!?"\u201d\u2019\u0964\u0965)\]]\s*$', out)
            out += ("\n\n" if ends_sentence else " ") + part
        return out

    def _merge_docs(self, docs, cap):
        """Overlap-trim and size-cap a run of chunks. Returns (parts, truncated)."""
        parts, seen, total, truncated = [], set(), 0, False
        for d in docs:
            text = d.page_content.strip()
            if parts:
                text = self._trim_overlap(parts[-1], text)
            if not text or text in seen:
                continue
            seen.add(text)
            if total + len(text) > cap:
                truncated = True
                break
            parts.append(text)
            total += len(text) + 2
        return parts, truncated

    def _fetch_chapter_docs(self, target, vector, query):
        """All chunks of the target's chapter, in reading order. Returns (docs, maybe_more)."""
        flt = self._chapter_filter(target["metadata"])
        if not flt:
            return None, False

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
            log.warning("Chapter fetch failed for %s: %s", flt, e)
            return None, False

        docs = [d for d, _ in results if (d.page_content or "").strip()]

        def sort_key(d):
            idx = self._chunk_index(d)
            return (idx if idx is not None else 10**9, str(getattr(d, "id", "") or ""))

        docs.sort(key=sort_key)
        return docs, len(results) >= self.expand_max_chunks

    @staticmethod
    def _locate(docs, doc):
        """Position of `doc` (a search hit) inside the sorted chapter `docs`."""
        doc_id = getattr(doc, "id", None)
        chunk_id = (doc.metadata or {}).get("chunk_id")
        text = (doc.page_content or "").strip()

        if doc_id:
            for i, d in enumerate(docs):
                if getattr(d, "id", None) == doc_id:
                    return i
        if chunk_id:
            for i, d in enumerate(docs):
                if (d.metadata or {}).get("chunk_id") == chunk_id:
                    return i
        for i, d in enumerate(docs):
            if d.page_content.strip() == text:
                return i
        return None

    def _make_block(self, target, text, n_chunks, kind, rng, truncated):
        """A merged passage that replaces `target` in the results."""
        meta = dict(target["metadata"])
        meta.pop("chunk_index", None)
        meta["chunks_merged"] = n_chunks
        if kind == "chapter":
            meta["chunk_id"] = f"full chapter ({n_chunks} chunks)"
            meta["expanded"] = True
        else:
            meta["chunk_id"] = f"passage ({n_chunks} adjacent chunks)"
            meta["windowed"] = True

        block = dict(target)  # keeps reranker_score, vector_rank, authority_score ...
        block.update(
            {
                "document": Document(page_content=text, metadata=meta),
                "text": text,
                "metadata": meta,
                "expanded": kind == "chapter",
                "windowed": kind == "window",
                "chunks_merged": n_chunks,
                "truncated": truncated,
                "_range": rng,  # "all" for a full chapter, (lo_idx, hi_idx) for a window
            }
        )
        return block

    # =========================================================
    # FULL-CHAPTER EXPANSION
    # =========================================================

    def _pick_expansion_targets(self, mode, final_candidates, query):
        """Hits to swap for their whole chapter (at most expand_max_chapters)."""
        if mode == "off" or not final_candidates:
            return []

        def expandable(c):
            return c["collection"] in self.expand_collections

        if mode == "always":
            pool = [c for c in final_candidates if expandable(c)]
        else:
            # auto: the overall top hit must be in a chaptered collection AND show a clear signal
            top = final_candidates[0]
            if not expandable(top):
                return []

            idx = self._chunk_index(top["document"])
            is_opening = (idx == 0) or bool(CHAPTER_OPENING.match(top["text"]))
            wants_narrative = bool(NARRATIVE_INTENT.search(query))
            if not (is_opening or wants_narrative):
                return []

            # A story can run across chapters, so narrative questions also consider the next hits.
            pool = [top]
            if wants_narrative:
                pool += [c for c in final_candidates[1:3] if expandable(c)]

        targets, seen = [], set()
        for c in pool:
            key = self._chapter_key(c)
            if key is None or key in seen:
                continue
            seen.add(key)
            targets.append(c)
            if len(targets) >= max(1, self.expand_max_chapters):
                break
        return targets

    def _expand_chapter(self, target, vector, query, max_chars=None):
        docs, maybe_more = self._fetch_chapter_docs(target, vector, query)
        if not docs or len(docs) <= 1:
            return None  # nothing more to add

        parts, truncated = self._merge_docs(docs, max_chars or self.expand_max_chars)
        if not parts:
            return None

        truncated = truncated or maybe_more
        text = self._join_parts(parts)
        if truncated:
            text += "\n\n[The chapter continues; this excerpt was shortened for length.]"

        return self._make_block(target, text, len(parts), "chapter", "all", truncated)

    # =========================================================
    # NEIGHBOUR-CHUNK WINDOWS  (question in one chunk, answer in the next)
    # =========================================================

    def _pick_window_groups(self, final_candidates):
        """Group the hits worth widening by chapter (best-ranked hit first in each group)."""
        chosen = []
        for rank, c in enumerate(final_candidates):
            if c["collection"] not in self.expand_collections:
                continue
            awaiting_answer = "?" in c["text"][-400:]  # ends on a question -> answer is next
            if rank < self.window_hits or awaiting_answer:
                chosen.append(c)

        groups = OrderedDict()
        for c in chosen:
            key = self._chapter_key(c)
            if key is None:
                continue
            groups.setdefault(key, []).append(c)
        return list(groups.values())[: max(1, self.window_max_groups)]

    def _expand_window(self, group, vector, query):
        """
        Widen each hit by window_before / window_after chunks. Windows that overlap or touch are
        merged; far-apart hits in the same chapter stay separate blocks (so we never pull in the
        long stretch between them). Returns a list of (hits_in_block, block).
        """
        docs, _ = self._fetch_chapter_docs(group[0], vector, query)
        if not docs or len(docs) <= 1:
            return []

        order = {id(t): i for i, t in enumerate(group)}  # rank order of the hits
        located = []
        for t in group:
            pos = self._locate(docs, t["document"])
            if pos is not None:
                located.append((pos, t))
        if not located:
            return []

        located.sort(key=lambda x: x[0])
        intervals = []  # [lo, hi, [hits]]
        for pos, t in located:
            lo = max(0, pos - self.window_before)
            hi = min(len(docs), pos + self.window_after + 1)
            if intervals and lo <= intervals[-1][1]:  # overlaps or touches the previous window
                intervals[-1][1] = max(intervals[-1][1], hi)
                intervals[-1][2].append(t)
            else:
                intervals.append([lo, hi, [t]])

        results = []
        for lo, hi, hits in intervals:
            span = docs[lo:hi]
            if len(span) <= 1:
                continue

            parts, truncated = self._merge_docs(span, self.window_max_chars)
            if len(parts) <= 1:
                continue

            text = self._join_parts(parts)
            if truncated:
                text += "\n\n[Passage shortened for length.]"

            first, last = self._chunk_index(span[0]), self._chunk_index(span[-1])
            rng = (first, last) if first is not None and last is not None else None

            anchor = min(hits, key=lambda t: order[id(t)])  # best-ranked hit in this block
            block = self._make_block(anchor, text, len(parts), "window", rng, truncated)
            results.append((anchor, hits, block))
        return results

    # =========================================================
    # RETRIEVE + RERANK (+ expansion)
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

        # 4. expansion (all jobs run in parallel)
        expanded_info = "no"
        jobs = []  # (kind, group_of_hits, future)

        chapter_targets = self._pick_expansion_targets(mode, final_candidates, key)
        if chapter_targets:
            per_chapter_cap = max(
                4000, min(self.expand_max_chars, self.expand_total_chars // len(chapter_targets))
            )
            for t in chapter_targets:
                jobs.append(
                    ("chapter", [t],
                     self._pool.submit(self._expand_chapter, t, vector, key, per_chapter_cap))
                )
        elif mode == "auto":
            for group in self._pick_window_groups(final_candidates):
                jobs.append(
                    ("window", group, self._pool.submit(self._expand_window, group, vector, key))
                )

        blocks = []  # (anchor_hit, hits_in_block, block)
        for kind, group, fut in jobs:
            try:
                res = fut.result(timeout=15)
            except Exception as e:
                log.warning("Context expansion error: %s", e)
                res = None

            if kind == "chapter":
                if res is not None:
                    blocks.append((group[0], [group[0]], res))
            elif res:
                blocks.extend(res)

        if blocks:
            block_by_anchor = {id(anchor): b for anchor, _, b in blocks}
            merged_ids = {id(t) for _, hits, _ in blocks for t in hits}
            blocks_by_chapter = {}
            for anchor, _, b in blocks:
                blocks_by_chapter.setdefault(self._chapter_key(anchor), []).append(b)

            def covered(c):
                ci = self._chunk_index(c["document"])
                for b in blocks_by_chapter.get(self._chapter_key(c), []):
                    rng = b.get("_range")
                    if rng == "all":
                        return True
                    if rng is not None and ci is not None and rng[0] <= ci <= rng[1]:
                        return True
                return False

            merged_list = []
            for c in final_candidates:
                if id(c) in block_by_anchor:
                    merged_list.append(block_by_anchor[id(c)])
                elif id(c) in merged_ids or covered(c):
                    continue  # already inside a merged block
                else:
                    merged_list.append(c)
            final_candidates = merged_list

            expanded_info = " + ".join(
                f"{'chapter' if b['expanded'] else 'window'}:{b['chunks_merged']}"
                for _, _, b in blocks
            )
        t_expand = time.perf_counter()

        log.info(
            "retrieve[chapters-v3]: embed=%dms search=%dms rerank=%dms (%d docs, %s) "
            "expand=%dms [%s, mode=%s] total=%dms hits=%s",
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
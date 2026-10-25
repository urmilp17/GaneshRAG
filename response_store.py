"""
response_store.py - persist every question/answer turn in Astra DB.

Target (configurable through env vars):
    database   : the one behind ASTRA_DB_API_ENDPOINT (e.g. ganesh_data)
    keyspace   : default_keyspace            (ASTRA_RESPONSES_KEYSPACE)
    collection : documents                   (ASTRA_RESPONSES_COLLECTION)

Design goals
  * Never slow down or break the app: saving runs in a background thread and every error is
    logged, never raised.
  * A fixed, versioned document schema (see build_record) so the data stays easy to query.
  * Safe with Astra limits: indexed strings are limited to 8,000 bytes, so when this module
    creates the collection it switches indexing OFF for the long text fields, and when it writes
    to a pre-existing "index everything" collection it retries with truncated text if needed.

Document schema (schema_version = 1)
{
  "_id":            "<uuid4>",
  "schema_version": 1,
  "created_at":     <datetime, UTC>,
  "session_id":     "<random id of the browser session>",
  "question":       "...",
  "question_hash":  "<sha256 of the normalised question>",     # indexed - use it to group / look up
  "search_query":   "..." | null,        # final (possibly rewritten) query used for retrieval
  "answer":         "...",
  "model":          "provider/model" | null,
  "chapter_mode":   "auto" | "always" | "off" | null,
  "timing": {
      "retrieval_ms":    1025.3 | null,  # time spent in retrieve() (summed over rewrite passes)
      "total_ms":        6120.8 | null,  # whole graph run, as seen by the UI
      "retrieval_passes": 1
  },
  "retrieval": {
      "num_sources": 4,
      "sources": [ {collection, source, source_type, section, chapter, chapter_number,
                    page_number, citation, chunk_id, vector_rank, vector_score,
                    reranker_score, expanded, windowed, chunks_merged}, ... ]
  },
  "usage": {prompt_tokens, completion_tokens, total_tokens, reasoning_tokens, cached_tokens,
            cost_usd, usd_to_inr, cost_inr},
  "app": {"name": "ganesh-tattvagyan", "interface": "streamlit"}
}
"""

import hashlib
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

log = logging.getLogger("ganesh-rag.responses")

SCHEMA_VERSION = 1
APP_NAME = "ganesh-tattvagyan"

# Astra's limit for a string inside an INDEXED field is 8,000 bytes (UTF-8).
MAX_INDEXED_BYTES = 7_500
# Long free-text fields: never indexed when we create the collection ourselves.
UNINDEXED_FIELDS = ["question", "answer", "search_query"]

SOURCE_FIELDS = (
    "collection", "source", "source_type", "section", "chapter", "chapter_number",
    "page_number", "citation", "chunk_id", "vector_rank", "vector_score",
    "reranker_score", "expanded", "windowed", "chunks_merged",
)


# ============================================================
# RECORD BUILDING  (pure functions - easy to test)
# ============================================================

def _clean(value: Any) -> Any:
    """Make a value JSON-safe (numpy scalars -> float, unknown objects -> str)."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _number(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def normalise_question(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "").strip().lower())


def question_hash(question: str) -> str:
    return hashlib.sha256(normalise_question(question).encode("utf-8")).hexdigest()


def build_record(
    question: str,
    result: Dict[str, Any],
    *,
    total_ms: Optional[float] = None,
    chapter_mode: Optional[str] = None,
    session_id: Optional[str] = None,
    usd_to_inr: Optional[float] = None,
    interface: str = "streamlit",
) -> Dict[str, Any]:
    """Turn one graph result into a document that follows the schema above."""
    result = result or {}
    usage = result.get("usage") or {}

    cost_usd = _number(usage.get("cost")) or 0.0
    rate = _number(usd_to_inr)

    sources = []
    for item in result.get("retrieval") or []:
        sources.append(
            {k: _clean(item.get(k)) for k in SOURCE_FIELDS if item.get(k) not in (None, "")}
        )

    search_query = result.get("search_query")
    if search_query is not None and search_query.strip() == question.strip():
        search_query = None  # only keep it when the query was actually rewritten

    rewrite_count = int(result.get("rewrite_count", 0) or 0)

    return {
        "_id": str(uuid.uuid4()),
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc),
        "session_id": session_id,
        "question": question.strip(),
        "question_hash": question_hash(question),
        "search_query": search_query,
        "answer": (result.get("answer") or "").strip(),
        "model": result.get("model"),
        "chapter_mode": chapter_mode,
        "timing": {
            "retrieval_ms": None if result.get("retrieval_ms") is None
            else round(float(result["retrieval_ms"]), 1),
            "total_ms": None if total_ms is None else round(float(total_ms), 1),
            "retrieval_passes": rewrite_count + 1,
        },
        "retrieval": {
            "num_sources": len(sources),
            "sources": sources,
        },
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "reasoning_tokens": int(usage.get("reasoning_tokens", 0) or 0),
            "cached_tokens": int(usage.get("cached_tokens", 0) or 0),
            "cost_usd": cost_usd,
            "usd_to_inr": rate,
            "cost_inr": None if rate is None else round(cost_usd * rate, 6),
        },
        "app": {"name": APP_NAME, "interface": interface},
    }


def _truncate_bytes(text: str, limit: int = MAX_INDEXED_BYTES):
    raw = (text or "").encode("utf-8")
    if len(raw) <= limit:
        return text, False
    return raw[:limit].decode("utf-8", errors="ignore"), True


def _shrink_long_strings(record: Dict[str, Any]) -> bool:
    """Truncate the long text fields in place. Returns True if anything changed."""
    changed = []
    for key in UNINDEXED_FIELDS:
        value = record.get(key)
        if isinstance(value, str):
            shorter, did = _truncate_bytes(value)
            if did:
                record[key] = shorter
                changed.append(key)
    if changed:
        record["truncated_fields"] = changed
    return bool(changed)


# ============================================================
# STORE
# ============================================================

class ResponseStore:
    """Thin wrapper around an astrapy Collection."""

    def __init__(self, collection):
        self.collection = collection

    # ---------------- construction ----------------

    @classmethod
    def from_env(cls) -> "ResponseStore":
        token = os.getenv("ASTRA_RESPONSES_TOKEN") or os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        endpoint = os.getenv("ASTRA_RESPONSES_API_ENDPOINT") or os.getenv("ASTRA_DB_API_ENDPOINT")
        keyspace = os.getenv("ASTRA_RESPONSES_KEYSPACE", "default_keyspace")
        name = os.getenv("ASTRA_RESPONSES_COLLECTION", "documents")

        if not token or not endpoint:
            raise RuntimeError("Astra credentials missing (token / API endpoint).")

        from astrapy import DataAPIClient

        client = DataAPIClient(token)
        try:
            database = client.get_database(endpoint, keyspace=keyspace)
        except TypeError:  # very old astrapy
            database = client.get_database(endpoint, namespace=keyspace)

        return cls(cls._get_or_create_collection(database, name))

    @staticmethod
    def _get_or_create_collection(database, name):
        if name in database.list_collection_names():
            return database.get_collection(name)

        log.info("Creating collection %r (long text fields not indexed)", name)
        deny = {"deny": UNINDEXED_FIELDS}

        def v2_definition():  # astrapy >= 2
            from astrapy.info import CollectionDefinition
            return database.create_collection(
                name, definition=CollectionDefinition(indexing=deny)
            )

        def v1_kwarg():  # astrapy 1.x
            return database.create_collection(name, indexing=deny)

        def plain():  # last resort: default indexing (the retry-with-truncation path covers it)
            return database.create_collection(name)

        attempts = [v2_definition, v1_kwarg, plain]
        last_error = None
        for attempt in attempts:
            try:
                return attempt()
            except Exception as e:  # noqa: BLE001 - try the next API flavour
                last_error = e
                if name in database.list_collection_names():
                    return database.get_collection(name)
        raise RuntimeError(f"Could not create collection {name!r}: {last_error}")

    # ---------------- writing ----------------

    def insert(self, record: Dict[str, Any]) -> bool:
        try:
            self.collection.insert_one(record)
            return True
        except Exception as first_error:  # noqa: BLE001
            # Typical cause on a pre-existing "index everything" collection: a string > 8,000 bytes.
            if _shrink_long_strings(record):
                try:
                    self.collection.insert_one(record)
                    log.info("Stored response with truncated text fields")
                    return True
                except Exception as second_error:  # noqa: BLE001
                    log.warning("Could not store response: %s", second_error)
                    return False
            log.warning("Could not store response: %s", first_error)
            return False


# ============================================================
# PROCESS-WIDE SINGLETON + BACKGROUND SAVING
# ============================================================

_store: Optional[ResponseStore] = None
_store_lock = threading.Lock()
_last_failure = 0.0
_RETRY_AFTER_SECONDS = 300
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="response-store")
_warm_started = False


def _enabled() -> bool:
    return os.getenv("STORE_RESPONSES", "true").strip().lower() not in ("0", "false", "off", "no")


def get_default_store() -> Optional[ResponseStore]:
    """Connect once; after a failure, wait a few minutes before trying again."""
    global _store, _last_failure

    if _store is not None:
        return _store
    if not _enabled():
        return None

    with _store_lock:
        if _store is not None:
            return _store
        if _last_failure and time.time() - _last_failure < _RETRY_AFTER_SECONDS:
            return None
        try:
            _store = ResponseStore.from_env()
            log.info("Response store ready")
        except Exception as e:  # noqa: BLE001
            _last_failure = time.time()
            log.warning("Response store unavailable: %s", e)
            return None
        return _store


def warm_up_async() -> None:
    """Open the DB connection in the background (once) so the first save isn't slow."""
    global _warm_started
    if _warm_started or not _enabled():
        return
    _warm_started = True
    _executor.submit(get_default_store)


def _save_worker(record: Dict[str, Any]) -> None:
    try:
        store = get_default_store()
        if store is not None:
            store.insert(record)
    except Exception as e:  # noqa: BLE001 - never let logging break the app
        log.warning("Saving response failed: %s", e)


def save_response_async(record: Dict[str, Any]) -> None:
    """Fire-and-forget: returns immediately, the insert happens in a worker thread."""
    if not _enabled():
        return
    _executor.submit(_save_worker, record)
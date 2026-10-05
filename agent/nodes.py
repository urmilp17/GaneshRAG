import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
import dotenv

dotenv.load_dotenv(override=True)

from pydantic import BaseModel, Field

from agent.prompts import (
    GRADE_PROMPT,
    REWRITE_PROMPT,
    ANSWER_PROMPT,
)

from retrieval.retriever import GaneshRetriever

log = logging.getLogger("ganesh-rag.nodes")


# ============================================================
# TUNING KNOBS (override with env vars)
# ============================================================

# If the best reranker score is >= this, skip LLM grading entirely.
# bge-reranker-v2-m3 via sentence-transformers returns 0..1 (sigmoid).
# Calibrate by looking at `retrieval[*].reranker_score` for known good/bad queries.
RELEVANCE_SKIP_THRESHOLD = float(os.getenv("RELEVANCE_SKIP_THRESHOLD", "0.5"))

# When the reranker isn't confident, LLM-grade only the top N docs (in parallel).
GRADE_MAX_DOCS = int(os.getenv("GRADE_MAX_DOCS", "3"))

# Max characters of a doc sent to the grader.
GRADE_DOC_CHARS = int(os.getenv("GRADE_DOC_CHARS", "1500"))

GRADE_TIMEOUT = int(os.getenv("GRADE_TIMEOUT", "15"))
REWRITE_TIMEOUT = int(os.getenv("REWRITE_TIMEOUT", "15"))
ANSWER_TIMEOUT = int(os.getenv("ANSWER_TIMEOUT", "60"))


# ============================================================
# GLOBAL RETRIEVER (loaded once at import)
# ============================================================

retriever = GaneshRetriever(
    puranas_collection="puranas",
    research_collection="research",
    iconography_collection="iconography",
    rahasya_collection="rahasya",
    sahastranaam_collection="sahastranaam",
    upanishad_collection="upanishads",
    retrieve_k=10,
    top_k=6,
)


# ============================================================
# USAGE HELPERS
# ============================================================

_USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cached_tokens",
)


def empty_usage():
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
    }


def add_usage(existing_usage, new_usage):
    """Sum usage across all LLM calls in the agentic workflow."""
    existing_usage = existing_usage or empty_usage()
    new_usage = new_usage or empty_usage()

    total = {k: existing_usage.get(k, 0) + new_usage.get(k, 0) for k in _USAGE_KEYS}
    total["cost"] = float(existing_usage.get("cost", 0)) + float(new_usage.get("cost", 0))
    return total


# ============================================================
# OPENROUTER CALL (pooled connections, per-call timeout)
# ============================================================

_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_connections=10, pool_maxsize=20))

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODELS = [
    "deepseek/deepseek-v4.1-flash",
    "inclusionai/ling-3.0-flash-vl",
    "nex-agi/nex-n2.5-pro",
]


def call_openrouter(
    prompt,
    models=None,
    temperature=0.2,
    max_tokens=800,
    reasoning_effort="none",
    timeout=60,
):
    """
    Returns {"answer": str, "model": str, "usage": dict}.
    `timeout` is per model attempt. Keep it short for grade/rewrite calls so a slow
    primary model falls through to the next one quickly instead of stalling for 120s.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured.")

    models = models or DEFAULT_MODELS
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    errors = []

    for model in models:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "reasoning": {"effort": reasoning_effort},
            }

            response = _session.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=timeout
            )

            if response.status_code != 200:
                errors.append(f"{model}: HTTP {response.status_code} - {response.text}")
                continue

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                errors.append(f"{model}: No choices returned.")
                continue

            answer = (choices[0].get("message", {}) or {}).get("content", "")
            if not answer:
                errors.append(f"{model}: Empty answer returned.")
                continue

            usage_data = data.get("usage", {}) or {}
            completion_details = usage_data.get("completion_tokens_details", {}) or {}
            prompt_details = usage_data.get("prompt_tokens_details", {}) or {}

            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
                "cost": float(usage_data.get("cost", 0) or 0),
                "reasoning_tokens": completion_details.get("reasoning_tokens", 0),
                "cached_tokens": prompt_details.get("cached_tokens", 0),
            }

            return {
                "answer": answer.strip(),
                "model": data.get("model", model),
                "usage": usage,
            }

        except requests.exceptions.Timeout:
            errors.append(f"{model}: Request timed out.")
        except requests.exceptions.RequestException as e:
            errors.append(f"{model}: Request error - {e}")
        except ValueError as e:
            errors.append(f"{model}: Invalid JSON response - {e}")
        except Exception as e:
            errors.append(f"{model}: Unexpected error - {e}")

    raise RuntimeError("All OpenRouter models failed:\n\n" + "\n".join(errors))


# ============================================================
# NODE 1 - GENERATE SEARCH QUERY
# ============================================================

def generate_query(state):
    question = state["question"]
    return {
        "search_query": question,
        "rewrite_count": state.get("rewrite_count", 0),
        "usage": state.get("usage", empty_usage()),
    }


# ============================================================
# NODE 2 - RETRIEVE DOCUMENTS
# ============================================================

_CANDIDATE_FIELDS = ("collection", "vector_rank", "vector_score", "reranker_score")
_METADATA_FIELDS = (
    "source",
    "source_type",
    "authority",
    "tradition",
    "section",
    "chapter",
    "chapter_number",
    "page_number",
    "citation",
    "chunk_id",
)


def retrieve_documents(state):
    candidates = retriever.retrieve(state["search_query"])

    documents, retrieval_info = [], []

    for candidate in candidates:
        documents.append(candidate["document"])
        metadata = candidate.get("metadata", {}) or {}

        info = {k: candidate.get(k) for k in _CANDIDATE_FIELDS}
        info.update({k: metadata.get(k) for k in _METADATA_FIELDS})
        retrieval_info.append(info)

    return {"documents": documents, "retrieval": retrieval_info}


# ============================================================
# NODE 3 - GRADE DOCUMENTS
#   1) Trust the cross-encoder when it is confident (0 LLM calls).
#   2) Otherwise grade the top few docs IN PARALLEL and stop at the first YES.
#   Original: one sequential LLM call per document (6 calls) on every pass.
# ============================================================

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="YES if relevant, NO if irrelevant")


def _grade_one(question, document):
    prompt = GRADE_PROMPT.format(
        question=question,
        context=document.page_content[:GRADE_DOC_CHARS],
    )
    return call_openrouter(
        prompt,
        temperature=0,
        max_tokens=10,
        timeout=GRADE_TIMEOUT,
    )


def grade_documents(state):
    question = state["question"]
    documents = state.get("documents", [])
    retrieval = state.get("retrieval", []) or []
    usage = state.get("usage", empty_usage())

    if not documents:
        return {"documents_relevant": False, "usage": usage}

    # ---- fast path: reranker is confident -> no LLM call ----
    top_score = max((r.get("reranker_score") or 0.0) for r in retrieval) if retrieval else 0.0
    if top_score >= RELEVANCE_SKIP_THRESHOLD:
        log.info("grade: skipped LLM (top reranker score %.3f)", top_score)
        return {"documents_relevant": True, "usage": usage}

    # ---- slow path: parallel grading of the top docs, early exit on first YES ----
    to_grade = documents[:GRADE_MAX_DOCS]  # already sorted best-first by the reranker
    relevant = False

    pool = ThreadPoolExecutor(max_workers=len(to_grade))
    futures = [pool.submit(_grade_one, question, d) for d in to_grade]
    try:
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as e:
                log.warning("Document grading failed: %s", e)
                continue

            usage = add_usage(usage, result.get("usage"))

            if result.get("answer", "").strip().upper() == "YES":
                relevant = True
                break
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return {"documents_relevant": relevant, "usage": usage}


# ============================================================
# NODE 4 - REWRITE QUESTION
# ============================================================

def rewrite_question(state):
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)
    current_usage = state.get("usage", empty_usage())

    if rewrite_count >= 2:
        return {
            "search_query": question,
            "rewrite_count": rewrite_count,
            "usage": current_usage,
        }

    try:
        result = call_openrouter(
            REWRITE_PROMPT.format(question=question),
            temperature=0,
            max_tokens=100,
            timeout=REWRITE_TIMEOUT,
        )
        rewritten = result.get("answer", state.get("search_query", question))
        total_usage = add_usage(current_usage, result.get("usage", empty_usage()))

    except Exception as e:
        log.warning("Query rewrite failed: %s", e)
        rewritten = state.get("search_query", question)
        total_usage = current_usage

    return {
        "search_query": rewritten.strip(),
        "rewrite_count": rewrite_count + 1,
        "usage": total_usage,
    }


# ============================================================
# NODE 5 - BUILD CONTEXT
#   Original built an f-string with ~12 spaces of indentation on every line
#   and "Not specified" for every empty field. That is a lot of wasted prompt
#   tokens. This version emits only the fields that exist.
# ============================================================

_CONTEXT_FIELDS = (
    ("Collection", "collection"),
    ("Source", "source"),
    ("Source Type", "source_type"),
    ("Authority", "authority"),
    ("Tradition", "tradition"),
    ("Section", "section"),
    ("Chapter", "chapter"),
    ("Chapter Number", "chapter_number"),
    ("Chapter Title", "chapter_title"),
    ("Page Number", "page_number"),
    ("Citation", "citation"),
    ("Chunk ID", "chunk_id"),
)


def build_context(state):
    documents = state.get("documents", [])
    parts = []

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}

        lines = [f"=== SOURCE {index} ==="]
        for label, key in _CONTEXT_FIELDS:
            value = metadata.get(key)
            if value not in (None, "", "unknown"):
                lines.append(f"{label}: {value}")
        lines.append("--- CONTENT ---")
        lines.append(document.page_content)

        parts.append("\n".join(lines))

    return {"context": "\n\n".join(parts)}


# ============================================================
# DETAILED ANSWER DETECTOR
# ============================================================

_DETAILED_PHRASES = (
    "answer in detail",
    "explain in detail",
    "explain this in detail",
    "elaborate in detail",
    "explain thoroughly",
    "give a detailed explanation",
    "provide a detailed answer",
    "explain deeply",
    "in great detail",
    "answer thoroughly",
    "detailed explanation",
)


def wants_detailed_answer(question: str) -> bool:
    q = question.lower()
    return any(phrase in q for phrase in _DETAILED_PHRASES)


# ============================================================
# NODE 6 - GENERATE ANSWER
# ============================================================

def generate_answer(state):
    question = state["question"]
    context = state.get("context", "")
    current_usage = state.get("usage", empty_usage())

    if not context:
        return {
            "answer": (
                "I could not find sufficient information in the available "
                "sources to answer this question."
            ),
            "model": None,
            "usage": current_usage,
        }

    max_tokens = 2000 if wants_detailed_answer(question) else 600

    prompt = ANSWER_PROMPT.format(question=question, context=context)

    try:
        response = call_openrouter(
            prompt,
            temperature=0.2,
            max_tokens=max_tokens,
            timeout=ANSWER_TIMEOUT,
        )

        return {
            "answer": response.get("answer", "No answer generated."),
            "model": response.get("model", "Unknown"),
            "usage": add_usage(current_usage, response.get("usage", empty_usage())),
        }

    except Exception as e:
        log.warning("Answer generation failed: %s", e)
        return {
            "answer": (
                "The language model could not generate an answer "
                "at this time. Please try again."
            ),
            "model": None,
            "usage": current_usage,
        }
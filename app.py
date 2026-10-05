"""
FastAPI version of the Ganesh Tattvagyan RAG Streamlit app.

Routes:
    GET  /health      -> 200 OK (for cronjob / uptime checks)
    POST /query       -> Run RAG query, returns structured JSON
    GET  /            -> Root info

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import html
import logging
from typing import Any, Dict, List, Optional

import dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.graph import graph
from embedder import SentenceTransformerEmbeddings
from langchain_astradb import AstraDBVectorStore


# ============================================================
# ENVIRONMENT
# ============================================================

dotenv.load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ganesh-rag")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Ganesh Tattvagyan RAG API",
    description=(
        "REST API for the Ganesh Tattvagyan RAG pipeline. "
        "Two-stage RAG: Vector Retrieval (Astra DB) + "
        "Cross-Encoder Reranking + Grounded Generation."
    ),
    version="1.0.0",
)

# <-- 2. Add CORS Middleware here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; replace with specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows OPTIONS, POST, GET, etc.
    allow_headers=["*"],
)


# ============================================================
# LAZY SINGLETONS (replaces @st.cache_resource)
# ============================================================

_embedder: Optional[SentenceTransformerEmbeddings] = None
_vector_store: Optional[AstraDBVectorStore] = None
_initialized: bool = False
_init_error: Optional[str] = None


def get_embedder() -> SentenceTransformerEmbeddings:
    """Create (or return cached) embedding model."""
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model all-MiniLM-L6-v2...")
        _embedder = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2",
            device=None,
        )
    return _embedder


# Optional (disabled in Streamlit original, kept for parity)
# def get_vector_store() -> AstraDBVectorStore:
#     global _vector_store
#     if _vector_store is None:
#         embedder = get_embedder()
#         _vector_store = AstraDBVectorStore(
#             collection_name="puranas",
#             embedding=embedder,
#             token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
#             api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
#         )
#     return _vector_store


def initialize_system() -> bool:
    """
    Initialize the RAG system once at startup.
    Returns True on success, False on failure.
    """
    global _initialized, _init_error
    try:
        # Trigger embedder load (mirrors Streamlit's cached_resource warm-up)
        get_embedder()
        _initialized = True
        _init_error = None
        logger.info("RAG system initialized.")
        return True
    except Exception as e:
        _initialized = False
        _init_error = str(e)
        logger.exception("Failed to initialize RAG system")
        return False


@app.on_event("startup")
def _startup() -> None:
    initialize_system()


# ============================================================
# SCHEMAS
# ============================================================

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    rewrite_count: int = Field(0, ge=0, description="Internal rewrite counter")

    # Optional runtime knobs (matching sidebar defaults)
    retrieve_k: int = Field(15, ge=1, le=100)
    top_k: int = Field(5, ge=1, le=50)
    temperature: float = Field(0.2, ge=0.0, le=2.0)


class SourceItem(BaseModel):
    source: Optional[str] = None
    page_number: Optional[Any] = None
    vector_rank: Optional[Any] = None
    vector_score: Optional[float] = None
    reranker_score: Optional[float] = None
    chunk_id: Optional[Any] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    reasoning_tokens: int = 0
    cached_tokens: int = 0


class QueryResponse(BaseModel):
    answer: str
    model: Optional[str] = None
    usage: UsageInfo = UsageInfo()
    retrieval: List[Dict[str, Any]] = []
    documents: List[Dict[str, Any]] = []
    context: Optional[Any] = None
    errors: List[str] = []


class HealthResponse(BaseModel):
    status: str = "ok"


# ============================================================
# HELPERS
# ============================================================

def _serialize_documents(documents: Any) -> List[Dict[str, Any]]:
    """Convert LangChain Document objects to plain dicts."""
    if not documents:
        return []

    serialized: List[Dict[str, Any]] = []
    for document in documents:
        if hasattr(document, "metadata"):
            metadata = document.metadata or {}
            page_content = getattr(document, "page_content", None)
        elif isinstance(document, dict):
            metadata = document.get("metadata", {}) or {}
            page_content = document.get("page_content")
        else:
            metadata = {}
            page_content = None

        # Flatten nested metadata (mirrors Streamlit behaviour)
        if isinstance(metadata, dict) and isinstance(metadata.get("metadata"), dict):
            metadata = metadata["metadata"]

        serialized.append(
            {
                "page_content": page_content,
                "metadata": metadata,
            }
        )
    return serialized


def _normalize_usage(raw: Any) -> UsageInfo:
    if not raw or not isinstance(raw, dict):
        return UsageInfo()
    return UsageInfo(
        prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
        completion_tokens=int(raw.get("completion_tokens", 0) or 0),
        total_tokens=int(raw.get("total_tokens", 0) or 0),
        cost=float(raw.get("cost", 0.0) or 0.0),
        reasoning_tokens=int(raw.get("reasoning_tokens", 0) or 0),
        cached_tokens=int(raw.get("cached_tokens", 0) or 0),
    )


# ============================================================
# ROUTES
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["health"],
    summary="Health check (for cronjob / uptime monitoring)",
)
def health() -> HealthResponse:
    """
    Lightweight health endpoint. Always returns 200 OK if the
    process is alive. Does not depend on downstream services.
    """
    return HealthResponse(status="ok")


@app.get("/", tags=["info"])
def root() -> Dict[str, Any]:
    return {
        "service": "Ganesh Tattvagyan RAG API",
        "version": "1.0.0",
        "initialized": _initialized,
        "endpoints": {
            "health": "GET /health",
            "query": "POST /query",
            "docs": "GET /docs",
        },
    }


@app.get("/status", tags=["info"])
def status_endpoint() -> Dict[str, Any]:
    """Deeper status check (reports init state)."""
    return {
        "initialized": _initialized,
        "error": _init_error,
    }


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["rag"],
    summary="Run the RAG pipeline",
)
def query(payload: QueryRequest) -> QueryResponse:
    if not payload.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`question` must be a non-empty string.",
        )

    if not _initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "RAG system is not initialized. "
                f"{_init_error or 'Unknown initialization error.'}"
            ),
        )

    try:
        result = graph.invoke(
            {
                "question": payload.question,
                "rewrite_count": payload.rewrite_count,
            }
        )
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {e}",
        )

    retrieval = result.get("retrieval", []) or []
    documents = _serialize_documents(result.get("documents", []))

    # Normalize errors list (may contain non-str objects)
    raw_errors = result.get("errors", []) or []
    errors = [str(e) for e in raw_errors]

    return QueryResponse(
        answer=result.get("answer", "No answer generated."),
        model=result.get("model"),
        usage=_normalize_usage(result.get("usage")),
        retrieval=retrieval,
        documents=documents,
        context=result.get("context"),
        errors=errors,
    )


# ============================================================
# OPTIONAL: CACHE RESET
# ============================================================

@app.post("/reset-cache", tags=["admin"])
def reset_cache() -> Dict[str, str]:
    """Clear cached singletons and re-initialize the RAG system."""
    global _embedder, _vector_store, _initialized, _init_error
    _embedder = None
    _vector_store = None
    _initialized = False
    _init_error = None
    ok = initialize_system()
    if not ok:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": _init_error or "unknown"},
        )
    return {"status": "ok", "message": "Cache cleared and system re-initialized."}


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="localhost",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
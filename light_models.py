"""
light_models.py - memory-light replacements for sentence-transformers / CrossEncoder.

Why: `import sentence_transformers` pulls in torch (hundreds of MB of RSS before any
model is loaded) and BAAI/bge-reranker-v2-m3 alone is ~2 GB of weights. Neither fits
in a 512 MB container.

  * LightEmbedder  - all-MiniLM-L6-v2 via ONNX Runtime (fastembed). Same model as before,
                     so the vectors already stored in Astra DB stay valid (no re-ingestion).
  * build_reranker - picks a backend from env vars:
        RERANKER_BACKEND=jina    -> Jina hosted reranker  (needs JINA_API_KEY)
        RERANKER_BACKEND=cohere  -> Cohere hosted reranker (needs COHERE_API_KEY)
        RERANKER_BACKEND=local   -> small ONNX cross-encoder (ms-marco-MiniLM-L-6, ~80 MB)
    If RERANKER_BACKEND is unset: jina if JINA_API_KEY, else cohere if COHERE_API_KEY,
    else local.

All rerankers expose:  score(query, texts) -> list[float]  (higher = more relevant, 0..1)
"""

import math
import os
import logging
from typing import List
import dotenv

import requests
from requests.adapters import HTTPAdapter

dotenv.load_dotenv(override=True)

log = logging.getLogger("ganesh-rag.models")

# Models are cached inside the project dir so a build step can pre-download them
# (Render's default temp dir is wiped on every restart/deploy).
MODEL_CACHE_DIR = os.getenv(
    "MODEL_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache"),
)
ONNX_THREADS = int(os.getenv("ONNX_THREADS", "1"))


# ============================================================
# EMBEDDINGS
# ============================================================

class LightEmbedder:
    """Duck-typed LangChain Embeddings (embed_query / embed_documents)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=MODEL_CACHE_DIR,
            threads=ONNX_THREADS,
        )

    def embed_query(self, text: str) -> List[float]:
        return next(iter(self.model.embed([text]))).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self.model.embed(list(texts), batch_size=32)]


# ============================================================
# RERANKERS
# ============================================================

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class LocalReranker:
    """Small ONNX cross-encoder. English, MS-MARCO trained; ~80 MB on disk."""

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(
            model_name=model_name,
            cache_dir=MODEL_CACHE_DIR,
            threads=ONNX_THREADS,
        )

    def score(self, query: str, texts: List[str]) -> List[float]:
        # raw logits -> 0..1 so thresholds behave like the old bge (sigmoid) scores
        return [_sigmoid(float(s)) for s in self.model.rerank(query, texts)]


class ApiReranker:
    """Hosted reranker. Jina and Cohere share the same request/response shape."""

    PROVIDERS = {
        "jina": (
            "https://api.jina.ai/v1/rerank",
            "JINA_API_KEY",
            "jina-reranker-v2-base-multilingual",
        ),
        "cohere": (
            "https://api.cohere.com/v2/rerank",
            "COHERE_API_KEY",
            "rerank-v3.5",
        ),
    }

    def __init__(self, provider: str, model: str = None, timeout: int = 10, max_chars: int = 2000):
        url, key_env, default_model = self.PROVIDERS[provider]
        key = os.getenv(key_env)
        if not key:
            raise ValueError(f"{key_env} is not set (RERANKER_BACKEND={provider}).")

        self.url = url
        self.model = model or default_model
        self.timeout = timeout
        self.max_chars = max_chars
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=8))

    def score(self, query: str, texts: List[str]) -> List[float]:
        payload = {
            "model": self.model,
            "query": query,
            "documents": [t[: self.max_chars] for t in texts],
            "top_n": len(texts),
        }
        r = self.session.post(self.url, headers=self.headers, json=payload, timeout=self.timeout)
        r.raise_for_status()

        scores = [0.0] * len(texts)
        for item in r.json()["results"]:
            scores[item["index"]] = float(item["relevance_score"])
        return scores


def build_reranker():
    backend = os.getenv("RERANKER_BACKEND", "").strip().lower()

    if not backend:
        if os.getenv("JINA_API_KEY"):
            backend = "jina"
        elif os.getenv("COHERE_API_KEY"):
            backend = "cohere"
        else:
            backend = "local"

    model_override = os.getenv("RERANKER_MODEL") or None
    log.info("Reranker backend: %s", backend)

    if backend in ("jina", "cohere"):
        return ApiReranker(backend, model=model_override)

    return LocalReranker(model_override or "Xenova/ms-marco-MiniLM-L-6-v2")
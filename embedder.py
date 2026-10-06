"""
Lightweight serving-time embedder.

Same class name and public methods as before, so retriever.py / LangChain / any
scripts keep working - but backed by fastembed (ONNX Runtime) instead of
sentence-transformers + PyTorch.

Why: `import torch` + sentence-transformers alone uses a few hundred MB of RAM,
which blows a 512 MB Render instance. all-MiniLM-L6-v2 via ONNX is ~90 MB on disk
and produces the same 384-d normalised vectors, so your existing Astra DB
collections do NOT need to be re-embedded.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastembed import TextEmbedding


class SentenceTransformerEmbeddings:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device=None,                       # accepted for backward compatibility, ignored (CPU/ONNX)
        cache_dir: Optional[str] = None,
        threads: int = 1,                  # 1 thread = lower RAM and fine for short queries
    ):
        if "/" not in model_name:
            model_name = f"sentence-transformers/{model_name}"

        self.model_name = model_name
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir or os.getenv("FASTEMBED_CACHE_DIR"),
            threads=threads,
        )

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------
    def _encode(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        return [
            np.asarray(v, dtype=np.float32)
            for v in self.model.embed(list(texts), batch_size=batch_size)
        ]

    # ------------------------------------------------------------------
    # LangChain Embeddings interface (used by AstraDBVectorStore)
    # ------------------------------------------------------------------
    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self._encode(texts)]

    # ------------------------------------------------------------------
    # Original helper methods (kept so other scripts don't break)
    # ------------------------------------------------------------------
    def embed_single_text(self, text: str) -> np.ndarray:
        return self._encode([text])[0]

    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> List[np.ndarray]:
        return self._encode(texts, batch_size=batch_size)

    def embed_chunks_in_batches(self, chunks, batch_size: int = 32, show_progress: bool = False) -> List[np.ndarray]:
        texts = [chunk.page_content for chunk in chunks]
        return self._encode(texts, batch_size=batch_size)

    def create_embeddings_with_metadata(self, chunks, embeddings_list: List[np.ndarray]) -> List[Dict[str, Any]]:
        return [
            {"text": c.page_content, "embedding": e, "metadata": c.metadata}
            for c, e in zip(chunks, embeddings_list)
            if e is not None
        ]

    def get_embedding_dimension(self) -> int:
        return len(self._encode(["dimension probe"])[0])

    def save_embeddings(self, embedded_documents: List[Dict[str, Any]], filepath: str = "embeddings.npy"):
        np.save(filepath, np.array([d["embedding"] for d in embedded_documents]))

    def load_embeddings(self, filepath: str) -> np.ndarray:
        return np.load(filepath)
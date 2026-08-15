from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional, Dict, Any
from tqdm import tqdm

class SentenceTransformerEmbeddings(Embeddings):
    """
    A class to handle embeddings using Sentence Transformers.
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        self.model = SentenceTransformer(
            model_name,
            device=device
        )
    
    def embed_chunks_in_batches(self, chunks, batch_size: int = 32, show_progress: bool = True) -> List[np.ndarray]:
        """
        Embed chunks in batches.
        
        Args:
            chunks: List of document chunks with page_content attribute
            batch_size: Number of chunks per batch
            show_progress: Whether to show progress bar
        
        Returns:
            List of embedding vectors
        """
        # Extract texts from chunks
        texts = [chunk.page_content for chunk in chunks]
        
        # Embed in batches
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Convert to list of numpy arrays
        embeddings_list = [emb for emb in embeddings]
        
        print(f"Generated embeddings for {len(embeddings_list)} chunks")
        print(f"Embedding shape: {embeddings.shape}")
        
        return embeddings_list
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[np.ndarray]:
        """
        Embed a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts per batch
            show_progress: Whether to show progress bar
        
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return [emb for emb in embeddings]
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Text string to embed
        
        Returns:
            Embedding vector
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def create_embeddings_with_metadata(self, chunks, embeddings_list: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Create a list of dictionaries with text, embeddings, and metadata.
        
        Args:
            chunks: List of document chunks
            embeddings_list: List of embedding vectors
        
        Returns:
            List of dictionaries containing text, embedding, and metadata
        """
        embedded_docs = []
        
        for chunk, embedding in zip(chunks, embeddings_list):
            if embedding is not None:
                embedded_docs.append({
                    "text": chunk.page_content,
                    "embedding": embedding,
                    "metadata": chunk.metadata
                })
        
        print(f"Created {len(embedded_docs)} embedded documents with metadata")
        return embedded_docs
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
    
    def save_embeddings(self, embedded_documents: List[Dict[str, Any]], filepath: str = "embeddings.npy"):
        """
        Save embeddings to a numpy file.
        
        Args:
            embedded_documents: List of embedded documents
            filepath: Path to save the embeddings
        """
        embeddings = [doc["embedding"] for doc in embedded_documents]
        embeddings_array = np.array(embeddings)
        np.save(filepath, embeddings_array)
        print(f"Saved {len(embeddings)} embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """
        Load embeddings from a numpy file.
        
        Args:
            filepath: Path to the embeddings file
        
        Returns:
            Numpy array of embeddings
        """
        return np.load(filepath)
    
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return np.asarray(embeddings).tolist()
    
    def embed_query(self, text):
        embedding = self.model.encode(text)
        return np.asarray(embedding).tolist()
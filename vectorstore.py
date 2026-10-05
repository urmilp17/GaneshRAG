import uuid
from typing import List, Dict, Any, Optional
from astrapy import DataAPIClient
from langchain_core.documents import Document
import os
import numpy as np

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\\Users\\Pawaskar\Desktop\\Jenkins\\Research_Project\\Model\\ganeshrag-3dbd797255c6.json"

class VectorStore:
    """
    A class to handle vector storage operations using Google Firestore.
    """
    
    def __init__(
        self,
        token,
        endpoint,
        collection_name,
        embedder
    ):
        """
        Initialize the VectorStore with Firestore collection and embedder.
        
        Args:
            collection_name: Name of the Firestore collection
            embedder: SentenceTransformerEmbedder instance
        """
        self.embedder = embedder

        client = DataAPIClient(token)

        self.database = client.get_database(endpoint)

        self.collection = self.database.get_collection(collection_name)
               
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects
            ids: Optional list of IDs for the documents
        
        Returns:
            List of document IDs
        """
        if not self.vector_store:
            self.initialize_vector_store()
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        elif len(ids) != len(documents):
            raise ValueError("Number of IDs must match number of documents")
        
        # Add documents to vector store
        doc_ids = self.vector_store.add_documents(documents, ids=ids)
        print(f"Added {len(doc_ids)} documents to vector store")
        
        return doc_ids
    
    def add_texts(self, texts, metadatas=None, ids=None):

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        documents = []

        embeddings = self.embedder.embed_documents(texts)

        for text, metadata, embedding, doc_id in zip(
            texts,
            metadatas,
            embeddings,
            ids
        ):

            documents.append(
                {
                    "_id": doc_id,
                    "$vector": embedding,
                    "content": text,
                    "metadata": metadata
                }
            )

        self.collection.insert_many(documents)

        return ids
    
    def add_embedded_documents(self, embedded_documents: List[Dict[str, Any]], 
                               ids: Optional[List[str]] = None) -> List[str]:
        """
        Add already embedded documents to the vector store.
        
        Args:
            embedded_documents: List of dictionaries containing text, embedding, and metadata
            ids: Optional list of IDs for the documents
        
        Returns:
            List of document IDs
        """
        if not self.vector_store:
            self.initialize_vector_store()
        
        # Extract texts and metadata
        texts = [doc["text"] for doc in embedded_documents]
        metadatas = [doc["metadata"] for doc in embedded_documents]
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        elif len(ids) != len(texts):
            raise ValueError("Number of IDs must match number of documents")
        
        # Add to vector store
        doc_ids = self.vector_store.add_texts(texts, metadatas=metadatas, ids=ids)
        print(f"Added {len(doc_ids)} embedded documents to vector store")
        
        return doc_ids
    
    def similarity_search(
        self,
        query_embedding,
        k=5
    ):

        return [
            doc
            for doc, score
            in self.similarity_search_with_score(
                query_embedding,
                k
            )
        ]
        
    def similarity_search_with_score(
        self,
        query_embedding,
        k=5
    ):

        cursor = self.collection.find(
            sort={
                "$vector": query_embedding
            },
            limit=k,
            include_similarity=True
        )

        results = []

        for doc in cursor:

            document = Document(
                page_content=doc["content"],
                metadata=doc["metadata"]
            )

            score = doc.get("$similarity", 0)

            results.append(
                (
                    document,
                    score
                )
            )

        return results
    
    def delete_documents(self, ids):
        for doc_id in ids:
            self.collection.delete_one(
                {
                    "_id": doc_id
                }
            )
    
    def get_document_count(self):
        return self.collection.count_documents({})
    
    def set_embedder(self, embedder):
        """
        Set or update the embedder.
        
        Args:
            embedder: SentenceTransformerEmbedder instance
        """
        self.embedder = embedder
        self.initialize_vector_store()
    
    def clear_collection(self):
        self.collection.delete_many({})
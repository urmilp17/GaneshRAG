from typing import List, Dict, Any, Optional
import numpy as np

class RAGRetriever:
    """
    A class to handle retrieval for RAG (Retrieval-Augmented Generation) systems.
    """
    
    def __init__(self, vector_store: VectorStore, embedding_manager):
        """
        Initialize the RAGRetriever.
        
        Args:
            vector_store: VectorStore instance for document retrieval
            embedding_manager: SentenceTransformerEmbedder instance for query embedding
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
           
    def retriever(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on the query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold (0.0 to 1.0)
        
        Returns:
            List of dictionaries containing retrieved documents with metadata
        """
        print(f"\n{'='*60}")
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        print(f"{'='*60}")
        
        try:
            # Generate query embedding
            if hasattr(self.embedding_manager, 'embed_single_text'):
                query_embedding = self.embedding_manager.embed_single_text(query)
                if isinstance(query_embedding, np.ndarray):
                    query_embedding = query_embedding.tolist()
            else:
                query_embedding = self.embedding_manager.model.encode(query)
            
            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(query_embedding, k=top_k)
            
            retrieved_docs = []
            
            # Process results
            if results and isinstance(results, list):
                for rank, result in enumerate(results, 1):
                    if isinstance(result, tuple):
                        doc, score = result
                        
                        # Extract metadata - handle both cases where metadata is in doc.metadata or as separate field
                        metadata = {}

                        if hasattr(doc, "metadata") and doc.metadata:
                            metadata = doc.metadata

                            # FirestoreVectorStore stores user metadata under the "metadata" key
                            if isinstance(metadata, dict) and "metadata" in metadata:
                                metadata = metadata["metadata"]

                            # If metadata is stored as a JSON string
                            if isinstance(metadata, str):
                                try:
                                    import json
                                    metadata = json.loads(metadata)

                                    if isinstance(metadata, dict) and "metadata" in metadata:
                                        metadata = metadata["metadata"]

                                except Exception:
                                    metadata = {}
                        
                        # print("\n===== RAW DOCUMENT METADATA =====")
                        # print(doc.metadata)
                        # print(type(doc.metadata))
                        
                        # Get document ID
                        doc_id = getattr(doc, "id", None)

                        if not doc_id:
                            doc_id = getattr(doc, "document_id", None)

                        if not doc_id:
                            doc_id = metadata.get("id")

                        if not doc_id:
                            doc_id = metadata.get("document_id")

                        if not doc_id:
                            doc_id = f"doc_{rank}"
                        
                        # Convert score to similarity (if it's distance)
                        similarity_score = score
                        
                        if similarity_score >= score_threshold:
                            retrieved_docs.append({
                                'id': doc_id,
                                'content': doc.page_content,
                                'metadata': metadata,
                                'similarity_score': similarity_score,
                                'distance': 1 - similarity_score if similarity_score <= 1 else 0,
                                'rank': rank
                            })
                    else:
                        # If result is just a Document without score
                        doc = result
                        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                        doc_id = getattr(doc, 'id', f"doc_{rank}")
                        
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': doc.page_content,
                            'metadata': metadata,
                            'similarity_score': 1.0 - (rank - 1) * 0.1,
                            'distance': (rank - 1) * 0.1,
                            'rank': rank
                        })
            
            print(f"Retrieved {len(retrieved_docs)} documents after filtering")
            
            # Display summary
            if retrieved_docs:
                print(f"\nTop {min(3, len(retrieved_docs))} results:")
                for i, doc in enumerate(retrieved_docs[:3], 1):
                    book = doc['metadata'].get('book', 'Unknown')
                    chapter = doc['metadata'].get('chapter_info', 'Unknown')
                    score = doc['similarity_score']
                    print(f"  {i}. Book: {book}, Chapter: {chapter}, Score: {score:.4f}")
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def retrieve_with_context(self, query: str, top_k: int = 5, 
                              score_threshold: float = 0.0,
                              include_context: bool = True,
                              context_chars: int = 200) -> Dict[str, Any]:
        """
        Retrieve documents with additional context.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            include_context: Whether to include surrounding context
            context_chars: Number of characters to include around the match
        
        Returns:
            Dictionary with query, retrieved documents, and context
        """
        # Get retrieved documents
        retrieved_docs = self.retriever(query, top_k, score_threshold)
        
        # Prepare response
        response = {
            'query': query,
            'retrieved_documents': retrieved_docs,
            'num_results': len(retrieved_docs)
        }
        
        # Add context if requested
        if include_context and retrieved_docs:
            contexts = []
            for doc in retrieved_docs:
                content = doc['content']
                # Extract context around the content
                if len(content) > context_chars * 2:
                    # Take middle section or relevant part
                    start_idx = max(0, (len(content) - context_chars * 2) // 2)
                    context = content[start_idx:start_idx + context_chars * 2]
                    contexts.append({
                        'context': context,
                        'metadata': doc['metadata'],
                        'score': doc['similarity_score']
                    })
                else:
                    contexts.append({
                        'context': content,
                        'metadata': doc['metadata'],
                        'score': doc['similarity_score']
                    })
            response['contexts'] = contexts
        
        return response
    
    def format_for_llm(self, retrieved_docs: List[Dict[str, Any]], 
                    max_chars_per_doc: int = 500) -> str:
        """
        Format retrieved documents for LLM input.
        
        Args:
            retrieved_docs: List of retrieved documents
            max_chars_per_doc: Maximum characters per document
        
        Returns:
            Formatted string for LLM context
        """
        if not retrieved_docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc['content'][:max_chars_per_doc]
            if len(doc['content']) > max_chars_per_doc:
                content += "..."
            
            metadata = doc.get('metadata', {})
            book = metadata.get('book', 'Unknown')
            chapter = metadata.get('chapter_info', 'Unknown')
            score = doc.get('similarity_score', 0.0)
            doc_id = doc.get('id', 'Unknown')
            
            context_parts.append(
                f"[Document {i}] ID: {doc_id}, Book: {book}, Chapter: {chapter} (Relevance: {score:.3f})\n"
                f"Content: {content}\n"
            )
        
        return "\n".join(context_parts)
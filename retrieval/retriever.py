import os

from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

from langchain_astradb import AstraDBVectorStore

from embedder import SentenceTransformerEmbeddings


load_dotenv(override=True)


class GaneshRetriever:

    def __init__(
        self,
        collection_name="puranas",
        retrieve_k=15,
        top_k=5,
        reranker_model="BAAI/bge-reranker-v2-m3"
    ):

        self.retrieve_k = retrieve_k

        self.top_k = top_k

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        self.embedder = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2",
            device=None
        )

        # ----------------------------------------------------
        # AstraDB
        # ----------------------------------------------------

        self.vector_store = AstraDBVectorStore(

            collection_name=collection_name,

            embedding=self.embedder,

            token=os.getenv(
                "ASTRA_DB_APPLICATION_TOKEN"
            ),

            api_endpoint=os.getenv(
                "ASTRA_DB_API_ENDPOINT"
            )
        )

        # ----------------------------------------------------
        # Cross Encoder
        # ----------------------------------------------------

        self.reranker = CrossEncoder(
            reranker_model
        )


    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query: str
    ):

        results = (
            self.vector_store
            .similarity_search_with_score(
                query,
                k=self.retrieve_k
            )
        )

        candidates = []

        for rank, result in enumerate(
            results,
            start=1
        ):

            document = result[0]

            vector_score = (
                result[1]
                if len(result) > 1
                else None
            )

            metadata = (
                document.metadata
                or {}
            )

            text = (
                document.page_content
                or ""
            )

            if not text.strip():
                continue

            candidates.append({

                "document": document,

                "text": text,

                "metadata": metadata,

                "vector_score": vector_score,

                "vector_rank": rank
            })


        if not candidates:
            return []


        # ====================================================
        # RERANK
        # ====================================================

        pairs = [
            (
                query,
                candidate["text"]
            )
            for candidate in candidates
        ]


        scores = self.reranker.predict(
            pairs,
            show_progress_bar=False
        )


        for candidate, score in zip(
            candidates,
            scores
        ):

            candidate[
                "reranker_score"
            ] = float(score)


        candidates.sort(
            key=lambda x:
                x["reranker_score"],
            reverse=True
        )


        return candidates[
            :self.top_k
        ]
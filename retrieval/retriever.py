import os

from dotenv import load_dotenv

from sentence_transformers import CrossEncoder

from langchain_astradb import AstraDBVectorStore

from embedder import SentenceTransformerEmbeddings


load_dotenv(override=True)


class GaneshRetriever:

    def __init__(
        self,
        puranas_collection="puranas",
        research_collection="research",
        retrieve_k=10,
        top_k=6,
        reranker_model="BAAI/bge-reranker-v2-m3"
    ):

        self.retrieve_k = retrieve_k
        self.top_k = top_k


        # =====================================================
        # EMBEDDING MODEL
        # =====================================================

        self.embedder = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2",
            device=None
        )


        # =====================================================
        # PURANAS VECTOR STORE
        # =====================================================

        self.puranas_vector_store = AstraDBVectorStore(

            collection_name=puranas_collection,

            embedding=self.embedder,

            token=os.getenv(
                "ASTRA_DB_APPLICATION_TOKEN"
            ),

            api_endpoint=os.getenv(
                "ASTRA_DB_API_ENDPOINT"
            )
        )


        # =====================================================
        # RESEARCH VECTOR STORE
        # =====================================================

        self.research_vector_store = AstraDBVectorStore(

            collection_name=research_collection,

            embedding=self.embedder,

            token=os.getenv(
                "ASTRA_DB_APPLICATION_TOKEN"
            ),

            api_endpoint=os.getenv(
                "ASTRA_DB_API_ENDPOINT"
            )
        )


        # =====================================================
        # CROSS ENCODER RERANKER
        # =====================================================

        self.reranker = CrossEncoder(
            reranker_model
        )


    # =========================================================
    # RETRIEVE FROM SINGLE COLLECTION
    # =========================================================

    def retrieve_from_collection(
        self,
        vector_store,
        query,
        collection_name
    ):

        try:

            results = (
                vector_store
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

                vector_score = result[1]

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


                # Add collection information
                # This is useful later for debugging
                # and source authority handling

                metadata["collection"] = collection_name

                document.metadata = metadata


                candidates.append({

                    "document": document,

                    "text": text,

                    "metadata": metadata,

                    "vector_score": vector_score,

                    "vector_rank": rank,

                    "collection": collection_name
                })


            return candidates


        except Exception as e:

            print(
                f"Error retrieving from "
                f"{collection_name}: {str(e)}"
            )

            return []


    # =========================================================
    # RETRIEVE FROM ALL COLLECTIONS
    # =========================================================
    def get_authority_score(
        self,
        metadata
    ):

        source_type = (
            metadata
            .get(
                "source_type",
                ""
            )
            .lower()
        )


        authority_scores = {

            "purana": 1.0,

            "upanishad": 0.95,

            "traditional_text": 0.90,

            "sahasranama": 0.85,

            "iconography": 0.70,

            "research": 0.60,

            "etic": 0.50
        }


        return authority_scores.get(
            source_type,
            0.40
        )
    
    def retrieve(
        self,
        query: str
    ):


        # -----------------------------------------------------
        # 1. Retrieve from PURANAS
        # -----------------------------------------------------

        puranas_candidates = (
            self.retrieve_from_collection(

                vector_store=
                    self.puranas_vector_store,

                query=query,

                collection_name="puranas"
            )
        )


        # -----------------------------------------------------
        # 2. Retrieve from RESEARCH
        # -----------------------------------------------------

        research_candidates = (
            self.retrieve_from_collection(

                vector_store=
                    self.research_vector_store,

                query=query,

                collection_name="research"
            )
        )


        # -----------------------------------------------------
        # 3. Combine candidates
        # -----------------------------------------------------

        candidates = (

            puranas_candidates
            +
            research_candidates
        )


        print(
            f"\nRetrieved Candidates:"
        )

        print(
            f"Puranas: "
            f"{len(puranas_candidates)}"
        )

        print(
            f"Research: "
            f"{len(research_candidates)}"
        )

        print(
            f"Total: "
            f"{len(candidates)}"
        )


        if not candidates:

            return []


        # =====================================================
        # 4. CROSS-ENCODER RERANKING
        # =====================================================

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


        # Attach reranker scores

        for candidate, score in zip(
            candidates,
            scores
        ):

            candidate[
                "reranker_score"
            ] = float(score)

            authority_score = (
                self.get_authority_score(
                    candidate["metadata"]
                )
            )

            candidate[
                "authority_score"
            ] = authority_score


        # =====================================================
        # 5. SORT BY RERANKER SCORE
        # =====================================================

        candidates.sort(

            key=lambda x:
                x["reranker_score"],

            reverse=True
        )


        # =====================================================
        # 6. FINAL TOP K
        # =====================================================

        final_candidates = candidates[
            :self.top_k
        ]


        # =====================================================
        # DEBUG OUTPUT
        # =====================================================

        print(
            "\nFinal Reranked Results:"
        )


        for index, candidate in enumerate(

            final_candidates,

            start=1

        ):

            print(
                f"{index}. "
                f"[{candidate['collection']}] "
                f"{candidate['metadata'].get('source', 'Unknown')} "
                f"| Reranker: "
                f"{candidate['reranker_score']:.4f}"
            )


        return final_candidates
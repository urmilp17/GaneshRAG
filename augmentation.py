import os
import requests
import dotenv

from sentence_transformers import CrossEncoder


# ============================================================
# ENVIRONMENT
# ============================================================

dotenv.load_dotenv(override=True)


# ============================================================
# AUGMENTATION
# ============================================================

class Augmentation:

    def __init__(
        self,
        vector_store=None,
        api_key=None,
        models=None,
        reranker_model=None
    ):
        """
        RAG augmentation layer.

        Pipeline:

            User Query
                |
                v
            AstraDB Vector Search
                |
                | retrieve_k candidates
                v
            Cross-Encoder Reranker
                |
                | top_k reranked chunks
                v
            Context Construction
                |
                v
            OpenRouter LLM
                |
                v
            Final Answer

        `retrieve_k` should normally be larger than `top_k`.
        Example:

            retrieve_k = 15
            top_k       = 5
        """

        self.vector_store = vector_store

        # ----------------------------------------------------
        # OpenRouter API key
        # ----------------------------------------------------

        self.api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file."
            )

        # ----------------------------------------------------
        # OpenRouter fallback models
        # ----------------------------------------------------

        self.models = models or [
            "nvidia/nemotron-3.5-lightning",
            "liquid/lfm-2.5-2.6b",
            "inclusionai/ling-3.0-tiny"
        ]

        self.url = (
            "https://openrouter.ai/api/v1/chat/completions"
        )

        # ----------------------------------------------------
        # Cross-Encoder reranker
        # ----------------------------------------------------
        #
        # BGE reranker v2 m3 is multilingual and is a good
        # choice for a scripture-focused RAG where queries
        # may contain Sanskrit/transliterated terms.
        #
        # You can override it using:
        #
        # RERANKER_MODEL=...
        #
        # in .env
        # ----------------------------------------------------

        self.reranker_model_name = (
            reranker_model
            or os.getenv(
                "RERANKER_MODEL",
                "BAAI/bge-reranker-v2-m3"
            )
        )

        print(
            f"Loading reranker: "
            f"{self.reranker_model_name}"
        )

        self.reranker = CrossEncoder(
            self.reranker_model_name
        )

        print("Reranker loaded successfully.")


    # ========================================================
    # AUGMENT
    # ========================================================

    def augment(
        self,
        query: str,
        top_k: int = 5,
        retrieve_k: int = 15,
        temperature: float = 0.2
    ):
        """
        Retrieve, rerank and generate an answer.

        Args:
            query:
                User question.

            top_k:
                Number of final chunks passed to the LLM.

            retrieve_k:
                Number of candidates initially retrieved from
                AstraDB before reranking.

                Recommended:
                    10-20

            temperature:
                LLM generation temperature.

        Returns:
            Dictionary containing:

                query
                answer
                context
                documents
                model
                errors
        """

        if not query or not query.strip():
            return {
                "query": query,
                "answer": "Please provide a question.",
                "context": "",
                "documents": [],
                "model": None,
                "errors": []
            }

        if self.vector_store is None:
            raise ValueError(
                "vector_store is not initialized."
            )

        if retrieve_k < top_k:
            retrieve_k = top_k

        # ====================================================
        # 1. INITIAL VECTOR RETRIEVAL
        # ====================================================

        print(
            f"\nVector retrieval: top {retrieve_k}"
        )

        retrieved_docs = (
            self.vector_store
            .similarity_search_with_score(
                query,
                retrieve_k
            )
        )

        # ====================================================
        # 2. NO RESULTS
        # ====================================================

        if not retrieved_docs:
            return {
                "query": query,
                "answer": (
                    "I could not find any relevant information "
                    "in the provided sources."
                ),
                "context": "",
                "documents": [],
                "model": None,
                "errors": []
            }

        print(
            f"Retrieved {len(retrieved_docs)} "
            f"candidate chunks."
        )

        # ====================================================
        # 3. NORMALIZE RETRIEVED DOCUMENTS
        # ====================================================

        candidates = []

        for original_rank, item in enumerate(
            retrieved_docs,
            start=1
        ):
            document, vector_score = (
                self._unpack_result(item)
            )

            metadata = self._get_metadata(
                document
            )

            text = self._get_text(
                document
            )

            if not text.strip():
                continue

            candidates.append({
                "document": document,
                "metadata": metadata,
                "text": text,
                "vector_score": vector_score,
                "vector_rank": original_rank
            })

        if not candidates:
            return {
                "query": query,
                "answer": (
                    "I could not find any usable text "
                    "in the retrieved sources."
                ),
                "context": "",
                "documents": [],
                "model": None,
                "errors": []
            }

        # ====================================================
        # 4. CROSS-ENCODER RERANKING
        # ====================================================

        print(
            f"Reranking {len(candidates)} candidates..."
        )

        reranker_pairs = [
            (
                query,
                candidate["text"]
            )
            for candidate in candidates
        ]

        reranker_scores = (
            self.reranker.predict(
                reranker_pairs,
                show_progress_bar=False
            )
        )

        for candidate, reranker_score in zip(
            candidates,
            reranker_scores
        ):
            candidate["reranker_score"] = (
                float(reranker_score)
            )

        # Highest Cross-Encoder score first

        candidates.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        # ====================================================
        # 5. KEEP TOP-K RERANKED DOCUMENTS
        # ====================================================

        reranked_candidates = candidates[
            :top_k
        ]

        print(
            f"Selected top {len(reranked_candidates)} "
            f"reranked chunks."
        )

        # ====================================================
        # 6. BUILD CONTEXT
        # ====================================================

        context_parts = []

        for i, candidate in enumerate(
            reranked_candidates,
            start=1
        ):
            metadata = candidate["metadata"]
            text = candidate["text"]

            citation = self._format_citation(
                metadata
            )

            source_block = f"""
==========================
SOURCE {i}
==========================

Source Type:
{metadata.get("source_type", "unknown")}

Source:
{metadata.get("source", "Unknown Source")}

Book Title:
{metadata.get("book_title", "Not specified")}

Authority:
{metadata.get("authority", "unknown")}

Tradition:
{metadata.get("tradition", "unknown")}

Language:
{metadata.get("language", "unknown")}

Section:
{metadata.get("section", "Not specified")}

Chapter:
{metadata.get("chapter", "Not specified")}

Chapter Number:
{metadata.get("chapter_number", "Not specified")}

Chapter Title:
{metadata.get("chapter_title", "Not specified")}

Page Number:
{metadata.get("page_number", "Not specified")}

Page Label:
{metadata.get("page_label", "Not specified")}

Chunk ID:
{metadata.get("chunk_id", f"source_{i}")}

Chunk Index:
{metadata.get("chunk_index", "Not specified")}

Vector Retrieval Rank:
{candidate["vector_rank"]}

Vector Retrieval Score:
{candidate["vector_score"] if candidate["vector_score"] is not None else "Not available"}

Reranker Score:
{candidate["reranker_score"]:.6f}

Citation:
{citation}

Content:
{text}

--------------------------
"""

            context_parts.append(
                source_block
            )

        context = "\n".join(
            context_parts
        )

        # ====================================================
        # 7. PROMPT
        # ====================================================

        prompt = f"""
You are an expert scholar of Hindu scriptures and Ganapatya
literature.

Your task is to answer the user's question ONLY using the supplied
retrieved context.

The retrieved context has been selected using semantic vector
retrieval followed by a Cross-Encoder reranker. The reranker score
indicates relevance to the user's question, but it does NOT
indicate scriptural authority.

The retrieved sources may include:

- Primary scriptures
- Upanishadic texts
- Traditional commentaries
- Secondary scholarly research
- Iconographic descriptions
- Collections of names and meanings
- Other specialized Ganesh-related texts

Pay close attention to the metadata.

A research book is secondary scholarship and must not be presented
as though it were a primary scriptural statement.

If multiple sources are relevant, synthesize them carefully while
preserving their source distinctions.

If the retrieved context is insufficient, say so explicitly.

If the retrieved context does not contain the answer, respond:

"I could not find the answer in the provided sources."

Do not use external knowledge to fill missing information.

============================================================
CONTEXT
============================================================

{context}

============================================================
QUESTION
============================================================

{query}

============================================================
INSTRUCTIONS
============================================================

1. Answer ONLY from the supplied context.

2. Do NOT use external knowledge, prior knowledge, or assumptions.

3. Do not invent scriptural details.

4. If the evidence is incomplete, clearly state the limitation.

5. Write a scholarly, coherent narrative.

6. Prefer complete paragraphs over bullet points.

7. Use bullet points or numbered lists only when the question
   specifically requires a list.

8. Explain narrative context, symbolism, philosophical meaning,
   and spiritual significance only when supported by the retrieved
   sources.

9. When multiple sources provide relevant information, synthesize
   them while preserving their source distinctions.

10. Clearly distinguish between:

    - Scriptural statements
    - Traditional interpretations
    - Scholarly interpretations
    - Interpretive conclusions

11. Preserve Sanskrit names, terms and transliterations.

12. Do not claim that a statement comes from a scripture merely
    because a research source discusses that scripture.

13. Do not repeatedly mention the source in every sentence when
    several consecutive claims come from the same source.

14. Never fabricate a citation.

============================================================
CITATIONS
============================================================

Every significant factual or textual claim derived from the
retrieved context should have an appropriate citation.

Use ONLY metadata supplied in the SOURCE blocks.

For Puranas:

(Ganesh Purana, Krida Khanda, Chapter 41)

For Mudgal Purana:

(Mudgal Purana, Khanda 6, Chapter 43)

For research books with page metadata:

(Book Title, p. 42)

If chapter and page metadata are both available:

(Book Title, Chapter 4, p. 42)

For other sources, adapt the citation to the actual metadata.

IMPORTANT:

- Do NOT invent chapter numbers.
- Do NOT invent page numbers.
- Do NOT invent source names.
- Do NOT fabricate citations.
- Do not cite a source that does not support the statement.

============================================================
SOURCE AUTHORITY
============================================================

When sources differ in authority, prefer:

1. Primary scripture
2. Upanishadic / traditional scripture
3. Traditional commentary
4. Scholarly research
5. Modern interpretive material

However, source relevance and source authority are separate.

A highly relevant research passage must not automatically override
a directly relevant primary-scripture passage.

============================================================
STYLE
============================================================

Write in fluent academic English.

Maintain a neutral, scholarly and respectful tone.

The answer should read like a concise scholarly exposition rather
than a casual chatbot response.

When appropriate, structure the explanation naturally as:

1. Direct answer.
2. Textual or narrative context.
3. Meaning and significance.
4. Philosophical or spiritual interpretation.

Do not force this structure when the retrieved material does not
support it.

============================================================
ANSWER
============================================================
"""

        # ====================================================
        # 8. GENERATE RESPONSE
        # ====================================================

        print(
            "\nGenerating response using OpenRouter...\n"
        )

        headers = {
            "Authorization":
                f"Bearer {self.api_key}",

            "Content-Type":
                "application/json"
        }

        response = None
        successful_model = None
        errors = []

        # ====================================================
        # 9. TRY MODELS SEQUENTIALLY
        # ====================================================

        for model in self.models:

            payload = {
                "model": model,

                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                "temperature": temperature
            }

            try:

                print(
                    f"Trying model: {model}"
                )

                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=120
                )

                if response.status_code == 200:

                    successful_model = model
                    break

                error_message = response.text

                errors.append(
                    f"{model}: "
                    f"HTTP {response.status_code} - "
                    f"{error_message}"
                )

                print(
                    f"Model failed: {model} "
                    f"(HTTP {response.status_code})"
                )

            except requests.exceptions.RequestException as e:

                errors.append(
                    f"{model}: {str(e)}"
                )

                print(
                    f"Request failed for "
                    f"{model}: {e}"
                )

        # ====================================================
        # 10. ALL MODELS FAILED
        # ====================================================

        if (
            response is None
            or successful_model is None
        ):
            return {
                "query": query,

                "answer": (
                    "The language model service is "
                    "currently unavailable. "
                    "Please try again later."
                ),

                "context": context,

                "documents": [
                    item["document"]
                    for item in reranked_candidates
                ],

                "model": None,

                "errors": errors
            }

        # ====================================================
        # 11. PARSE RESPONSE
        # ====================================================

        try:

            response_data = response.json()

            answer = (
                response_data[
                    "choices"
                ][0][
                    "message"
                ][
                    "content"
                ]
            )

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError
        ):

            answer = (
                "The language model returned "
                "an unexpected response."
            )

            errors.append(
                response.text
            )

        # ====================================================
        # 12. RETURN RESULT
        # ====================================================

        return {
            "query": query,

            "answer": answer,

            "context": context,

            "documents": [
                item["document"]
                for item in reranked_candidates
            ],

            "model": successful_model,

            "errors": errors,

            # Useful for RAG evaluation/debugging
            "retrieval": [
                {
                    "vector_rank":
                        item["vector_rank"],

                    "vector_score":
                        item["vector_score"],

                    "reranker_score":
                        item["reranker_score"],

                    "source":
                        item["metadata"].get(
                            "source"
                        ),

                    "page_number":
                        item["metadata"].get(
                            "page_number"
                        ),

                    "chunk_id":
                        item["metadata"].get(
                            "chunk_id"
                        )
                }

                for item in reranked_candidates
            ]
        }


    # ========================================================
    # RESULT HELPERS
    # ========================================================

    @staticmethod
    def _unpack_result(item):
        """
        Handle Astra/LangChain retrieval formats.
        """

        if isinstance(item, tuple):

            document = item[0]

            score = (
                item[1]
                if len(item) > 1
                else None
            )

            return document, score

        return item, None


    @staticmethod
    def _get_metadata(document):
        """
        Extract metadata from a LangChain Document or dict.
        """

        if hasattr(
            document,
            "metadata"
        ):

            metadata = (
                document.metadata
                or {}
            )

        elif isinstance(
            document,
            dict
        ):

            metadata = document.get(
                "metadata",
                {}
            )

        else:

            metadata = {}

        # Handle nested metadata

        if (
            isinstance(metadata, dict)
            and
            isinstance(
                metadata.get("metadata"),
                dict
            )
        ):

            metadata = metadata[
                "metadata"
            ]

        return metadata


    @staticmethod
    def _get_text(document):
        """
        Extract text from a LangChain Document or dict.
        """

        if hasattr(
            document,
            "page_content"
        ):

            return (
                document.page_content
                or ""
            )

        if hasattr(
            document,
            "content"
        ):

            return (
                document.content
                or ""
            )

        if isinstance(
            document,
            dict
        ):

            return document.get(
                "page_content",
                document.get(
                    "content",
                    ""
                )
            ) or ""

        return str(document)


    # ========================================================
    # CITATION FORMATTER
    # ========================================================

    @staticmethod
    def _format_citation(metadata):
        """
        Create a citation from the new unified metadata.

        Examples:

            (Ganesh Purana, Krida Khanda, Chapter 41)

            (Mudgal Purana, Khanda 6, Chapter 43)

            (Some Research Book, Chapter 4, p. 42)

            (Some Research Book, p. 42)

            (Vallabhesha Upanishad, Chapter 2)
        """

        source = metadata.get(
            "source",
            "Unknown Source"
        )

        parts = [source]

        section = metadata.get(
            "section"
        )

        chapter_number = metadata.get(
            "chapter_number"
        )

        chapter = metadata.get(
            "chapter"
        )

        page_number = metadata.get(
            "page_number"
        )

        # ----------------------------------------------------
        # Section
        # ----------------------------------------------------

        if section:
            parts.append(
                str(section)
            )

        # ----------------------------------------------------
        # Chapter
        # ----------------------------------------------------

        if chapter_number is not None:

            parts.append(
                f"Chapter {chapter_number}"
            )

        elif chapter:

            parts.append(
                str(chapter)
            )

        # ----------------------------------------------------
        # Page
        # ----------------------------------------------------

        if page_number is not None:

            parts.append(
                f"p. {page_number}"
            )

        return (
            "("
            + ", ".join(parts)
            + ")"
        )

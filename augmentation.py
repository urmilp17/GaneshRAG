import os
import requests
import dotenv


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
        models=None
    ):

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


    # ========================================================
    # AUGMENT
    # ========================================================

    def augment(
        self,
        query: str,
        top_k: int = 4,
        temperature: float = 0.2
    ):
        """
        Retrieve relevant documents from AstraDB and generate
        an answer using OpenRouter.

        Args:
            query:
                User question.

            top_k:
                Number of documents/chunks to retrieve.

            temperature:
                Generation temperature.

        Returns:
            Dictionary containing:

                query
                answer
                context
                documents
                model
                errors
        """

        # ====================================================
        # 1. RETRIEVE DOCUMENTS
        # ====================================================

        retrieved_docs = (
            self.vector_store
            .similarity_search_with_score(
                query,
                top_k
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
                    "in the provided scriptures."
                ),

                "context": "",

                "documents": [],

                "model": None
            }


        # ====================================================
        # 3. BUILD CONTEXT
        # ====================================================

        context_parts = []


        for i, doc in enumerate(
            retrieved_docs,
            1
        ):

            # ------------------------------------------------
            # Default values
            # ------------------------------------------------

            metadata = {}

            text = ""

            score = None


            # =================================================
            # CASE 1:
            # (Document, score)
            # =================================================

            if isinstance(doc, tuple):

                document, score = doc


                # ---------------------------------------------
                # Metadata
                # ---------------------------------------------

                if hasattr(
                    document,
                    "metadata"
                ):

                    metadata = (
                        document.metadata
                        or {}
                    )


                # ---------------------------------------------
                # Handle nested metadata
                # ---------------------------------------------

                if (
                    isinstance(
                        metadata,
                        dict
                    )
                    and
                    "metadata" in metadata
                    and
                    isinstance(
                        metadata["metadata"],
                        dict
                    )
                ):

                    metadata = metadata[
                        "metadata"
                    ]


                # ---------------------------------------------
                # Document content
                # ---------------------------------------------

                if hasattr(
                    document,
                    "page_content"
                ):

                    text = (
                        document.page_content
                    )

                elif hasattr(
                    document,
                    "content"
                ):

                    text = (
                        document.content
                    )

                else:

                    text = str(
                        document
                    )


            # =================================================
            # CASE 2:
            # Dictionary result
            # =================================================

            else:

                if isinstance(
                    doc,
                    dict
                ):

                    metadata = doc.get(
                        "metadata",
                        {}
                    )

                    # -----------------------------------------
                    # Nested metadata
                    # -----------------------------------------

                    if (
                        isinstance(
                            metadata,
                            dict
                        )
                        and
                        "metadata" in metadata
                        and
                        isinstance(
                            metadata["metadata"],
                            dict
                        )
                    ):

                        metadata = (
                            metadata["metadata"]
                        )


                    # -----------------------------------------
                    # Content
                    # -----------------------------------------

                    text = doc.get(
                        "page_content",
                        doc.get(
                            "content",
                            ""
                        )
                    )

                else:

                    text = str(
                        doc
                    )


            # =================================================
            # 4. NEW METADATA STRUCTURE
            # =================================================

            source_type = metadata.get(
                "source_type",
                "unknown"
            )


            source = metadata.get(
                "source",
                "Unknown Source"
            )


            authority = metadata.get(
                "authority",
                "unknown"
            )


            tradition = metadata.get(
                "tradition",
                "unknown"
            )


            language = metadata.get(
                "language",
                "unknown"
            )


            section = metadata.get(
                "section",
                "Unknown Section"
            )


            chapter = metadata.get(
                "chapter",
                "Unknown Chapter"
            )


            chapter_number = metadata.get(
                "chapter_number",
                None
            )


            chapter_title = metadata.get(
                "chapter_title",
                ""
            )


            chunk_id = metadata.get(
                "chunk_id",
                f"source_{i}"
            )


            chunk_index = metadata.get(
                "chunk_index",
                None
            )


            # =================================================
            # 5. CREATE HUMAN-READABLE CITATION
            # =================================================

            citation = self._format_citation(
                source=source,
                section=section,
                chapter=chapter,
                chapter_number=chapter_number,
                chapter_title=chapter_title
            )


            # =================================================
            # 6. BUILD SOURCE BLOCK
            # =================================================

            source_block = f"""
==========================
SOURCE {i}
==========================

Source Type:
{source_type}

Source:
{source}

Authority:
{authority}

Tradition:
{tradition}

Section:
{section}

Chapter:
{chapter}

Chapter Number:
{chapter_number if chapter_number is not None else "Not specified"}

Chapter Title:
{chapter_title if chapter_title else "Not specified"}

Chunk ID:
{chunk_id}

Chunk Index:
{chunk_index if chunk_index is not None else "Not specified"}

Retrieval Score:
{score if score is not None else "Not available"}

Citation:
{citation}

Content:
{text}

--------------------------
"""


            context_parts.append(
                source_block
            )


        # ====================================================
        # 7. COMBINE CONTEXT
        # ====================================================

        context = "\n".join(
            context_parts
        )


        # ====================================================
        # 8. PROMPT
        # ====================================================

        prompt = f"""
You are an expert scholar of Hindu scriptures with deep knowledge of
the Ganesh Purana, Mudgala Purana, Ganapati Atharvashirsha,
Upanishads, Ganapatya traditions, and traditional Sanskrit
commentaries.

Your task is to answer the user's question ONLY using the supplied
context.

The retrieved passages are the sole authoritative evidence for
your answer.

Do NOT use external knowledge, general knowledge, or assumptions
that are not supported by the supplied context.

The retrieved passages may represent different types of sources,
including:

- Primary scriptures
- Upanishadic texts
- Secondary scholarly research
- Iconographic descriptions
- Collections of names and meanings
- Other specialized Ganesh-related texts

Pay attention to the source metadata.

Primary scriptures should be distinguished from secondary
scholarship.

Do not present a modern scholarly interpretation as though it were
an explicit statement from a scripture.

If multiple sources provide relevant information, synthesize them
carefully while preserving their source distinctions.

If the retrieved passages contain only partial information,
provide only what can reasonably be established from those
passages.

If the retrieved passages contain no relevant information, answer:

"I could not find the answer in the provided scriptures."

Do not invent information.

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

4. If the retrieved passages contain relevant but incomplete
   information, explain only what can reasonably be established.

5. Write the answer as a scholarly, coherent narrative rather
   than automatically using bullet points.

6. Use complete paragraphs.

7. Use bullet points or numbered lists only when the question
   specifically requires a list.

8. Explain narrative context, symbolism, philosophical meaning,
   and spiritual significance whenever such information is
   explicitly supported by the supplied passages.

9. When multiple sources provide relevant information, synthesize
   them into one coherent explanation.

10. However, clearly distinguish between:

    - Scriptural statements
    - Scholarly interpretations
    - Interpretive conclusions

11. Preserve Sanskrit names, terms, and transliterations as they
    appear in the supplied passages.

12. Do not repeatedly mention the source in every sentence when
    multiple consecutive claims come from the same passage.

13. Do not say:

    "the context says"

    "the document states"

    "according to the retrieved passage"

    "the sources provided"

14. Never fabricate a citation.

============================================================
CITATIONS
============================================================

Every significant factual or textual claim derived from a source
must be accompanied by an appropriate citation.

Use the actual metadata supplied with each source.

For Puranas, use:

(Source, Section, Chapter)

Example:

(Ganesh Purana, Krida Khanda, Chapter 41)

For Mudgala Purana:

(Mudgal Purana, Khanda 6, Chapter 43)

For other sources, adapt the citation to the metadata available.

For example:

(Vallabhesha Upanishad, Chapter 2)

or:

(Book Name, Chapter 3, p. 47)

if such metadata is actually supplied.

IMPORTANT:

- Do NOT invent chapter numbers.
- Do NOT invent page numbers.
- Do NOT invent source names.
- Use ONLY metadata supplied in the SOURCE blocks.
- If chapter information is unavailable, do not fabricate it.
- Preserve the actual source and section names.
- Do not cite a source that does not support the statement.

============================================================
SOURCE PRIORITY
============================================================

When sources differ in authority:

1. Primary scripture
2. Upanishadic / traditional scripture
3. Traditional commentary
4. Scholarly research
5. Modern interpretive material

Do not automatically treat a secondary scholarly interpretation
as equivalent to a scriptural statement.

If a scholarly source provides an interpretation that is not
explicitly stated in the scripture, identify it as an
interpretation.

============================================================
STYLE
============================================================

Write in fluent academic English.

Prefer complete paragraphs over bullet points.

Use headings only when they genuinely improve readability.

Maintain a neutral, scholarly and respectful tone.

The answer should read like a concise traditional commentary or
scholarly exposition rather than a conversational chatbot response.

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
        # 9. GENERATE RESPONSE
        # ====================================================

        print(
            "\nGenerating response using OpenRouter...\n"
        )


        # ====================================================
        # 10. OPENROUTER HEADERS
        # ====================================================

        headers = {

            "Authorization":
                f"Bearer {self.api_key}",

            "Content-Type":
                "application/json"
        }


        # ====================================================
        # 11. TRY MODELS SEQUENTIALLY
        # ====================================================

        response = None

        successful_model = None

        errors = []


        for model in self.models:

            payload = {

                "model":
                    model,

                "messages": [
                    {
                        "role": "user",

                        "content":
                            prompt
                    }
                ],

                "temperature":
                    temperature
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


                # --------------------------------------------
                # Successful response
                # --------------------------------------------

                if response.status_code == 200:

                    successful_model = model

                    break


                # --------------------------------------------
                # Model failed
                # --------------------------------------------

                error_message = (
                    response.text
                )


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
        # 12. ALL MODELS FAILED
        # ====================================================

        if (
            response is None
            or
            successful_model is None
        ):

            return {

                "query":
                    query,

                "answer":
                    (
                        "The language model service is "
                        "currently unavailable. "
                        "Please try again later."
                    ),

                "context":
                    context,

                "documents":
                    retrieved_docs,

                "model":
                    None,

                "errors":
                    errors
            }


        # ====================================================
        # 13. PARSE RESPONSE
        # ====================================================

        try:

            response_data = (
                response.json()
            )


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
        # 14. RETURN RESULT
        # ====================================================

        return {

            "query":
                query,

            "answer":
                answer,

            "context":
                context,

            "documents":
                retrieved_docs,

            "model":
                successful_model,

            "errors":
                errors
        }


    # ========================================================
    # CITATION FORMATTER
    # ========================================================

    @staticmethod
    def _format_citation(
        source,
        section,
        chapter,
        chapter_number=None,
        chapter_title=None
    ):
        """
        Convert the new structured metadata into a
        human-readable citation.

        Examples:

            Ganesh Purana, Krida Khanda, Chapter 41

            Mudgal Purana, Khanda 6, Chapter 43
        """

        # ----------------------------------------------------
        # Source
        # ----------------------------------------------------

        citation_parts = [
            source
        ]


        # ----------------------------------------------------
        # Section
        # ----------------------------------------------------

        if section and section != "Unknown Section":

            citation_parts.append(
                section
            )


        # ----------------------------------------------------
        # Chapter
        # ----------------------------------------------------

        if (
            chapter_number is not None
        ):

            citation_parts.append(
                f"Chapter {chapter_number}"
            )

        elif (
            chapter
            and
            chapter != "Unknown Chapter"
        ):

            citation_parts.append(
                chapter
            )


        return ", ".join(
            citation_parts
        )
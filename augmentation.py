import os
import requests
import dotenv


dotenv.load_dotenv(override=True)


class Augmentation:

    def __init__(
        self,
        vector_store=None,
        api_key=None,
        models=None
    ):
        self.vector_store = vector_store

        self.api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file."
            )

        # OpenRouter fallback models
        self.models = models or [
            "nvidia/nemotron-3.5-lightning",
            "liquid/lfm-2.5-2.6b",
            "inclusionai/ling-3.0-tiny"
        ]

        self.url = "https://openrouter.ai/api/v1/chat/completions"

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
            query: User question.
            top_k: Number of documents to retrieve.
            temperature: Generation temperature.

        Returns:
            Dictionary containing:
                query
                answer
                context
                documents
                model
        """

        # -----------------------------
        # Retrieve documents
        # -----------------------------

        retrieved_docs = (
            self.vector_store.similarity_search_with_score(
                query,
                top_k
            )
        )

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

        # -----------------------------
        # Build Context
        # -----------------------------

        context_parts = []

        for i, doc in enumerate(retrieved_docs, 1):

            # Document + score
            if isinstance(doc, tuple):

                document, score = doc

                metadata = (
                    document.metadata
                    if hasattr(document, "metadata")
                    else {}
                )

                # Compatibility with nested metadata
                if (
                    isinstance(metadata, dict)
                    and "metadata" in metadata
                ):
                    metadata = metadata["metadata"]

                book = metadata.get(
                    "original_book",
                    "Unknown Book"
                )

                chapter = metadata.get(
                    "original_chapter_info",
                    "Unknown Chapter"
                )

                text = (
                    document.content
                    if hasattr(document, "content")
                    else str(document)
                )

            # Dictionary format
            else:

                metadata = doc.get(
                    "metadata",
                    {}
                )

                if (
                    isinstance(metadata, dict)
                    and "metadata" in metadata
                ):
                    metadata = metadata["metadata"]

                book = metadata.get(
                    "original_book",
                    "Unknown Book"
                )

                chapter = metadata.get(
                    "original_chapter_info",
                    "Unknown Chapter"
                )

                text = doc.get(
                    "content",
                    ""
                )

            context_parts.append(
                f"""
==========================
SOURCE {i}
==========================

Book:
{book}

Chapter:
{chapter}

Content:
{text}

--------------------------
"""
            )

        context = "\n".join(context_parts)

        # -----------------------------
        # Prompt
        # -----------------------------

        prompt = f"""
You are an expert scholar of Hindu scriptures with deep knowledge of the
Ganesh Purana, Mudgala Purana, Ganapati Atharvashirsha, Upanishads,
and traditional Sanskrit commentaries.

Your task is to answer the user's question ONLY using the supplied context.
Treat the retrieved passages as the sole authoritative source.

The retrieved passages may contain partial portions of a larger chapter.
Therefore, carefully synthesize all relevant information contained in
the retrieved passages.

If the retrieved passages contain partial information relevant to the
question, use that information to construct the most complete answer
possible.

Only reply:

"I could not find the answer in the provided scriptures."

when none of the retrieved passages contain information relevant to
the question.

==========================
CONTEXT
==========================

{context}

==========================
QUESTION
==========================

{query}

==========================
INSTRUCTIONS
==========================

1. Answer ONLY from the supplied context.

2. Do NOT use external knowledge, prior knowledge, or assumptions.

3. If the retrieved passages contain relevant but incomplete information,
   explain only what can reasonably be established from those passages.

4. Write the answer as a scholarly, coherent narrative rather than
   automatically using bullet points.

5. Use complete paragraphs. Use bullet points or numbered lists only
   when the question specifically requires a list.

6. Explain the meaning, narrative context, symbolism, philosophical
   significance, and spiritual significance whenever such information
   is supported by the supplied passages.

7. When multiple sources provide relevant information, synthesize them
   into one coherent explanation rather than discussing each source
   separately.

8. Do not repeat the same information unnecessarily.

9. Preserve Sanskrit names, terms, and transliterations as they appear
   in the supplied passages.

10. Do not say phrases such as:
    "the context says"
    "the document states"
    "according to the retrieved passage"
    or
    "the sources provided".

11. Do not invent scriptural details.

==========================
CITATIONS
==========================

Every factual statement or significant claim derived from the scriptures
must be followed by a citation.

Use this exact citation format:

(Ganesh Purana, Krida Khand, Chapter 41)

For example:

Ganesha is described as the embodiment of all forms.
(Ganesh Purana, Krida Khand, Chapter 17)

If a statement is supported by multiple sources, cite all relevant sources:

(Ganesh Purana, Krida Khand, Chapter 41;
Mudgala Purana, Khand 6, Chapter 43)

IMPORTANT:

- Do NOT invent citations.
- Use ONLY the Book and Chapter information supplied in the SOURCE
  sections.
- Preserve the actual Purana and Khand names from the supplied metadata.
- If the supplied metadata says "Mudgal_Puran_Khand_6", interpret it
  as "Mudgal Purana, Khand 6" when formatting the citation.
- Do not cite a source that does not support the statement.

==========================
STYLE
==========================

Write in fluent academic English.

Prefer complete paragraphs over bullet points.

Use headings only when they genuinely improve readability.

Maintain a neutral, scholarly and respectful tone.

The answer should read like a concise traditional commentary or
scholarly exposition rather than a conversational chatbot response.

When appropriate, structure the explanation naturally as:

1. Direct answer to the question.
2. Narrative or textual context.
3. Meaning and significance.
4. Philosophical or spiritual interpretation.

Do not force this structure when the retrieved material does not support it.

==========================
ANSWER
==========================
"""

        print("\nGenerating response using OpenRouter...\n")

        # -----------------------------
        # OpenRouter headers
        # -----------------------------

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # -----------------------------
        # Try models sequentially
        # -----------------------------

        response = None
        successful_model = None
        errors = []

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

                print(f"Trying model: {model}")

                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=120
                )

                # Successful response
                if response.status_code == 200:

                    successful_model = model
                    break

                # Model unavailable / rate limited
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
                    f"Request failed for {model}: {e}"
                )

        # -----------------------------
        # All models failed
        # -----------------------------

        if response is None or successful_model is None:

            return {
                "query": query,
                "answer": (
                    "The language model service is currently "
                    "unavailable. Please try again later."
                ),
                "context": context,
                "documents": retrieved_docs,
                "model": None,
                "errors": errors
            }

        # -----------------------------
        # Parse OpenRouter response
        # -----------------------------

        try:

            response_data = response.json()

            answer = (
                response_data["choices"][0]["message"]["content"]
            )

        except (KeyError, IndexError, TypeError, ValueError):

            answer = (
                "The language model returned an unexpected response."
            )

            errors.append(response.text)

        # -----------------------------
        # Return result
        # -----------------------------

        return {
            "query": query,
            "answer": answer,
            "context": context,
            "documents": retrieved_docs,
            "model": successful_model,
            "errors": errors
        }
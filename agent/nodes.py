import os
import requests

from pydantic import BaseModel, Field

from agent.prompts import (
    GRADE_PROMPT,
    REWRITE_PROMPT,
    ANSWER_PROMPT
)

from retrieval.retriever import GaneshRetriever


# ============================================================
# GLOBAL RETRIEVER
# ============================================================

retriever = GaneshRetriever(
    puranas_collection="puranas",
    research_collection="research",
    retrieve_k=10,
    top_k=6
)


# ============================================================
# USAGE HELPERS
# ============================================================

def empty_usage():
    """
    Return a standardized empty usage dictionary.
    """

    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "reasoning_tokens": 0,
        "cached_tokens": 0
    }


def add_usage(existing_usage, new_usage):
    """
    Add usage from multiple LLM calls.

    This is important because the Agentic RAG pipeline
    can make multiple OpenRouter calls:

    1. Document grading
    2. Query rewriting
    3. Final answer generation

    The total usage displayed to the user should include
    all of them.
    """

    existing_usage = existing_usage or empty_usage()
    new_usage = new_usage or empty_usage()

    return {
        "prompt_tokens": (
            existing_usage.get("prompt_tokens", 0)
            + new_usage.get("prompt_tokens", 0)
        ),

        "completion_tokens": (
            existing_usage.get("completion_tokens", 0)
            + new_usage.get("completion_tokens", 0)
        ),

        "total_tokens": (
            existing_usage.get("total_tokens", 0)
            + new_usage.get("total_tokens", 0)
        ),

        "cost": (
            float(existing_usage.get("cost", 0))
            + float(new_usage.get("cost", 0))
        ),

        "reasoning_tokens": (
            existing_usage.get("reasoning_tokens", 0)
            + new_usage.get("reasoning_tokens", 0)
        ),

        "cached_tokens": (
            existing_usage.get("cached_tokens", 0)
            + new_usage.get("cached_tokens", 0)
        )
    }


# ============================================================
# OPENROUTER CALL
# ============================================================

def call_openrouter(
    prompt,
    models=None,
    temperature=0.2,
    max_tokens=600
):
    """
    Call OpenRouter API.

    Always returns a dictionary with:

    {
        "answer": str,
        "model": str,
        "usage": dict
    }

    This prevents tuple unpacking errors.
    """

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not configured."
        )

    # --------------------------------------------------------
    # FALLBACK MODELS
    # --------------------------------------------------------

    models = models or [

        "nvidia/nemotron-3.5-lightning",
        "inclusionai/ling-3.0-flash-fin",
        "liquid/lfm-2.5-2.6b"
    ]

    url = (
        "https://openrouter.ai/api/v1/"
        "chat/completions"
    )

    headers = {

        "Authorization":
            f"Bearer {api_key}",

        "Content-Type":
            "application/json"
    }

    errors = []

    # --------------------------------------------------------
    # TRY FALLBACK MODELS
    # --------------------------------------------------------

    for model in models:

        try:

            payload = {

                "model": model,

                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                "temperature": temperature,

                "max_tokens": max_tokens
            }

            response = requests.post(

                url,

                headers=headers,

                json=payload,

                timeout=120
            )

            # ------------------------------------------------
            # HANDLE FAILED MODEL
            # ------------------------------------------------

            if response.status_code != 200:

                errors.append(
                    f"{model}: "
                    f"HTTP {response.status_code} - "
                    f"{response.text}"
                )

                continue

            # ------------------------------------------------
            # PARSE RESPONSE
            # ------------------------------------------------

            data = response.json()

            choices = data.get(
                "choices",
                []
            )

            if not choices:

                errors.append(
                    f"{model}: "
                    "No choices returned."
                )

                continue

            message = choices[0].get(
                "message",
                {}
            )

            answer = message.get(
                "content",
                ""
            )

            if not answer:

                errors.append(
                    f"{model}: "
                    "Empty answer returned."
                )

                continue

            # =================================================
            # OPENROUTER USAGE ACCOUNTING
            # =================================================

            usage_data = data.get(
                "usage",
                {}
            )

            completion_details = usage_data.get(
                "completion_tokens_details",
                {}
            ) or {}

            prompt_details = usage_data.get(
                "prompt_tokens_details",
                {}
            ) or {}

            usage = {

                "prompt_tokens":
                    usage_data.get(
                        "prompt_tokens",
                        0
                    ),

                "completion_tokens":
                    usage_data.get(
                        "completion_tokens",
                        0
                    ),

                "total_tokens":
                    usage_data.get(
                        "total_tokens",
                        0
                    ),

                "cost":
                    float(
                        usage_data.get(
                            "cost",
                            0
                        )
                    ),

                "reasoning_tokens":
                    completion_details.get(
                        "reasoning_tokens",
                        0
                    ),

                "cached_tokens":
                    prompt_details.get(
                        "cached_tokens",
                        0
                    )
            }

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            return {

                "answer": answer.strip(),

                "model":
                    data.get(
                        "model",
                        model
                ),

                "usage": usage
            }

        except requests.exceptions.Timeout:

            errors.append(
                f"{model}: Request timed out."
            )

        except requests.exceptions.RequestException as e:

            errors.append(
                f"{model}: Request error - {str(e)}"
            )

        except ValueError as e:

            errors.append(
                f"{model}: Invalid JSON response - {str(e)}"
            )

        except Exception as e:

            errors.append(
                f"{model}: Unexpected error - {str(e)}"
            )

    # --------------------------------------------------------
    # ALL MODELS FAILED
    # --------------------------------------------------------

    raise RuntimeError(
        "All OpenRouter models failed:\n\n"
        + "\n".join(errors)
    )


# ============================================================
# NODE 1
# GENERATE SEARCH QUERY
# ============================================================

def generate_query(state):

    question = state["question"]

    return {

        "search_query": question,

        "rewrite_count":
            state.get(
                "rewrite_count",
                0
            ),

        "usage":
            state.get(
                "usage",
                empty_usage()
            )
    }


# ============================================================
# NODE 2
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(state):

    query = state[
        "search_query"
    ]

    candidates = retriever.retrieve(
        query
    )

    documents = []

    retrieval_info = []

    for candidate in candidates:

        document = candidate[
            "document"
        ]

        metadata = candidate.get(
            "metadata",
            {}
        ) or {}

        documents.append(
            document
        )

        retrieval_info.append({

            "collection":
                candidate.get(
                    "collection"
                ),

            "vector_rank":
                candidate.get(
                    "vector_rank"
                ),

            "vector_score":
                candidate.get(
                    "vector_score"
                ),

            "reranker_score":
                candidate.get(
                    "reranker_score"
                ),

            "source":
                metadata.get(
                    "source"
                ),

            "source_type":
                metadata.get(
                    "source_type"
                ),

            "authority":
                metadata.get(
                    "authority"
                ),

            "tradition":
                metadata.get(
                    "tradition"
                ),

            "section":
                metadata.get(
                    "section"
                ),

            "chapter":
                metadata.get(
                    "chapter"
                ),

            "chapter_number":
                metadata.get(
                    "chapter_number"
                ),

            "page_number":
                metadata.get(
                    "page_number"
                ),

            "citation":
                metadata.get(
                    "citation"
                ),

            "chunk_id":
                metadata.get(
                    "chunk_id"
                )
        })

    return {

        "documents": documents,

        "retrieval": retrieval_info
    }


# ============================================================
# NODE 3
# GRADE DOCUMENTS
# ============================================================

class GradeDocuments(BaseModel):

    binary_score: str = Field(

        description=(
            "YES if relevant, "
            "NO if irrelevant"
        )
    )


def grade_documents(state):
    """
    Grade reranked documents for relevance.

    Uses small max_tokens because the answer should only be:
    YES or NO.
    """

    question = state[
        "question"
    ]

    documents = state.get(
        "documents",
        []
    )

    current_usage = state.get(
        "usage",
        empty_usage()
    )

    if not documents:

        return {

            "documents_relevant": False,

            "usage": current_usage
        }

    relevant_count = 0

    total_usage = current_usage

    # --------------------------------------------------------
    # GRADE EACH DOCUMENT
    # --------------------------------------------------------

    for document in documents:

        context = document.page_content

        prompt = GRADE_PROMPT.format(

            question=question,

            context=context
        )

        try:

            result = call_openrouter(

                prompt,

                temperature=0,

                max_tokens=10
            )

            response = result.get(
                "answer",
                ""
            )

            usage = result.get(
                "usage",
                empty_usage()
            )

            total_usage = add_usage(
                total_usage,
                usage
            )

            score = (
                response
                .strip()
                .upper()
            )

            # Strict YES detection
            if score == "YES":

                relevant_count += 1

        except Exception as e:

            print(
                f"Document grading failed: {e}"
            )

            continue

    is_relevant = (
        relevant_count > 0
    )

    return {

        "documents_relevant":
            is_relevant,

        "usage":
            total_usage
    }


# ============================================================
# NODE 4
# REWRITE QUESTION
# ============================================================

def rewrite_question(state):
    """
    Rewrite query when retrieved documents
    are not sufficiently relevant.
    """

    question = state[
        "question"
    ]

    rewrite_count = state.get(
        "rewrite_count",
        0
    )

    current_usage = state.get(
        "usage",
        empty_usage()
    )

    # --------------------------------------------------------
    # PREVENT INFINITE LOOP
    # --------------------------------------------------------

    if rewrite_count >= 2:

        return {

            "search_query": question,

            "rewrite_count":
                rewrite_count,

            "usage":
                current_usage
        }

    prompt = REWRITE_PROMPT.format(

        question=question
    )

    try:

        result = call_openrouter(

            prompt,

            temperature=0,

            max_tokens=100
        )

        rewritten = result.get(

            "answer",

            state.get(
                "search_query",
                question
            )
        )

        usage = result.get(
            "usage",
            empty_usage()
        )

        total_usage = add_usage(

            current_usage,

            usage
        )

    except Exception as e:

        print(
            f"Query rewrite failed: {e}"
        )

        rewritten = state.get(
            "search_query",
            question
        )

        total_usage = current_usage

    return {

        "search_query":
            rewritten.strip(),

        "rewrite_count":
            rewrite_count + 1,

        "usage":
            total_usage
    }


# ============================================================
# NODE 5
# BUILD CONTEXT
# ============================================================

def build_context(state):

    documents = state.get(
        "documents",
        []
    )

    context_parts = []

    for index, document in enumerate(

        documents,

        start=1
    ):

        metadata = (
            document.metadata
            or {}
        )

        # ====================================================
        # BASIC METADATA
        # ====================================================

        source = metadata.get(

            "source",

            "Unknown Source"
        )

        source_type = metadata.get(

            "source_type",

            "unknown"
        )

        collection = metadata.get(

            "collection",

            "unknown"
        )

        authority = metadata.get(

            "authority",

            "unknown"
        )

        citation = metadata.get(
            "citation"
        )

        # ====================================================
        # BUILD CONTEXT
        # ====================================================

        context_parts.append(

            f"""
==================================================
SOURCE {index}
==================================================

Collection:
{collection}

Source:
{source}

Source Type:
{source_type}

Authority:
{authority}

Tradition:
{metadata.get("tradition", "Not specified")}

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

Citation:
{citation if citation else "Not specified"}

Chunk ID:
{metadata.get("chunk_id", "Not specified")}


---------------- CONTENT ----------------

{document.page_content}


==================================================
"""
        )

    context = "\n".join(
        context_parts
    )

    return {

        "context": context
    }


# ============================================================
# DETAILED ANSWER DETECTOR
# ============================================================

def wants_detailed_answer(question: str) -> bool:
    """
    Detect whether the user explicitly requests
    a detailed answer.
    """

    question = question.lower()

    detailed_phrases = [

        "answer in detail",

        "explain in detail",

        "explain this in detail",

        "elaborate in detail",

        "explain thoroughly",

        "give a detailed explanation",

        "provide a detailed answer",

        "explain deeply",

        "in great detail",

        "answer thoroughly",

        "detailed explanation"
    ]

    return any(

        phrase in question

        for phrase in detailed_phrases
    )


# ============================================================
# NODE 6
# GENERATE ANSWER
# ============================================================

def generate_answer(state):
    """
    Generate the final answer.

    Default:
        max_tokens = 600

    Explicit detailed request:
        max_tokens = 2000
    """

    question = state[
        "question"
    ]

    context = state.get(

        "context",

        ""
    )

    current_usage = state.get(

        "usage",

        empty_usage()
    )

    # --------------------------------------------------------
    # NO CONTEXT
    # --------------------------------------------------------

    if not context:

        return {

            "answer": (
                "I could not find sufficient "
                "information in the available sources "
                "to answer this question."
            ),

            "model": None,

            "usage": current_usage
        }

    # --------------------------------------------------------
    # OUTPUT TOKEN CONTROL
    # --------------------------------------------------------

    if wants_detailed_answer(question):

        max_tokens = 2000

    else:

        max_tokens = 600

    # --------------------------------------------------------
    # BUILD ANSWER PROMPT
    # --------------------------------------------------------

    prompt = ANSWER_PROMPT.format(

        question=question,

        context=context
    )

    # --------------------------------------------------------
    # CALL OPENROUTER
    # --------------------------------------------------------

    try:

        response = call_openrouter(

            prompt,

            temperature=0.2,

            max_tokens=max_tokens
        )

        answer = response.get(

            "answer",

            "No answer generated."
        )

        model = response.get(

            "model",

            "Unknown"
        )

        final_usage = response.get(

            "usage",

            empty_usage()
        )

        # ----------------------------------------------------
        # TOTAL AGENTIC WORKFLOW USAGE
        # ----------------------------------------------------

        total_usage = add_usage(

            current_usage,

            final_usage
        )

        return {

            "answer": answer,

            "model": model,

            "usage": total_usage
        }

    except Exception as e:

        print(
            f"Answer generation failed: {e}"
        )

        return {
            "answer": (
                {e},
                "The language model could not generate "
                "an answer at this time. Please try again."
            ),

            "model": None,

            "usage": current_usage
        }

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
# OPENROUTER CALL
# ============================================================

def call_openrouter(
    prompt,
    models=None,
    temperature=0.2,
    max_tokens=600
):

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "OPENROUTER_API_KEY is not configured."
        )

    models = models or [

        "nvidia/nemotron-3.5-lightning",

        "liquid/lfm-2.5-2.6b",

        "inclusionai/ling-3.0-tiny"
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

    for model in models:

        payload = {

            "model": model,

            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            "temperature": temperature,

            # "max_tokens": max_tokens
        }

        try:

            response = requests.post(

                url,

                headers=headers,

                json=payload,

                timeout=120
            )

            if response.status_code == 200:

                data = response.json()

                answer = (data["choices"][0]["message"]["content"])

                # ============================================
                # TOKEN USAGE
                # ============================================

                usage = data.get(
                    "usage",
                    {}
                )

                input_tokens = usage.get(
                    "prompt_tokens",
                    0
                )

                output_tokens = usage.get(
                    "completion_tokens",
                    0
                )

                total_tokens = usage.get(
                    "total_tokens",
                    input_tokens + output_tokens
                )

                return {"answer": answer,

                        "model": data.get(
                            "model",
                            model
                        ),
                        "usage": {
                            "prompt_tokens": input_tokens,

                            "completion_tokens": output_tokens,

                            "total_tokens": total_tokens
                        }}

            errors.append(
                f"{model}: "
                f"{response.status_code} "
                f"{response.text}"
            )

        except Exception as e:

            errors.append(
                f"{model}: {str(e)}"
            )

    raise RuntimeError(
        "\n".join(errors)
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
            )
    }


# ============================================================
# NODE 2
# RETRIEVE
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

        metadata = candidate[
            "metadata"
        ]

        documents.append(
            document
        )

        retrieval_info.append({

            # Which AstraDB collection
            "collection":
                candidate.get(
                    "collection"
                ),

            # Vector retrieval information
            "vector_rank":
                candidate.get(
                    "vector_rank"
                ),

            "vector_score":
                candidate.get(
                    "vector_score"
                ),

            # Cross Encoder Score
            "reranker_score":
                candidate.get(
                    "reranker_score"
                ),

            # Metadata
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

        "documents":
            documents,

        "retrieval":
            retrieval_info
    }


# ============================================================
# NODE 3
# GRADE DOCUMENTS
# ============================================================

class GradeDocuments(BaseModel):

    binary_score: str = Field(
        description=(
            "yes if relevant, "
            "no if irrelevant"
        )
    )


def grade_documents(state):

    question = state[
        "question"
    ]

    documents = state.get(
        "documents",
        []
    )

    if not documents:

        return {
            "documents_relevant": False
        }

    relevant_count = 0

    # Grade the reranked documents

    for document in documents:

        context = (
            document.page_content
        )

        prompt = GRADE_PROMPT.format(

            question=question,

            context=context
        )

        grading_prompt = (
            prompt
            + "\n\nReturn ONLY YES or NO."
        )

        try:

            answer, _ = call_openrouter(

                grading_prompt,

                temperature=0
            )

            score = (
                answer
                .strip()
                .lower()
            )

            if (
                "yes"
                in score
            ):

                relevant_count += 1

        except Exception:

            continue

    # At least one useful source

    is_relevant = (
        relevant_count > 0
    )

    return {

        "documents_relevant":
            is_relevant
    }


# ============================================================
# NODE 4
# REWRITE QUESTION
# ============================================================

def rewrite_question(state):

    question = state[
        "question"
    ]

    rewrite_count = (
        state.get(
            "rewrite_count",
            0
        )
    )

    # Prevent infinite loops

    if rewrite_count >= 2:

        return {

            "search_query":
                question,

            "rewrite_count":
                rewrite_count
        }

    prompt = REWRITE_PROMPT.format(

        question=question
    )

    rewritten, _ = call_openrouter(

        prompt,

        temperature=0
    )

    return {

        "search_query":
            rewritten.strip(),

        "rewrite_count":
            rewrite_count + 1
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

        # ================================================
        # BASIC METADATA
        # ================================================

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

        # ================================================
        # CITATION
        # ================================================

        citation = metadata.get(
            "citation"
        )

        # ================================================
        # BUILD CONTEXT
        # ================================================

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

        "context":
            context
    }

# ============================================================
# NODE 6
# GENERATE ANSWER
# ============================================================


def wants_detailed_answer(question: str) -> bool:
    """
    Detect whether the user explicitly requested
    a detailed explanation.
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
    ]

    return any(
        phrase in question
        for phrase in detailed_phrases
    )


def generate_answer(state):

    question = state[
        "question"
    ]

    context = state.get(
        "context",
        ""
    )

    if not context:

        return {

            "answer":
                "I could not find sufficient "
                "information in the provided sources."
        }

    # Decide output length based on explicit user request
    # if wants_detailed_answer(question):
    #     max_tokens = 2000
    # else:
    #     max_tokens = 600

    prompt = ANSWER_PROMPT.format(

        question=question,

        context=context
    )

    response = call_openrouter(

        prompt,

        temperature=0.2
    )

    answer = response.get(
    "answer",
    "No answer generated."
    )


    model = response.get(
        "model",
        "Unknown"
    )


    usage = response.get(
        "usage",
        {}
    )


    return {

        "answer": answer,

        "model": model,

        "usage": usage
    }

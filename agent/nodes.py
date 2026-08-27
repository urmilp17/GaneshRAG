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
    collection_name="puranas",
    retrieve_k=15,
    top_k=5
)


# ============================================================
# OPENROUTER CALL
# ============================================================

def call_openrouter(
    prompt,
    models=None,
    temperature=0.2
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

            "temperature": temperature
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

                answer = (
                    data["choices"][0]
                    ["message"]["content"]
                )

                return answer, model


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

        documents.append(
            candidate["document"]
        )


        metadata = (
            candidate["metadata"]
        )


        retrieval_info.append({

            "vector_rank":
                candidate[
                    "vector_rank"
                ],

            "vector_score":
                candidate[
                    "vector_score"
                ],

            "reranker_score":
                candidate[
                    "reranker_score"
                ],

            "source":
                metadata.get(
                    "source"
                ),

            "page_number":
                metadata.get(
                    "page_number"
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


        source = metadata.get(
            "source",
            "Unknown Source"
        )


        context_parts.append(

            f"""
==========================
SOURCE {index}
==========================

Source:
{source}

Source Type:
{metadata.get("source_type", "unknown")}

Authority:
{metadata.get("authority", "unknown")}

Tradition:
{metadata.get("tradition", "unknown")}

Section:
{metadata.get("section", "Not specified")}

Chapter:
{metadata.get("chapter", "Not specified")}

Chapter Number:
{metadata.get("chapter_number", "Not specified")}

Chapter Title:
{metadata.get("chapter_title", "Not specified")}

Page:
{metadata.get("page_number", "Not specified")}

Citation:
{metadata.get("citation", "Not specified")}

Chunk ID:
{metadata.get("chunk_id", "Not specified")}

Content:

{document.page_content}

==========================
"""
        )


    return {

        "context":
            "\n".join(
                context_parts
            )
    }


# ============================================================
# NODE 6
# GENERATE ANSWER
# ============================================================

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


    prompt = ANSWER_PROMPT.format(

        question=question,

        context=context
    )


    answer, model = call_openrouter(

        prompt,

        temperature=0.2
    )


    return {

        "answer":
            answer,

        "model":
            model
    }
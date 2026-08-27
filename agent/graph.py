from typing import Literal
import langgraph

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from agent.state import AgentState

from agent.nodes import (
    generate_query,
    retrieve_documents,
    grade_documents,
    rewrite_question,
    build_context,
    generate_answer
)


# ============================================================
# ROUTER AFTER GRADING
# ============================================================

def route_after_grading(
    state
) -> Literal[
    "build_context",
    "rewrite_question"
]:

    if state.get(
        "documents_relevant",
        False
    ):

        return "build_context"


    rewrite_count = state.get(
        "rewrite_count",
        0
    )


    # Prevent infinite rewrite loop

    if rewrite_count >= 2:

        return "build_context"


    return "rewrite_question"


# ============================================================
# BUILD GRAPH
# ============================================================

def create_graph():

    workflow = StateGraph(
        AgentState
    )


    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    workflow.add_node(
        "generate_query",
        generate_query
    )

    workflow.add_node(
        "retrieve_documents",
        retrieve_documents
    )

    workflow.add_node(
        "grade_documents",
        grade_documents
    )

    workflow.add_node(
        "rewrite_question",
        rewrite_question
    )

    workflow.add_node(
        "build_context",
        build_context
    )

    workflow.add_node(
        "generate_answer",
        generate_answer
    )


    # --------------------------------------------------------
    # Initial flow
    # --------------------------------------------------------

    workflow.add_edge(
        START,
        "generate_query"
    )


    workflow.add_edge(
        "generate_query",
        "retrieve_documents"
    )


    workflow.add_edge(
        "retrieve_documents",
        "grade_documents"
    )


    # --------------------------------------------------------
    # Conditional edge
    # --------------------------------------------------------

    workflow.add_conditional_edges(

        "grade_documents",

        route_after_grading,

        {

            "build_context":
                "build_context",

            "rewrite_question":
                "rewrite_question"
        }
    )


    # --------------------------------------------------------
    # Rewrite → retrieve again
    # --------------------------------------------------------

    workflow.add_edge(

        "rewrite_question",

        "retrieve_documents"
    )


    # --------------------------------------------------------
    # Context → answer
    # --------------------------------------------------------

    workflow.add_edge(

        "build_context",

        "generate_answer"
    )


    workflow.add_edge(

        "generate_answer",

        END
    )


    return workflow.compile()


# ============================================================
# CREATE GRAPH
# ============================================================

graph = create_graph()
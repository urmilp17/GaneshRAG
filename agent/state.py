from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict, total=False):

    # Original user question
    question: str

    # Current rewritten question
    search_query: str

    # Retrieved and reranked documents
    documents: List[Any]

    # Retrieval diagnostics
    retrieval: List[Dict[str, Any]]

    # Formatted context
    context: str

    # Generated answer
    answer: str

    # Whether documents were considered relevant
    documents_relevant: bool

    # Number of rewrite attempts
    rewrite_count: int

    # Model used
    model: str

    # Errors
    errors: List[str]
    
    #Usage
    usage: dict
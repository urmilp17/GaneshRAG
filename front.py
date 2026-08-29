import os
import html
import streamlit as st
import dotenv

from agent.graph import graph
from embedder import SentenceTransformerEmbeddings
from langchain_astradb import AstraDBVectorStore


# ============================================================
# ENVIRONMENT
# ============================================================

dotenv.load_dotenv(override=True)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Ganesh Tattvagyan RAG",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    :root {
        --saffron: #FF7722;
        --saffron-light: #FF8F3F;
        --saffron-dark: #E65A00;
        --saffron-glow: rgba(255, 119, 34, 0.18);
    }

    /* ==============================
       Main Header
       ============================== */

    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--saffron) !important;
        margin-bottom: 0.4rem;
        text-shadow: 0 2px 10px rgba(255, 119, 34, 0.2);
    }

    .subtitle {
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }


    /* ==============================
       ANSWER BOX
       ============================== */

    .answer-box {
        background-color: var(--secondary-background-color) !important;

        color: var(--text-color) !important;

        padding: 1.5rem;

        border-radius: 10px;

        border-left: 5px solid var(--saffron) !important;

        margin: 1rem 0;

        line-height: 1.75;

        font-size: 1rem;

        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }


    /* ==============================
       MODEL INFO
       ============================== */

    .model-info {
        background-color: var(--secondary-background-color) !important;

        color: var(--text-color) !important;

        padding: 0.8rem;

        border-radius: 8px;

        margin-top: 0.5rem;

        border: 1px solid rgba(255, 119, 34, 0.35) !important;
    }


    /* ==============================
       CONTEXT BOX
       ============================== */

    .context-box {
        background-color: var(--secondary-background-color) !important;

        color: var(--text-color) !important;

        padding: 1rem;

        border-radius: 8px;

        border: 1px solid rgba(255, 119, 34, 0.25) !important;

        margin: 0.5rem 0;

        font-family: monospace;

        white-space: pre-wrap;

        word-wrap: break-word;

        line-height: 1.6;
    }


    /* ==============================
       SOURCE CARD
       ============================== */

    .source-card {
        background-color: var(--secondary-background-color) !important;

        color: var(--text-color) !important;

        padding: 1rem;

        border-radius: 8px;

        border-left: 4px solid var(--saffron);

        margin: 0.7rem 0;

        line-height: 1.6;
    }


    .source-title {
        color: var(--saffron) !important;

        font-size: 1.05rem;

        font-weight: 700;

        margin-bottom: 0.5rem;
    }


    .metadata-label {
        font-weight: 600;
    }


    /* ==============================
       TEXT INPUT
       ============================== */

    .stTextInput > div > div > input {
        font-size: 1.1rem;

        border-color: var(--saffron) !important;
    }


    .stTextInput > div > div > input:focus {
        border-color: var(--saffron) !important;

        box-shadow:
            0 0 0 2px
            rgba(255, 119, 34, 0.3) !important;
    }


    /* ==============================
       BUTTON
       ============================== */

    .stButton button {
        background-color: var(--saffron) !important;

        color: white !important;

        border: none !important;

        transition: all 0.3s ease !important;
    }


    .stButton button:hover {
        background-color: var(--saffron-dark) !important;

        transform: translateY(-2px) !important;

        box-shadow:
            0 4px 12px
            rgba(255, 119, 34, 0.4) !important;
    }


    /* ==============================
       DIVIDERS
       ============================== */

    hr {
        border-color:
            rgba(255, 119, 34, 0.3) !important;
    }


    /* ==============================
       ALERTS
       ============================== */

    .stAlert {
        border-left-color:
            var(--saffron) !important;
    }


    .stAlert svg {
        fill:
            var(--saffron) !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CREATE EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():

    return SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        device=None,
    )


# ============================================================
# CREATE ASTRA VECTOR STORE
# ============================================================

# @st.cache_resource(show_spinner="Connecting to Astra DB...")
# def get_vector_store():

#     embedder = get_embedder()

#     vector_store = AstraDBVectorStore(
#         collection_name="puranas",

#         embedding=embedder,

#         token=os.getenv(
#             "ASTRA_DB_APPLICATION_TOKEN"
#         ),

#         api_endpoint=os.getenv(
#             "ASTRA_DB_API_ENDPOINT"
#         ),
#     )

#     return vector_store


# ============================================================
# CREATE AUGMENTATION SYSTEM
# ============================================================

# @st.cache_resource(show_spinner="Loading RAG and reranker...")
# def get_augmentation():

#     vector_store = get_vector_store()

#     return Augmentation(
#         vector_store=vector_store
#     )


# ============================================================
# INITIALIZE SYSTEM
# ============================================================

try:

    # augmentation = get_augmentation()

    initialized = True

except Exception as e:

    augmentation = None

    initialized = False

    st.error(
        f"⚠️ Error initializing RAG system: {str(e)}"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        🕉️ Ganesh Tattvagyan RAG
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Ask questions and receive source-grounded answers
        from the Ganesh scripture knowledge base.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Retrieval Configuration")


    # --------------------------------------------------------
    # Initial vector retrieval
    # --------------------------------------------------------

    retrieve_k = st.slider(
        "Initial Retrieval K",

        min_value=5,

        max_value=30,

        value=15,

        step=1,

        help=(
            "Number of chunks retrieved from Astra DB "
            "before Cross-Encoder reranking."
        )
    )


    # --------------------------------------------------------
    # Final reranked contexts
    # --------------------------------------------------------

    top_k = st.slider(
        "Final Reranked Top K",

        min_value=1,

        max_value=10,

        value=5,

        step=1,

        help=(
            "Number of highest-ranked chunks passed "
            "to the language model after reranking."
        )
    )


    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    temperature = st.slider(
        "LLM Temperature",

        min_value=0.0,

        max_value=1.0,

        value=0.2,

        step=0.05,

        help=(
            "Lower values make the response more "
            "deterministic and conservative."
        )
    )


    st.divider()


    # ========================================================
    # DISPLAY OPTIONS
    # ========================================================

    st.header("🖥️ Display Options")


    show_sources = st.checkbox(
        "Show ranked sources",

        value=True,

        help=(
            "Display the final sources selected by "
            "the Cross-Encoder reranker."
        )
    )


    show_scores = st.checkbox(
        "Show retrieval scores",

        value=False,

        help=(
            "Display vector and reranker scores "
            "for debugging and evaluation."
        )
    )


    show_raw_context = st.checkbox(
        "Show raw context",

        value=False,

        help=(
            "Display the exact context supplied "
            "to the language model."
        )
    )


    st.divider()


    # ========================================================
    # SYSTEM INFORMATION
    # ========================================================

    st.header("📚 Knowledge Base")


    st.markdown(
        """
        **Database:** `ganesa_data`

        **Keyspace:** `default_keyspace`

        **Collection:** `puranas, research`

        **Embedding:** `all-MiniLM-L6-v2`

        **Reranker:** `BAAI/bge-reranker-v2-m3`
        """
    )


    st.divider()


    # ========================================================
    # SYSTEM STATUS
    # ========================================================

    st.header("📊 System Status")


    if initialized:

        st.success(
            "✅ RAG system initialized"
        )

        st.info(
            "🔹 Vector retrieval + reranking ready"
        )

    else:

        st.error(
            "❌ System initialization failed"
        )


    st.divider()


    # ========================================================
    # ABOUT
    # ========================================================

    st.header("ℹ️ About")

    st.markdown(
        """
        This system uses a two-stage RAG pipeline:

        **1. Vector Retrieval**

        Astra DB retrieves semantically similar
        chunks.

        **2. Cross-Encoder Reranking**

        A reranker evaluates query-document
        relevance more precisely.

        **3. Grounded Generation**

        The final selected sources are provided
        to the language model for answer generation.
        """
    )


# ============================================================
# QUERY AREA
# ============================================================

col1, col2 = st.columns(
    [3, 1]
)


with col1:

    query = st.text_input(
        "💬 Enter your question:",

        placeholder=(
            "e.g., What is Ganesh Hridayam?"
        ),

        help=(
            "Type your question and press Enter "
            "or click Search."
        )
    )


with col2:

    st.write("")

    st.write("")

    search_button = st.button(
        "🚀 Search",

        type="primary",

        use_container_width=True
    )


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_ranked_sources(
    retrieval_data
):
    """
    Display final reranked retrieval results.

    Expected structure:

        [
            {
                vector_rank,
                vector_score,
                reranker_score,
                source,
                page_number,
                chunk_id
            }
        ]
    """

    if not retrieval_data:

        st.info(
            "No ranked source information available."
        )

        return


    for index, item in enumerate(
        retrieval_data,
        start=1
    ):

        source = item.get(
            "source",
            "Unknown Source"
        )

        page_number = item.get(
            "page_number"
        )

        vector_rank = item.get(
            "vector_rank"
        )

        vector_score = item.get(
            "vector_score"
        )

        reranker_score = item.get(
            "reranker_score"
        )

        chunk_id = item.get(
            "chunk_id"
        )


        # ----------------------------------------------------
        # Page display
        # ----------------------------------------------------

        page_text = ""

        if page_number is not None:

            page_text = (
                f" • Page {page_number}"
            )


        with st.expander(
            f"#{index} — {source}{page_text}",
            expanded=(index == 1)
        ):

            st.markdown(
                f"""
                <div class="source-card">

                    <div class="source-title">
                        Source #{index}
                    </div>

                    <b>Source:</b>
                    {html.escape(str(source))}

                    <br><br>

                    <b>Page:</b>
                    {html.escape(
                        str(page_number)
                        if page_number is not None
                        else "Not specified"
                    )}

                    <br><br>

                    <b>Chunk ID:</b>
                    <code>
                    {html.escape(
                        str(chunk_id)
                        if chunk_id
                        else "Not specified"
                    )}
                    </code>

                </div>
                """,
                unsafe_allow_html=True
            )


            if show_scores:

                st.markdown(
                    "### 📊 Retrieval Scores"
                )


                score_col1, score_col2, score_col3 = (
                    st.columns(3)
                )


                with score_col1:

                    st.metric(
                        "Vector Rank",

                        (
                            str(vector_rank)
                            if vector_rank is not None
                            else "N/A"
                        )
                    )


                with score_col2:

                    if vector_score is not None:

                        st.metric(
                            "Vector Score",

                            f"{float(vector_score):.6f}"
                        )

                    else:

                        st.metric(
                            "Vector Score",
                            "N/A"
                        )


                with score_col3:

                    if reranker_score is not None:

                        st.metric(
                            "Reranker Score",

                            f"{float(reranker_score):.6f}"
                        )

                    else:

                        st.metric(
                            "Reranker Score",
                            "N/A"
                        )


# ============================================================
# DISPLAY CONTEXT
# ============================================================

def display_context(
    context_data
):
    """
    Safely display retrieved context.
    """

    if context_data is None:

        st.info(
            "No context data available."
        )

        return


    if isinstance(
        context_data,
        str
    ):

        st.markdown(
            '<div class="context-box">'
            + html.escape(context_data)
            + '</div>',
            unsafe_allow_html=True
        )

        return


    if isinstance(
        context_data,
        list
    ):

        for index, item in enumerate(
            context_data,
            start=1
        ):

            with st.expander(
                f"📄 Context {index}"
            ):

                if isinstance(
                    item,
                    dict
                ):

                    for key, value in (
                        item.items()
                    ):

                        st.markdown(
                            f"**{key}:**"
                        )

                        st.markdown(
                            '<div class="context-box">'
                            + html.escape(
                                str(value)
                            )
                            + '</div>',
                            unsafe_allow_html=True
                        )

                else:

                    st.markdown(
                        '<div class="context-box">'
                        + html.escape(
                            str(item)
                        )
                        + '</div>',
                        unsafe_allow_html=True
                    )

        return


    if isinstance(
        context_data,
        dict
    ):

        for key, value in (
            context_data.items()
        ):

            st.markdown(
                f"**{key}:**"
            )

            st.markdown(
                '<div class="context-box">'
                + html.escape(
                    str(value)
                )
                + '</div>',
                unsafe_allow_html=True
            )

        return


    st.markdown(
        '<div class="context-box">'
        + html.escape(
            str(context_data)
        )
        + '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY METADATA
# ============================================================

def display_document_metadata(
    documents
):
    """
    Display metadata from the final reranked LangChain
    documents.
    """

    if not documents:

        st.info(
            "No document metadata available."
        )

        return


    for index, document in enumerate(
        documents,
        start=1
    ):

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
            isinstance(
                metadata,
                dict
            )
            and
            isinstance(
                metadata.get(
                    "metadata"
                ),
                dict
            )
        ):

            metadata = metadata[
                "metadata"
            ]


        source = metadata.get(
            "source",
            "Unknown Source"
        )

        source_type = metadata.get(
            "source_type",
            "Unknown"
        )

        authority = metadata.get(
            "authority",
            "Unknown"
        )

        section = metadata.get(
            "section"
        )

        chapter = metadata.get(
            "chapter"
        )

        chapter_number = metadata.get(
            "chapter_number"
        )

        chapter_title = metadata.get(
            "chapter_title"
        )

        page_number = metadata.get(
            "page_number"
        )

        citation = metadata.get(
            "citation"
        )


        with st.expander(
            f"📖 {index}. {source}"
        ):

            st.markdown(
                f"**Source Type:** "
                f"`{source_type}`"
            )

            st.markdown(
                f"**Authority:** "
                f"`{authority}`"
            )


            if section:

                st.markdown(
                    f"**Section:** {section}"
                )


            if chapter:

                st.markdown(
                    f"**Chapter:** {chapter}"
                )


            if chapter_number is not None:

                st.markdown(
                    f"**Chapter Number:** "
                    f"{chapter_number}"
                )


            if chapter_title:

                st.markdown(
                    f"**Chapter Title:** "
                    f"{chapter_title}"
                )


            if page_number is not None:

                st.markdown(
                    f"**Page:** "
                    f"{page_number}"
                )


            if citation:

                st.markdown(
                    f"**Citation:** "
                    f"`{citation}`"
                )


# ============================================================
# PROCESS QUERY
# ============================================================

if (
    search_button
    or query
) and query.strip():

    if not initialized:

        st.error(
            "⚠️ RAG system is not initialized. "
            "Please check the configuration."
        )

    else:

        with st.spinner(
            "🔍 Retrieving, reranking and generating..."
        ):

            try:

                # ============================================
                # RUN RAG
                # ============================================

                result = graph.invoke(
                    {
                        "question": query,
                        "rewrite_count": 0
                    }
                )


                st.divider()


                # ============================================
                # ANSWER
                # ============================================

                st.subheader(
                    "📝 Answer"
                )


                answer = result.get(
                    "answer",
                    "No answer generated."
                )


                # IMPORTANT:
                #
                # Use Streamlit Markdown instead of placing
                # the answer inside an HTML div.
                #
                # This preserves:
                #
                # - Markdown
                # - headings
                # - citations
                # - bold text
                # - paragraphs
                # - lists

                st.markdown(
                    '<div class="answer-box">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    answer
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )


                # ============================================
                # MODEL
                # ============================================

                model = result.get(
                    "model"
                )


                st.markdown(
                    f"""
                    <div class="model-info">

                    🤖 <b>Model used:</b>
                    {html.escape(
                        str(model)
                        if model
                        else "Default model"
                    )}

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # ============================================
                # RANKED SOURCES
                # ============================================

                retrieval_data = result.get(
                    "retrieval",
                    []
                )
                
                if retrieval_data:
                    st.markdown(
                        "## 📚 Retrieved Sources"
                    )

                    for i, item in enumerate(
                        retrieval_data,
                        start=1
                    ):

                        st.write(
                            f"""
                            **{i}. {item.get("source", "Unknown")}**

                            Vector Rank:
                            {item.get("vector_rank")}

                            Vector Score:
                            {item.get("vector_score")}

                            Reranker Score:
                            {item.get("reranker_score")}

                            Page:
                            {item.get("page_number")}

                            Chunk:
                            {item.get("chunk_id")}
                            """
                        )


                if (
                    show_sources
                    and retrieval_data
                ):

                    st.divider()

                    st.subheader(
                        "📚 Ranked Sources"
                    )

                    st.caption(
                        "Sources shown below are the final "
                        "chunks selected after Cross-Encoder "
                        "reranking."
                    )


                    display_ranked_sources(
                        retrieval_data
                    )


                # ============================================
                # DOCUMENT METADATA
                # ============================================

                documents = result.get(
                    "documents",
                    []
                )


                if (
                    show_sources
                    and documents
                ):

                    with st.expander(
                        "🔖 Detailed Source Metadata",
                        expanded=False
                    ):

                        display_document_metadata(
                            documents
                        )


                # ============================================
                # RAW CONTEXT
                # ============================================

                if show_raw_context:

                    context = result.get(
                        "context"
                    )


                    with st.expander(
                        "🔍 Raw Context Sent to LLM",
                        expanded=False
                    ):

                        display_context(
                            context
                        )


                # ============================================
                # DEBUG / ERRORS
                # ============================================

                errors = result.get(
                    "errors",
                    []
                )


                if errors:

                    with st.expander(
                        "⚠️ Model / Retrieval Logs",
                        expanded=False
                    ):

                        for error in errors:

                            st.warning(
                                str(error)
                            )


                st.divider()

                st.success(
                    "✅ Query completed successfully!"
                )


            except Exception as e:

                st.error(
                    f"❌ Error processing query: "
                    f"{str(e)}"
                )


                with st.expander(
                    "🔍 Debug information",
                    expanded=False
                ):

                    st.exception(e)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

else:

    if not query:

        st.info(
            "💡 Enter a question above and press "
            "Search to get started."
        )


        col1, col2, col3 = st.columns(
            3
        )


        with col1:

            st.markdown(
                "**📖 Factual**"
            )

            st.markdown(
                "What is Ganesh Hridayam?"
            )


        with col2:

            st.markdown(
                "**🧠 Conceptual**"
            )

            st.markdown(
                "What is the significance of "
                "Ganesha's elephant head?"
            )


        with col3:

            st.markdown(
                "**📚 Cross-scriptural**"
            )

            st.markdown(
                "How is Ganesha described "
                "across the Puranas?"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🕉️ Ganesh Tattvagyan RAG • "
    "Vector Retrieval + Cross-Encoder Reranking"
)


# ============================================================
# RESET CACHE
# ============================================================

if st.button(
    "🔄 Reset Cache",
    help="Reload embedding model, Astra DB connection and reranker."
):

    st.cache_resource.clear()

    st.success(
        "Cache cleared successfully!"
    )

    st.rerun()
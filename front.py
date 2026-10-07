import os
import html
import streamlit as st
import dotenv

# The Agentic RAG graph owns the retriever.  The new retriever searches:
# puranas, research, iconography, rahasya, sahastranaam and upanishad(s).
dotenv.load_dotenv(override=True)

try:
    from agent.graph import graph
    initialized = True
    initialization_error = None
except Exception as e:
    graph = None
    initialized = False
    initialization_error = str(e)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Ganesh Tattvagyan RAG",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --saffron: #FF7722;
        --saffron-dark: #E65A00;
    }

    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--saffron) !important;
        margin-bottom: 0.4rem;
        text-shadow: 0 2px 10px rgba(255, 119, 34, 0.20);
    }

    .subtitle {
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

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

    .model-info {
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        padding: 0.8rem;
        border-radius: 8px;
        margin-top: 0.5rem;
        border: 1px solid rgba(255, 119, 34, 0.35) !important;
    }

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

    .stTextInput > div > div > input {
        font-size: 1.1rem;
        border-color: var(--saffron) !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--saffron) !important;
        box-shadow: 0 0 0 2px rgba(255, 119, 34, 0.30) !important;
    }

    .stButton button {
        background-color: var(--saffron) !important;
        color: white !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }

    .stButton button:hover {
        background-color: var(--saffron-dark) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(255, 119, 34, 0.40) !important;
    }

    hr {
        border-color: rgba(255, 119, 34, 0.30) !important;
    }

    .stAlert {
        border-left-color: var(--saffron) !important;
    }

    .stAlert svg {
        fill: var(--saffron) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_value(value, default="Not specified"):
    """Return a display-safe value for optional metadata."""
    if value is None or value == "" or value == -1:
        return default
    return value


def format_score(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def display_ranked_sources(retrieval_data, show_scores=True):
    """Display final candidates returned by the new retriever/graph."""
    if not retrieval_data:
        st.info("No ranked source information available.")
        return

    for index, item in enumerate(retrieval_data, start=1):
        source = safe_value(item.get("source"), "Unknown Source")
        collection = safe_value(item.get("collection"), "Unknown Collection")
        citation = item.get("citation")
        source_type = item.get("source_type")
        authority = item.get("authority")
        page_number = item.get("page_number")
        chunk_id = item.get("chunk_id")

        with st.expander(
            f"#{index} — {source} • {collection}",
            expanded=(index == 1),
        ):
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">Source #{index}</div>
                    <b>Collection:</b> {html.escape(str(collection))}<br><br>
                    <b>Source:</b> {html.escape(str(source))}<br><br>
                    <b>Source Type:</b> {html.escape(str(safe_value(source_type)))}<br><br>
                    <b>Authority:</b> {html.escape(str(safe_value(authority)))}<br><br>
                    <b>Page:</b> {html.escape(str(safe_value(page_number)))}<br><br>
                    <b>Citation:</b> {html.escape(str(safe_value(citation)))}<br><br>
                    <b>Chunk ID:</b> <code>{html.escape(str(safe_value(chunk_id)))}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if show_scores:
                score_col1, score_col2, score_col3 = st.columns(3)
                with score_col1:
                    st.metric("Vector Rank", safe_value(item.get("vector_rank"), "N/A"))
                with score_col2:
                    st.metric("Vector Score", format_score(item.get("vector_score")))
                with score_col3:
                    st.metric("Reranker Score", format_score(item.get("reranker_score")))


def display_context(context_data):
    """Safely display the exact context supplied to the LLM."""
    if context_data is None:
        st.info("No context data available.")
        return

    if isinstance(context_data, str):
        st.markdown(
            '<div class="context-box">'
            + html.escape(context_data)
            + '</div>',
            unsafe_allow_html=True,
        )
        return

    if isinstance(context_data, list):
        for index, item in enumerate(context_data, start=1):
            with st.expander(f"📄 Context {index}"):
                if isinstance(item, dict):
                    for key, value in item.items():
                        st.markdown(f"**{key}:**")
                        st.markdown(
                            '<div class="context-box">'
                            + html.escape(str(value))
                            + '</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<div class="context-box">'
                        + html.escape(str(item))
                        + '</div>',
                        unsafe_allow_html=True,
                    )
        return

    if isinstance(context_data, dict):
        for key, value in context_data.items():
            st.markdown(f"**{key}:**")
            st.markdown(
                '<div class="context-box">'
                + html.escape(str(value))
                + '</div>',
                unsafe_allow_html=True,
            )
        return

    st.markdown(
        '<div class="context-box">'
        + html.escape(str(context_data))
        + '</div>',
        unsafe_allow_html=True,
    )


def display_document_metadata(documents):
    """Display all useful metadata from the final LangChain Documents."""
    if not documents:
        st.info("No document metadata available.")
        return

    # These are intentionally ordered for readability. Any additional
    # metadata fields are displayed afterwards.
    preferred_fields = [
        "collection",
        "source",
        "source_name",
        "source_type",
        "document_type",
        "authority",
        "tradition",
        "upanishad",
        "upanishad_short_name",
        "author",
        "commentary",
        "commentator",
        "form_name",
        "name",
        "name_devanagari",
        "section",
        "chapter",
        "chapter_number",
        "chapter_title",
        "mantra_number",
        "shloka_number",
        "page_number",
        "citation",
        "file_name",
        "file_path",
        "chunk_id",
        "chunk_index",
        "total_chunks",
    ]

    for index, document in enumerate(documents, start=1):
        if hasattr(document, "metadata"):
            metadata = document.metadata or {}
        elif isinstance(document, dict):
            metadata = document.get("metadata", {}) or {}
        else:
            metadata = {}

        if (
            isinstance(metadata, dict)
            and isinstance(metadata.get("metadata"), dict)
        ):
            metadata = metadata["metadata"]

        source = metadata.get("source", "Unknown Source")

        with st.expander(f"📖 {index}. {source}"):
            shown = set()

            for key in preferred_fields:
                if key not in metadata:
                    continue
                value = metadata.get(key)
                if value is None or value == "" or value == -1:
                    continue

                shown.add(key)
                label = key.replace("_", " ").title()
                st.markdown(f"**{label}:** {value}")

            # Show any newly added metadata fields automatically.
            for key, value in metadata.items():
                if key in shown or key in {"metadata"}:
                    continue
                if value is None or value == "" or value == -1:
                    continue
                label = key.replace("_", " ").title()
                st.markdown(f"**{label}:** {value}")


def display_usage(usage):
    """Display aggregated OpenRouter usage from the Agentic RAG graph."""
    if not usage:
        st.info("No token usage information returned.")
        return

    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)
    reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)
    cached_tokens = int(usage.get("cached_tokens", 0) or 0)

    raw_cost = usage.get("cost", 0) or 0
    try:
        cost = float(raw_cost)
    except (TypeError, ValueError):
        cost = 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📥 Input Tokens", f"{prompt_tokens:,}")
    with col2:
        st.metric("📤 Output Tokens", f"{completion_tokens:,}")
    with col3:
        st.metric("🔢 Total Tokens", f"{total_tokens:,}")
    with col4:
        st.metric("💰 Cost", f"${cost:.6f}")

    with st.expander("🔍 Detailed Usage Information"):
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.metric("🧠 Reasoning Tokens", f"{reasoning_tokens:,}")
        with detail_col2:
            st.metric("⚡ Cached Tokens", f"{cached_tokens:,}")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-header">🕉️ Ganesh Tattvagyan RAG</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Ask questions and receive source-grounded answers '
    'from the Ganesh scripture knowledge base.</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Agentic RAG Configuration")

    st.markdown("### 🔎 Retrieval Pipeline")
    st.markdown(
        """
        **1. Query** → **2. Vector Retrieval** → **3. OpenRouter Reranking**  
        **4. Document Grading** → **5. Query Rewrite if needed**  
        **6. Grounded Generation**
        """
    )

    st.info(
        "The retrieval parameters are controlled by the backend "
        "`GaneshRetriever` used by the Agentic RAG graph."
    )

    st.markdown("### 📚 Knowledge Base")
    st.markdown(
        """
        **Database:** `ganesa_data`  
        **Keyspace:** `default_keyspace`

        **Collections:**
        - `puranas`
        - `research`
        - `iconography`
        - `rahasya`
        - `sahastranaam`
        - `upanishads`

        **Embedding:** `all-MiniLM-L6-v2`  
        **Reranking:** OpenRouter `/rerank` API
        """
    )

    st.markdown("### 🧠 Retrieval Settings")
    st.caption("Current backend defaults from the new retriever:")
    st.code(
        "retrieve_k = 10\n"
        "rerank_per_collection = 5\n"
        "top_k = 6",
        language="python",
    )

    st.markdown("### 🤖 Reranker Models")
    rerank_models = [
        m.strip()
        for m in os.getenv(
            "RERANK_MODELS",
            "cohere/rerank-v3.5,voyageai/rerank-2.5-lite,qwen/qwen3-reranker-8b",
        ).split(",")
        if m.strip()
    ]
    for model_name in rerank_models:
        st.caption(f"• `{model_name}`")

    st.divider()

    st.header("🖥️ Display Options")
    show_sources = st.checkbox("Show ranked sources", value=True)
    show_scores = st.checkbox("Show retrieval scores", value=False)
    show_raw_context = st.checkbox("Show raw context", value=False)
    show_metadata = st.checkbox("Show detailed metadata", value=True)
    show_token_usage = st.checkbox("Show token usage", value=True)

    st.divider()

    st.header("📊 System Status")
    if initialized:
        st.success("✅ Agentic RAG graph initialized")
        st.info("🔹 6-collection retrieval + OpenRouter reranking ready")
    else:
        st.error("❌ Agentic RAG graph initialization failed")
        if initialization_error:
            st.code(initialization_error)

    st.divider()

    st.header("ℹ️ About")
    st.markdown(
        """
        This system uses an Agentic RAG pipeline:

        **1. Vector Retrieval**  
        Astra DB collections are searched in parallel using one embedded query.

        **2. OpenRouter Reranking**  
        Candidate passages are reranked through the OpenRouter rerank API.

        **3. Document Grading**  
        Retrieved documents are checked for relevance.

        **4. Query Rewriting**  
        If retrieval is insufficient, the query can be rewritten and searched again.

        **5. Grounded Generation**  
        The final answer is generated from the selected source context.
        """
    )


# ============================================================
# QUERY AREA
# ============================================================

with st.form("query_form", clear_on_submit=False):
    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "💬 Enter your question:",
            placeholder="e.g., Who is Herambha?",
            help="Ask a question about the Ganesha knowledge base.",
        )

    with col2:
        st.write("")
        st.write("")
        search_button = st.form_submit_button(
            "🚀 Search",
            type="primary",
            use_container_width=True,
        )


# ============================================================
# PROCESS QUERY
# ============================================================

if search_button:
    if not query.strip():
        st.warning("Please enter a question first.")

    elif not initialized or graph is None:
        st.error(
            "⚠️ The Agentic RAG graph is not initialized. "
            "Please check your environment variables and backend configuration."
        )

    else:
        with st.spinner("🔍 Retrieving, reranking, grading and generating..."):
            try:
                result = graph.invoke(
                    {
                        "question": query.strip(),
                        "rewrite_count": 0,
                    }
                )

                st.divider()

                # ====================================================
                # ANSWER
                # ====================================================

                st.subheader("📝 Answer")

                answer = result.get(
                    "answer",
                    "No answer generated.",
                )

                st.markdown(
                    '<div class="answer-box">',
                    unsafe_allow_html=True,
                )
                st.markdown(answer)
                st.markdown(
                    '</div>',
                    unsafe_allow_html=True,
                )

                # ====================================================
                # MODEL / QUERY INFORMATION
                # ====================================================

                model = result.get("model")
                search_query = result.get("search_query")
                rewrite_count = result.get("rewrite_count", 0)

                st.markdown(
                    f"""
                    <div class="model-info">
                        🤖 <b>Model used:</b>
                        {html.escape(str(model) if model else "Default model")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.metric(
                        "🔄 Query Rewrites",
                        str(rewrite_count),
                    )
                with info_col2:
                    if search_query and search_query.strip() != query.strip():
                        st.markdown("**🔎 Final Search Query:**")
                        st.code(search_query, language="text")
                    else:
                        st.caption("Original query used for retrieval.")

                # ====================================================
                # TOKEN USAGE
                # ====================================================

                if show_token_usage:
                    st.markdown("### 📊 Token Usage")
                    display_usage(result.get("usage", {}))

                # ====================================================
                # RANKED SOURCES
                # ====================================================

                retrieval_data = result.get("retrieval", [])

                if retrieval_data:
                    st.markdown("## 📚 Retrieved Sources")

                    if not show_sources:
                        # Compact summary when full source cards are disabled.
                        st.caption(
                            f"{len(retrieval_data)} final source(s) selected by the retriever."
                        )

                    if show_sources:
                        st.caption(
                            "Sources below are the final candidates selected after "
                            "vector retrieval and OpenRouter reranking."
                        )
                        display_ranked_sources(
                            retrieval_data,
                            show_scores=show_scores,
                        )

                else:
                    st.info("No retrieval information was returned.")

                # ====================================================
                # DOCUMENT METADATA
                # ====================================================

                documents = result.get("documents", [])

                if show_sources and show_metadata and documents:
                    st.divider()
                    with st.expander(
                        "🔖 Detailed Source Metadata",
                        expanded=False,
                    ):
                        display_document_metadata(documents)

                # ====================================================
                # RAW CONTEXT
                # ====================================================

                if show_raw_context:
                    context = result.get("context")
                    st.divider()
                    with st.expander(
                        "🔍 Raw Context Sent to LLM",
                        expanded=False,
                    ):
                        display_context(context)

                # ====================================================
                # ERRORS / LOGS
                # ====================================================

                errors = result.get("errors", [])

                if errors:
                    with st.expander(
                        "⚠️ Model / Retrieval Logs",
                        expanded=False,
                    ):
                        for error in errors:
                            st.warning(str(error))

                st.divider()
                st.success("✅ Query completed successfully!")

            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                with st.expander("🔍 Debug information", expanded=False):
                    st.exception(e)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

if not search_button and not query:
    st.info("💡 Enter a question above and press Search to get started.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📖 Iconography**")
        st.markdown("Who is Herambha?")

    with col2:
        st.markdown("**🪔 Ritual**")
        st.markdown("Why is coconut offered to Ganesha?")

    with col3:
        st.markdown("**📚 Comparative**")
        st.markdown("How is Herambha described across different sources?")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "🕉️ Ganesh Tattvagyan RAG • "
    "6-Collection Vector Retrieval + OpenRouter Reranking + Agentic RAG"
)


# ============================================================
# RESET STREAMLIT CACHE
# ============================================================

if st.button(
    "🔄 Reset Streamlit Cache",
    help="Clear Streamlit's cached resources and rerun the application.",
):
    st.cache_resource.clear()
    st.success("Streamlit cache cleared. Reloading...")
    st.rerun()

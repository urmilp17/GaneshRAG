import os
import streamlit as st
import dotenv
from augmentation import Augmentation
from embedder import SentenceTransformerEmbeddings
from langchain_astradb import AstraDBVectorStore
import os
import dotenv

dotenv.load_dotenv(override=True)

embedder = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2",
    device=None,
    
)
    
    # Initialize vector store
vector_store = AstraDBVectorStore(
    collection_name="documents",
    embedding=embedder,
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
)


# Page configuration - Must be first Streamlit command
st.set_page_config(
    page_title="Augmentation Query System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with proper dark mode support and saffron theme
st.markdown("""
    <style>
    /* Saffron color variables */
    :root {
        --saffron: #FF7722;
        --saffron-light: #FF8F3F;
        --saffron-dark: #E65A00;
        --saffron-glow: rgba(255, 119, 34, 0.2);
    }
    
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--saffron) !important;
        margin-bottom: 1rem;
        text-shadow: 0 2px 10px var(--saffron-glow);
    }
    
    /* Answer box with saffron theme */
    .answer-box {
        background-color: var(--saffron-glow) !important;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF7722 !important;
        margin: 1rem 0;
        color: inherit !important;
    }
    
    /* Model info box */
    .model-info {
        background-color: var(--saffron-glow) !important;
        padding: 0.8rem;
        border-radius: 8px;
        margin-top: 0.5rem;
        border: 1px solid rgba(255, 119, 34, 0.3) !important;
        color: inherit !important;
    }
    
    /* Context boxes */
    .context-box {
        background-color: var(--saffron-glow) !important;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 119, 34, 0.2) !important;
        margin: 0.5rem 0;
        font-family: monospace;
        white-space: pre-wrap;
        word-wrap: break-word;
        color: inherit !important;
    }
    
    .context-item {
        background-color: var(--saffron-glow) !important;
        padding: 0.8rem;
        border-radius: 6px;
        border-left: 3px solid #FF7722 !important;
        margin: 0.5rem 0;
        color: inherit !important;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        border-color: #FF7722 !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #FF7722 !important;
        box-shadow: 0 0 0 2px rgba(255, 119, 34, 0.3) !important;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #FF7722 !important;
        color: white !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        background-color: #E65A00 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(255, 119, 34, 0.4) !important;
    }
    .stButton button:active {
        transform: translateY(0px) !important;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .answer-box {
            background-color: rgba(255, 119, 34, 0.15) !important;
            border-left-color: #FF7722 !important;
        }
        .model-info {
            background-color: rgba(255, 119, 34, 0.12) !important;
        }
        .context-box {
            background-color: rgba(255, 119, 34, 0.08) !important;
            border-color: rgba(255, 119, 34, 0.3) !important;
        }
        .context-item {
            background-color: rgba(255, 119, 34, 0.1) !important;
        }
    }
    
    /* Light mode specific */
    @media (prefers-color-scheme: light) {
        .answer-box {
            background-color: #FFF5EE !important;
        }
        .model-info {
            background-color: #FFF5EE !important;
        }
        .context-box {
            background-color: #FFFBF7 !important;
        }
        .context-item {
            background-color: #FFF5EE !important;
        }
    }
    
    /* Streamlit native dark mode support */
    [data-theme="dark"] .answer-box {
        background-color: rgba(255, 119, 34, 0.12) !important;
    }
    [data-theme="dark"] .model-info {
        background-color: rgba(255, 119, 34, 0.1) !important;
    }
    [data-theme="dark"] .context-box {
        background-color: rgba(255, 119, 34, 0.08) !important;
        border-color: rgba(255, 119, 34, 0.25) !important;
    }
    [data-theme="dark"] .context-item {
        background-color: rgba(255, 119, 34, 0.1) !important;
    }
    
    /* Saffron colored dividers */
    hr {
        border-color: rgba(255, 119, 34, 0.3) !important;
    }
    
    /* Saffron colored expander headers */
    .streamlit-expanderHeader {
        color: #FF7722 !important;
        font-weight: 600 !important;
    }
    
    /* Success messages in saffron */
    .stAlert {
        border-left-color: #FF7722 !important;
    }
    .stAlert svg {
        fill: #FF7722 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description with saffron accent
st.markdown('<p class="main-header">🔍 Augmentation Query System</p>', unsafe_allow_html=True)
st.markdown("Ask questions and get augmented answers with context from your knowledge base.")

# Initialize session state
if 'aug' not in st.session_state:
    try:
        st.session_state.aug = Augmentation(
            vector_store=vector_store,
        )
        st.session_state.initialized = True
    except NameError:
        st.error("⚠️ Please define 'vector_store' before initializing the Augmentation class.")
        st.session_state.initialized = False
    except Exception as e:
        st.error(f"⚠️ Error initializing augmentation system: {str(e)}")
        st.session_state.initialized = False

# Sidebar with parameters
with st.sidebar:
    st.header("⚙️ Configuration")
    
    top_k = st.slider(
        "Number of contexts to retrieve (Top K)",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
        help="Number of relevant documents to retrieve from vector store"
    )
    
    st.divider()
    st.header("🖥️ Display Options")
    
    show_raw_context = st.checkbox(
        "Show raw context",
        value=False,
        help="Display the raw context data without formatting"
    )
    
    show_sources = st.checkbox(
        "Show sources",
        value=True,
        help="Display the source documents used for augmentation"
    )
    
    st.divider()
    st.header("ℹ️ About")
    st.markdown("""
    This system uses Retrieval Augmented Generation (RAG) to:
    - Retrieve relevant context from your knowledge base
    - Generate accurate responses based on retrieved information
    - Provide transparency about the model used
    """)
    
    if st.session_state.get('initialized', False):
        st.divider()
        st.header("📊 System Status")
        st.success("✅ System initialized")
        st.info("🔹 Ready for queries")

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "💬 Enter your question:",
        placeholder="e.g., What is Ganesh Hridayam?",
        help="Type your question and press Enter or click the button"
    )

with col2:
    st.write("")
    st.write("")
    search_button = st.button("🚀 Search", type="primary", use_container_width=True)

def display_context(context_data):
    """Safely display context data regardless of format"""
    if context_data is None:
        st.info("No context data available")
        return
    
    if isinstance(context_data, str):
        if context_data.strip().startswith('{') or context_data.strip().startswith('['):
            try:
                import json
                parsed = json.loads(context_data)
                if isinstance(parsed, list):
                    for idx, item in enumerate(parsed):
                        with st.expander(f"📄 Context {idx + 1}"):
                            if isinstance(item, dict):
                                for key, value in item.items():
                                    st.markdown(f"**{key}:**")
                                    st.markdown(f'<div class="context-box">{value}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="context-box">{item}</div>', unsafe_allow_html=True)
                else:
                    st.json(parsed)
            except:
                st.markdown(f'<div class="context-box">{context_data}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="context-box">{context_data}</div>', unsafe_allow_html=True)
    
    elif isinstance(context_data, list):
        for idx, item in enumerate(context_data):
            with st.expander(f"📄 Document {idx + 1}"):
                if isinstance(item, dict):
                    for key, value in item.items():
                        st.markdown(f"**{key}:**")
                        st.markdown(f'<div class="context-box">{value}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="context-box">{item}</div>', unsafe_allow_html=True)
    
    elif isinstance(context_data, dict):
        for key, value in context_data.items():
            st.markdown(f"**{key}:**")
            st.markdown(f'<div class="context-box">{value}</div>', unsafe_allow_html=True)
    
    else:
        st.markdown(f'<div class="context-box">{str(context_data)}</div>', unsafe_allow_html=True)

# Process query
if (search_button or query) and query.strip():
    if not st.session_state.get('initialized', False):
        st.error("⚠️ System not initialized. Please check configuration.")
    else:
        with st.spinner("🔍 Searching and generating response..."):
            try:
                result = st.session_state.aug.augment(
                    query=query,
                    top_k=top_k
                )
                
                st.divider()
                
                st.subheader("📝 Answer")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                
                if "model" in result:
                    st.markdown(f'<div class="model-info">🤖 <b>Model used:</b> {result["model"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="model-info">🤖 <b>Model used:</b> Default model</div>', unsafe_allow_html=True)
                
                if "sources" in result and result["sources"] and show_sources:
                    with st.expander("📚 View sources", expanded=True):
                        st.markdown("**Sources used for this response:**")
                        display_context(result["sources"])
                
                context_key = None
                if "context" in result:
                    context_key = "context"
                elif "documents" in result:
                    context_key = "documents"
                elif "retrieved_context" in result:
                    context_key = "retrieved_context"
                
                if context_key:
                    with st.expander("📚 View retrieved context", expanded=False):
                        st.markdown("**Retrieved documents:**")
                        display_context(result[context_key])
                
                if show_raw_context:
                    with st.expander("🔍 Raw context data", expanded=False):
                        st.code(str(result), language="python")
                
                st.divider()
                st.success("✅ Query completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.info("Please check your augmentation configuration and try again.")
                
                with st.expander("🔍 Debug information", expanded=False):
                    st.code(str(e), language="python")

else:
    if not query:
        st.info("💡 Enter a question above and press Enter or click 'Search' to get started.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Example questions:**")
            st.markdown("• What is Ganesh Hridayam?")
        with col2:
            st.markdown("• What are the key features?")
        with col3:
            st.markdown("• How does this system work?")

st.divider()
st.caption("🔹 Augmentation System • Powered by RAG")

if st.button("🔄 Reset Cache", help="Clear cached responses"):
    st.cache_data.clear()
    st.success("Cache cleared successfully!")
    st.rerun()
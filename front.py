import os
import streamlit as st
import dotenv
from augmentation import Augmentation
from embedder import SentenceTransformerEmbeddings
from langchain_astradb import AstraDBVectorStore
from google.cloud import firestore
import uuid
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


# Page configuration
st.set_page_config(
    page_title="Augmentation Query System",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .model-info {
        background-color: #e8f4f8;
        padding: 0.8rem;
        border-radius: 8px;
        margin-top: 0.5rem;
    }
    .context-box {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
        font-family: monospace;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .context-item {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 6px;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="main-header">🔍 Augmentation Query System</p>', unsafe_allow_html=True)
st.markdown("Ask questions and get augmented answers with context from your knowledge base.")

# Initialize session state
if 'aug' not in st.session_state:
    try:
        # Initialize the augmentation system
        # Note: You need to define vector_store before this
        st.session_state.aug = Augmentation(
            vector_store=vector_store,  # Make sure this is defined
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
    
    # Top K parameter
    top_k = st.slider(
        "Number of contexts to retrieve (Top K)",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
        help="Number of relevant documents to retrieve from vector store"
    )
    
    # Display options
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
    
    # Additional settings
    st.divider()
    st.header("ℹ️ About")
    st.markdown("""
    This system uses Retrieval Augmented Generation (RAG) to:
    - Retrieve relevant context from your knowledge base
    - Generate accurate responses based on retrieved information
    - Provide transparency about the model used
    """)
    
    # Model info (if available)
    if st.session_state.get('initialized', False):
        st.divider()
        st.header("📊 System Status")
        st.success("✅ System initialized")
        st.info("🔹 Ready for queries")

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    # Query input
    query = st.text_input(
        "💬 Enter your question:",
        placeholder="e.g., Who is Bhanu Vinayak?",
        help="Type your question and press Enter or click the button"
    )

# Query button
with col2:
    st.write("")  # Spacer
    st.write("")  # Spacer
    search_button = st.button("🚀 Search", type="primary", use_container_width=True)

# Function to safely display context
def display_context(context_data):
    """Safely display context data regardless of format"""
    if context_data is None:
        st.info("No context data available")
        return
    
    # Check if it's a string
    if isinstance(context_data, str):
        # Try to parse as JSON if it looks like JSON
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
                # If JSON parsing fails, display as raw text
                st.markdown(f'<div class="context-box">{context_data}</div>', unsafe_allow_html=True)
        else:
            # Display as plain text with formatting
            st.markdown(f'<div class="context-box">{context_data}</div>', unsafe_allow_html=True)
    
    # Check if it's a list
    elif isinstance(context_data, list):
        for idx, item in enumerate(context_data):
            with st.expander(f"📄 Document {idx + 1}"):
                if isinstance(item, dict):
                    for key, value in item.items():
                        st.markdown(f"**{key}:**")
                        st.markdown(f'<div class="context-box">{value}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="context-box">{item}</div>', unsafe_allow_html=True)
    
    # Check if it's a dictionary
    elif isinstance(context_data, dict):
        for key, value in context_data.items():
            st.markdown(f"**{key}:**")
            st.markdown(f'<div class="context-box">{value}</div>', unsafe_allow_html=True)
    
    else:
        # Fallback: display as string
        st.markdown(f'<div class="context-box">{str(context_data)}</div>', unsafe_allow_html=True)

# Process query
if (search_button or query) and query.strip():
    if not st.session_state.get('initialized', False):
        st.error("⚠️ System not initialized. Please check configuration.")
    else:
        with st.spinner("🔍 Searching and generating response..."):
            try:
                # Execute augmentation
                result = st.session_state.aug.augment(
                    query=query,
                    top_k=top_k
                )
                
                # Display results
                st.divider()
                
                # Display answer
                st.subheader("📝 Answer")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                
                # Display model information
                if "model" in result:
                    st.markdown(f'<div class="model-info">🤖 <b>Model used:</b> {result["model"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="model-info">🤖 <b>Model used:</b> Default model</div>', unsafe_allow_html=True)
                
                # Display sources if available and enabled
                if "sources" in result and result["sources"] and show_sources:
                    with st.expander("📚 View sources", expanded=True):
                        st.markdown("**Sources used for this response:**")
                        display_context(result["sources"])
                
                # Display context if available
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
                
                # Show raw context if enabled
                if show_raw_context:
                    with st.expander("🔍 Raw context data", expanded=False):
                        st.code(str(result), language="python")
                
                # Add a divider and some spacing
                st.divider()
                
                # Show success message
                st.success("✅ Query completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.info("Please check your augmentation configuration and try again.")
                
                # Show debug info
                with st.expander("🔍 Debug information", expanded=False):
                    st.code(str(e), language="python")

else:
    # Show placeholder when no query is entered
    if not query:
        st.info("💡 Enter a question above and press Enter or click 'Search' to get started.")
        
        # Show example questions in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Example questions:**")
            st.markdown("• Who is Bhanu Vinayak?")
        with col2:
            st.markdown("• What are the key features?")
        with col3:
            st.markdown("• How does this system work?")

# Footer
st.divider()
st.caption("🔹 Augmentation System • Powered by RAG")

# Optional: Add a reset button to clear cache
if st.button("🔄 Reset Cache", help="Clear cached responses"):
    st.cache_data.clear()
    st.success("Cache cleared successfully!")
    st.rerun()
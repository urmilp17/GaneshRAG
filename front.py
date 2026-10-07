import os
import json
import html
import streamlit as st
import streamlit.components.v1 as components
import dotenv

# The Agentic RAG graph owns the retriever.  The retriever searches:
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
# CONSTANTS
# ============================================================

USD_TO_INR = 95                                   # $1 = ₹95
SHARE_LINK = "https://ganeshrag.streamlit.app/"

EXAMPLES = [
    ("Iconography", "Who is Herambha?"),
    ("Ritual", "Why is coconut offered to Ganesha?"),
    ("Comparative", "How is Herambha described across different sources?"),
]


# ============================================================
# PAGE CONFIGURATION  (no sidebar)
# ============================================================

st.set_page_config(
    page_title="Ganesh Tattvagyan",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS  -  parchment, ink, kumkum and gold
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Cormorant+Garamond:ital,wght@0,500;0,700;1,500;1,600&family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Noto+Serif:wght@500;700&family=Tiro+Devanagari+Sanskrit:ital@0;1&display=swap');

    :root {
        --parchment: #F4E8CF;
        --paper:     #FFF9EA;
        --ink:       #2E1F14;
        --muted:     #6E5640;
        --maroon:    #7A1F1F;
        --maroon-dk: #5A1414;
        --saffron:   #C8650B;
        --gold:      #B38A2E;
        --gold-soft: rgba(179, 138, 46, 0.45);
        --serif:     'EB Garamond', 'Tiro Devanagari Sanskrit', Georgia, 'Times New Roman', serif;
        --display:   'Cinzel', 'Cormorant Garamond', Georgia, serif;
        --italic:    'Cormorant Garamond', 'EB Garamond', Georgia, serif;
        --deva:      'Tiro Devanagari Sanskrit', 'Noto Serif', serif;
    }

    /* ---------- page ---------- */
    .stApp {
        background:
            url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.45  0 0 0 0 0.33  0 0 0 0 0.15  0 0 0 0.09 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>"),
            radial-gradient(ellipse at 50% -10%, #FBF3DE 0%, #F1E2C0 70%, #E9D6AE 100%);
        color: var(--ink);
        font-family: var(--serif);
    }

    .block-container {
        max-width: 920px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    /* hide Streamlit chrome + sidebar entirely */
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu, footer {
        display: none !important;
        visibility: hidden !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }

    /* ---------- typography ---------- */
    html, body, [class*="css"], .stMarkdown, label, button, input, textarea {
        font-family: var(--serif) !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        font-family: var(--serif);
        font-size: 1.14rem;
        line-height: 1.75;
        color: var(--ink);
    }

    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        font-family: var(--italic);
        font-weight: 700;
        color: var(--maroon);
    }

    [data-testid="stMarkdownContainer"] strong { color: var(--maroon-dk); }

    [data-testid="stMarkdownContainer"] a {
        color: var(--saffron);
        text-decoration: underline dotted;
    }

    [data-testid="stMarkdownContainer"] code {
        background: rgba(179, 138, 46, 0.15);
        color: var(--maroon-dk);
        border-radius: 2px;
        padding: 0 0.3em;
    }

    [data-testid="stCaptionContainer"] p {
        font-family: var(--italic);
        font-style: italic;
        font-size: 1.02rem;
        color: var(--muted);
    }

    .deva { font-family: var(--deva); }

    /* ---------- masthead ---------- */
    .toran {
        height: 14px;
        background-color: var(--gold);
        background-image:
            linear-gradient(135deg, var(--maroon) 25%, transparent 25%),
            linear-gradient(225deg, var(--maroon) 25%, transparent 25%);
        background-size: 14px 14px;
        background-position: 0 0, 0 0;
        border-top: 2px solid var(--maroon);
        margin-bottom: 1.1rem;
    }

    .masthead { text-align: center; padding: 0.3rem 0 0.2rem; }

    .invocation {
        font-size: 1.25rem;
        color: var(--saffron);
        letter-spacing: 0.04em;
    }

    .om {
        font-family: var(--deva);
        font-size: 3.2rem;
        line-height: 1.1;
        color: var(--maroon);
        margin: 0.1rem 0 0;
        text-shadow: 0 2px 0 rgba(179, 138, 46, 0.35);
    }

    .title {
        font-family: var(--display);
        font-weight: 700;
        font-size: 2.55rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--maroon);
        margin: 0.2rem 0 0;
        line-height: 1.15;
    }

    .title-deva {
        font-size: 1.5rem;
        color: var(--gold);
        margin-top: 0.15rem;
    }

    .subtitle {
        font-family: var(--italic);
        font-style: italic;
        font-weight: 500;
        font-size: 1.28rem;
        color: var(--muted);
        max-width: 640px;
        margin: 0.7rem auto 0;
        line-height: 1.5;
    }

    .ornament {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin: 1.1rem 0;
        color: var(--maroon);
        font-size: 1.15rem;
    }
    .ornament::before, .ornament::after {
        content: "";
        flex: 1;
        border-top: 3px double var(--gold);
    }

    /* ---------- section titles ---------- */
    .section-title {
        display: flex;
        align-items: baseline;
        gap: 0.8rem;
        margin: 1.6rem 0 0.7rem;
        padding-bottom: 0.25rem;
        border-bottom: 3px double var(--gold);
    }
    .section-title .deva { font-size: 1.55rem; color: var(--maroon); }
    .section-title .eng {
        font-family: var(--display);
        font-size: 0.82rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--muted);
    }

    /* ---------- form ---------- */
    [data-testid="stForm"] {
        background: rgba(255, 249, 234, 0.65);
        border: 1px solid var(--gold-soft) !important;
        border-radius: 2px;
        padding: 1.1rem 1.3rem 0.6rem;
        box-shadow: inset 0 0 0 4px rgba(255, 249, 234, 0.9), inset 0 0 0 5px var(--gold-soft);
    }

    .stTextInput label p {
        font-family: var(--italic) !important;
        font-style: italic;
        font-size: 1.2rem !important;
        color: var(--maroon) !important;
    }

    div[data-baseweb="input"], div[data-baseweb="base-input"] {
        background: #FFFBEF !important;
        border-radius: 2px !important;
    }
    div[data-baseweb="input"] { border: 1px solid var(--gold) !important; }
    div[data-baseweb="input"]:focus-within {
        border-color: var(--maroon) !important;
        box-shadow: 0 0 0 2px rgba(122, 31, 31, 0.18) !important;
    }
    .stTextInput input {
        font-family: var(--serif) !important;
        font-size: 1.2rem !important;
        color: var(--ink) !important;
        background: transparent !important;
    }
    .stTextInput input::placeholder { color: #9A8468 !important; font-style: italic; }

    /* ---------- buttons ---------- */
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: var(--maroon) !important;
        color: #FFF3D6 !important;
        border: 1px solid var(--gold) !important;
        border-radius: 2px !important;
        box-shadow: inset 0 0 0 2px var(--maroon), inset 0 0 0 3px rgba(255, 243, 214, 0.45);
        transition: background 0.2s ease, transform 0.2s ease !important;
    }
    div.stButton > button p,
    div[data-testid="stFormSubmitButton"] > button p {
        color: inherit !important;
        font-family: var(--display) !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: var(--maroon-dk) !important;
        transform: translateY(-1px);
    }

    /* example-question buttons: quieter, sentence case */
    .st-key-examples div.stButton > button {
        background: transparent !important;
        color: var(--ink) !important;
        border: 1px dashed var(--gold) !important;
        box-shadow: none !important;
        min-height: 3.2rem;
    }
    .st-key-examples div.stButton > button p {
        font-family: var(--italic) !important;
        font-style: italic;
        font-size: 1.1rem !important;
        letter-spacing: 0;
        text-transform: none;
    }
    .st-key-examples div.stButton > button:hover { background: rgba(179, 138, 46, 0.14) !important; }

    .eg-cat {
        font-family: var(--display);
        font-size: 0.78rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--saffron);
        margin: 0.2rem 0 0.35rem;
    }

    /* ---------- question + answer ---------- */
    .question-quote {
        font-family: var(--italic);
        font-style: italic;
        font-weight: 600;
        font-size: 1.45rem;
        line-height: 1.4;
        color: var(--maroon-dk);
        border-left: 3px solid var(--maroon);
        padding: 0.2rem 0 0.2rem 1rem;
        margin: 0.4rem 0 0.4rem;
    }

    .st-key-answer_card {
        position: relative;
        background: var(--paper);
        border: 1px solid var(--gold);
        padding: 1.7rem 2rem 1.2rem;
        margin: 1rem 6px 0.9rem;
        box-shadow:
            0 0 0 5px var(--paper),
            0 0 0 6px var(--gold-soft),
            0 10px 24px rgba(90, 60, 20, 0.12);
    }
    .st-key-answer_card::before {
        content: "❖";
        position: absolute;
        top: -0.85rem;
        left: 50%;
        transform: translateX(-50%);
        background: #F3E6CA;
        color: var(--maroon);
        padding: 0 0.7rem;
        font-size: 1.1rem;
    }
    .st-key-answer_card [data-testid="stMarkdownContainer"] > p:first-child::first-letter {
        font-family: var(--display);
        font-weight: 700;
        font-size: 3.3rem;
        line-height: 0.85;
        float: left;
        padding: 0.28rem 0.55rem 0 0;
        color: var(--maroon);
    }

    .copy-note {
        font-family: var(--italic);
        font-style: italic;
        color: var(--muted);
        font-size: 1.02rem;
        margin: -0.2rem 0 0.4rem;
    }

    .colophon {
        font-family: var(--italic);
        font-style: italic;
        font-size: 1.08rem;
        color: var(--muted);
        text-align: center;
        margin: 0.9rem 0 0.4rem;
    }
    .colophon b { color: var(--maroon); font-style: normal; font-family: var(--serif); }

    /* ---------- sources ---------- */
    .source-card {
        background: var(--paper);
        border: 1px solid var(--gold-soft);
        border-left: 4px solid var(--maroon);
        padding: 0.9rem 1.2rem;
        margin: 0.3rem 0 0.6rem;
    }
    .source-title {
        font-family: var(--display);
        font-size: 0.88rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--maroon);
        margin-bottom: 0.55rem;
    }
    .kv {
        display: grid;
        grid-template-columns: 8.5rem 1fr;
        gap: 0.6rem;
        padding: 0.18rem 0;
        border-bottom: 1px dotted var(--gold-soft);
        font-size: 1.08rem;
    }
    .kv:last-child { border-bottom: none; }
    .kv .k { font-style: italic; color: var(--muted); }
    .kv .v { color: var(--ink); word-break: break-word; }

    .context-box {
        background: var(--paper);
        color: var(--ink);
        padding: 1rem;
        border: 1px solid var(--gold-soft);
        margin: 0.5rem 0;
        font-family: 'Noto Serif', Georgia, serif;
        font-size: 0.95rem;
        white-space: pre-wrap;
        word-wrap: break-word;
        line-height: 1.6;
    }

    /* ---------- expanders, metrics, alerts, code ---------- */
    [data-testid="stExpander"] {
        background: rgba(255, 249, 234, 0.6);
        border: 1px solid var(--gold-soft) !important;
        border-radius: 2px !important;
    }
    [data-testid="stExpander"] summary p {
        font-family: var(--italic) !important;
        font-weight: 600;
        font-size: 1.15rem !important;
        color: var(--maroon) !important;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 249, 234, 0.75);
        border: 1px solid var(--gold-soft);
        border-radius: 2px;
        padding: 0.6rem 0.9rem;
    }
    [data-testid="stMetricLabel"] p {
        font-family: var(--display) !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Noto Serif', 'EB Garamond', serif !important;
        color: var(--maroon) !important;
        font-size: 1.55rem !important;
    }

    [data-testid="stAlert"] {
        background: #FBF0D5 !important;
        border: 1px solid var(--gold-soft) !important;
        border-left: 4px solid var(--maroon) !important;
        border-radius: 2px !important;
        color: var(--ink) !important;
    }
    [data-testid="stAlert"] p { color: var(--ink) !important; }

    [data-testid="stCode"] pre, [data-testid="stCode"] code {
        background: var(--paper) !important;
        border-radius: 2px;
    }

    [data-testid="stSpinner"] p {
        font-family: var(--italic) !important;
        font-style: italic;
        font-size: 1.15rem !important;
        color: var(--maroon) !important;
    }

    .footer-mantra {
        text-align: center;
        font-family: var(--deva);
        font-size: 1.3rem;
        color: var(--saffron);
        margin-top: 0.4rem;
    }

    hr { border-color: var(--gold-soft) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def keyed_container(key, **kwargs):
    """st.container(key=...) gives us a CSS hook (.st-key-<key>) on newer Streamlit."""
    try:
        return st.container(key=key, **kwargs)
    except TypeError:
        return st.container(**kwargs)


def ornament(symbol="❖"):
    st.markdown(
        f'<div class="ornament"><span>{symbol}</span></div>',
        unsafe_allow_html=True,
    )


def section_title(deva, english):
    st.markdown(
        f'<div class="section-title"><span class="deva">{deva}</span>'
        f'<span class="eng">{english}</span></div>',
        unsafe_allow_html=True,
    )


def roman(n):
    out = ""
    for value, symbol in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out += symbol
            n -= value
    return out or "—"


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


def format_inr(usd_amount):
    """Convert a USD amount to a rupee string at USD_TO_INR."""
    try:
        inr = float(usd_amount) * USD_TO_INR
    except (TypeError, ValueError):
        inr = 0.0
    if inr >= 1:
        return f"₹{inr:,.2f}"
    return f"₹{inr:.4f}"


def kv_row(label, value, code=False):
    text = html.escape(str(value))
    if code:
        text = f"<code>{text}</code>"
    return f'<div class="kv"><span class="k">{label}</span><span class="v">{text}</span></div>'


def build_copy_text(question, answer):
    return (
        f"Question: {question.strip()}\n\n"
        f"Answer: {answer.strip()}\n\n"
        f"Check it out - {SHARE_LINK}"
    )


def render_copy_button(text):
    """
    One-click copy using the browser clipboard (with an execCommand fallback).
    Rendered in a small iframe, so it carries its own styling.
    """
    payload = json.dumps(text).replace("</", "<\\/")

    components.html(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&display=swap');
          body {{ margin: 0; background: transparent; }}
          #copy {{
            font-family: 'Cinzel', Georgia, serif;
            font-size: 13px; letter-spacing: 0.16em; text-transform: uppercase;
            color: #FFF3D6; background: #7A1F1F;
            border: 1px solid #B38A2E; border-radius: 2px;
            padding: 11px 22px; cursor: pointer;
            box-shadow: inset 0 0 0 2px #7A1F1F, inset 0 0 0 3px rgba(255,243,214,.45);
            transition: background .2s ease;
          }}
          #copy:hover {{ background: #5A1414; }}
          #copy.ok {{ background: #3F5B2B; }}
        </style>
        <button id="copy">❐ &nbsp;Copy question, answer &amp; link</button>
        <script>
          const text = {payload};
          const btn = document.getElementById("copy");
          const original = btn.innerHTML;

          async function doCopy() {{
            try {{
              await navigator.clipboard.writeText(text);
              return true;
            }} catch (err) {{
              const ta = document.createElement("textarea");
              ta.value = text;
              ta.style.position = "fixed";
              ta.style.opacity = "0";
              document.body.appendChild(ta);
              ta.focus(); ta.select();
              let ok = false;
              try {{ ok = document.execCommand("copy"); }} catch (e) {{ ok = false; }}
              document.body.removeChild(ta);
              return ok;
            }}
          }}

          btn.addEventListener("click", async () => {{
            const ok = await doCopy();
            btn.classList.toggle("ok", ok);
            btn.innerHTML = ok ? "✓ &nbsp;Copied" : "Copy blocked - use the box below";
            setTimeout(() => {{ btn.classList.remove("ok"); btn.innerHTML = original; }}, 2400);
          }});
        </script>
        """,
        height=58,
    )


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def display_ranked_sources(retrieval_data, show_scores=True):
    """Display final candidates returned by the retriever/graph."""
    if not retrieval_data:
        st.info("No ranked source information available.")
        return

    for index, item in enumerate(retrieval_data, start=1):
        source = safe_value(item.get("source"), "Unknown Source")
        collection = safe_value(item.get("collection"), "Unknown Collection")

        with st.expander(
            f"{roman(index)}.  {source}  ·  {collection}",
            expanded=(index == 1),
        ):
            card = (
                '<div class="source-card">'
                f'<div class="source-title">Source {roman(index)}</div>'
                + kv_row("Collection", collection)
                + kv_row("Source", source)
                + kv_row("Source type", safe_value(item.get("source_type")))
                + kv_row("Authority", safe_value(item.get("authority")))
                + kv_row("Page", safe_value(item.get("page_number")))
                + kv_row("Citation", safe_value(item.get("citation")))
                + kv_row("Chunk ID", safe_value(item.get("chunk_id")), code=True)
                + "</div>"
            )
            st.markdown(card, unsafe_allow_html=True)

            if show_scores:
                score_col1, score_col2, score_col3 = st.columns(3)
                with score_col1:
                    st.metric("Vector Rank", safe_value(item.get("vector_rank"), "N/A"))
                with score_col2:
                    st.metric("Vector Score", format_score(item.get("vector_score")))
                with score_col3:
                    st.metric("Reranker Score", format_score(item.get("reranker_score")))


def _context_html(value):
    return '<div class="context-box">' + html.escape(str(value)) + "</div>"


def display_context(context_data):
    """Safely display the exact context supplied to the LLM."""
    if context_data is None:
        st.info("No context data available.")
        return

    if isinstance(context_data, list):
        for index, item in enumerate(context_data, start=1):
            with st.expander(f"Context {roman(index)}"):
                if isinstance(item, dict):
                    for key, value in item.items():
                        st.markdown(f"**{key}:**")
                        st.markdown(_context_html(value), unsafe_allow_html=True)
                else:
                    st.markdown(_context_html(item), unsafe_allow_html=True)
        return

    if isinstance(context_data, dict):
        for key, value in context_data.items():
            st.markdown(f"**{key}:**")
            st.markdown(_context_html(value), unsafe_allow_html=True)
        return

    st.markdown(_context_html(context_data), unsafe_allow_html=True)


def display_document_metadata(documents):
    """Display all useful metadata from the final LangChain Documents."""
    if not documents:
        st.info("No document metadata available.")
        return

    # Ordered for readability. Any additional fields are shown afterwards.
    preferred_fields = [
        "collection", "source", "source_name", "source_type", "document_type",
        "authority", "tradition", "upanishad", "upanishad_short_name", "author",
        "commentary", "commentator", "form_name", "name", "name_devanagari",
        "section", "chapter", "chapter_number", "chapter_title", "mantra_number",
        "shloka_number", "page_number", "citation", "file_name", "file_path",
        "chunk_id", "chunk_index", "total_chunks",
    ]

    for index, document in enumerate(documents, start=1):
        if hasattr(document, "metadata"):
            metadata = document.metadata or {}
        elif isinstance(document, dict):
            metadata = document.get("metadata", {}) or {}
        else:
            metadata = {}

        if isinstance(metadata, dict) and isinstance(metadata.get("metadata"), dict):
            metadata = metadata["metadata"]

        source = metadata.get("source", "Unknown Source")

        with st.expander(f"{roman(index)}.  {source}"):
            shown = set()
            rows = ""

            for key in preferred_fields:
                value = metadata.get(key)
                if value is None or value == "" or value == -1:
                    continue
                shown.add(key)
                rows += kv_row(key.replace("_", " ").title(), value)

            for key, value in metadata.items():
                if key in shown or key == "metadata":
                    continue
                if value is None or value == "" or value == -1:
                    continue
                rows += kv_row(key.replace("_", " ").title(), value)

            st.markdown(f'<div class="source-card">{rows}</div>', unsafe_allow_html=True)


def display_usage(usage):
    """Display aggregated OpenRouter usage; cost is shown in rupees."""
    if not usage:
        st.info("No token usage information returned.")
        return

    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)
    reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)
    cached_tokens = int(usage.get("cached_tokens", 0) or 0)

    try:
        cost_usd = float(usage.get("cost", 0) or 0)
    except (TypeError, ValueError):
        cost_usd = 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Input Tokens", f"{prompt_tokens:,}")
    with col2:
        st.metric("Output Tokens", f"{completion_tokens:,}")
    with col3:
        st.metric("Total Tokens", f"{total_tokens:,}")
    with col4:
        st.metric(
            "Cost",
            format_inr(cost_usd),
            help=f"Converted from US dollars at $1 = ₹{USD_TO_INR}.",
        )

    st.caption(f"Cost converted at $1 = ₹{USD_TO_INR}.")

    with st.expander("Detailed usage"):
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.metric("Reasoning Tokens", f"{reasoning_tokens:,}")
        with detail_col2:
            st.metric("Cached Tokens", f"{cached_tokens:,}")


# ============================================================
# MASTHEAD
# ============================================================

st.markdown(
    """
    <div class="toran"></div>
    <div class="masthead">
        <div class="invocation deva">॥ श्री गणेशाय नमः ॥</div>
        <div class="om">ॐ</div>
        <div class="title">Ganesh Tattvagyan</div>
        <div class="title-deva deva">गणेश तत्त्वज्ञान</div>
        <div class="subtitle">
            Ask a question and receive an answer grounded in the Purāṇas,
            Upaniṣads, Sahasranāma and other sacred texts of Śrī Gaṇeśa.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
ornament()

if not initialized:
    st.error(
        "The Agentic RAG graph could not be initialised. "
        "Please check your environment variables and backend configuration."
    )
    if initialization_error:
        st.code(initialization_error)


# ============================================================
# QUERY FORM
# ============================================================

with st.form("query_form", clear_on_submit=False):
    col1, col2 = st.columns([3.4, 1])

    with col1:
        query = st.text_input(
            "Pose your question",
            key="query_input",
            placeholder="e.g., Who is Herambha?",
            help="Ask a question about the Ganesha knowledge base.",
        )

    with col2:
        st.write("")
        st.write("")
        search_button = st.form_submit_button(
            "Search",
            type="primary",
            use_container_width=True,
        )

run_pending = st.session_state.pop("run_pending", False)


# ============================================================
# READING PREFERENCES  (formerly the sidebar display options)
# ============================================================

with st.expander("Reading preferences", expanded=False):
    pref_col1, pref_col2 = st.columns(2)
    with pref_col1:
        show_sources = st.checkbox("Show ranked sources", value=True, key="opt_sources")
        show_scores = st.checkbox("Show retrieval scores", value=False, key="opt_scores")
        show_metadata = st.checkbox("Show detailed metadata", value=True, key="opt_metadata")
    with pref_col2:
        show_raw_context = st.checkbox("Show raw context", value=False, key="opt_context")
        show_token_usage = st.checkbox("Show token usage and cost", value=True, key="opt_usage")


# ============================================================
# RUN QUERY  (result is kept in session_state so toggling an
# option or pressing Copy never wipes the answer)
# ============================================================

def run_query(question):
    with st.spinner("Consulting the scriptures — retrieving, reranking, grading…"):
        try:
            result = graph.invoke(
                {
                    "question": question,
                    "rewrite_count": 0,
                }
            )
            st.session_state["last"] = {"query": question, "result": result, "error": None}
        except Exception as e:
            st.session_state["last"] = {"query": question, "result": None, "error": e}


if search_button or run_pending:
    clean_query = (query or "").strip()

    if not clean_query:
        st.warning("Please enter a question first.")
    elif not initialized or graph is None:
        st.error(
            "The Agentic RAG graph is not initialised. "
            "Please check your environment variables and backend configuration."
        )
    else:
        run_query(clean_query)


# ============================================================
# RENDER RESULT
# ============================================================

def render_result(question, result):
    ornament()

    # ---------------- Question ----------------
    section_title("प्रश्नः", "Question")
    st.markdown(
        f'<div class="question-quote">{html.escape(question)}</div>',
        unsafe_allow_html=True,
    )

    # ---------------- Answer ----------------
    section_title("उत्तरम्", "Answer")
    answer = result.get("answer") or "No answer generated."

    with keyed_container("answer_card"):
        st.markdown(answer)

    # ---------------- Copy ----------------
    copy_text = build_copy_text(question, answer)
    render_copy_button(copy_text)
    st.markdown(
        '<div class="copy-note">Copies the question, the answer and a link to this app.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Copy manually"):
        st.code(copy_text, language=None)

    # ---------------- Colophon ----------------
    model = result.get("model")
    search_query = result.get("search_query")
    rewrite_count = result.get("rewrite_count", 0)

    st.markdown(
        '<div class="colophon">Composed by '
        f'<b>{html.escape(str(model) if model else "the default model")}</b>'
        f" &nbsp;·&nbsp; query rewrites: <b>{html.escape(str(rewrite_count))}</b></div>",
        unsafe_allow_html=True,
    )

    if search_query and search_query.strip() != question.strip():
        st.caption("Final search query used for retrieval:")
        st.code(search_query, language="text")

    # ---------------- Usage / cost ----------------
    if show_token_usage:
        section_title("व्ययः", "Usage &amp; Cost")
        display_usage(result.get("usage", {}))

    # ---------------- Sources ----------------
    retrieval_data = result.get("retrieval", [])

    section_title("प्रमाणानि", "Sources")
    if retrieval_data:
        if not show_sources:
            st.caption(f"{len(retrieval_data)} final source(s) selected by the retriever.")
        else:
            st.caption(
                "The passages below were selected after vector retrieval "
                "and OpenRouter reranking."
            )
            display_ranked_sources(retrieval_data, show_scores=show_scores)
    else:
        st.info("No retrieval information was returned.")

    # ---------------- Detailed metadata ----------------
    documents = result.get("documents", [])
    if show_sources and show_metadata and documents:
        with st.expander("Detailed source metadata", expanded=False):
            display_document_metadata(documents)

    # ---------------- Raw context ----------------
    if show_raw_context:
        with st.expander("Raw context sent to the model", expanded=False):
            display_context(result.get("context"))

    # ---------------- Logs ----------------
    errors = result.get("errors", [])
    if errors:
        with st.expander("Model / retrieval logs", expanded=False):
            for error in errors:
                st.warning(str(error))


last = st.session_state.get("last")

if last and last.get("error") is not None:
    st.error(f"Error processing query: {last['error']}")
    with st.expander("Debug information", expanded=False):
        st.exception(last["error"])

elif last and last.get("result") is not None:
    render_result(last["query"], last["result"])

else:
    # ------------------------------------------------------------
    # Example questions (click to ask)
    # ------------------------------------------------------------
    def use_example(example_question):
        st.session_state["query_input"] = example_question
        st.session_state["run_pending"] = True

    section_title("प्रश्नाः", "Questions to begin with")
    with keyed_container("examples"):
        example_cols = st.columns(3)
        for col, (category, example) in zip(example_cols, EXAMPLES):
            with col:
                st.markdown(f'<div class="eg-cat">{category}</div>', unsafe_allow_html=True)
                st.button(
                    example,
                    key=f"eg_{category}",
                    on_click=use_example,
                    args=(example,),
                    use_container_width=True,
                )


# ============================================================
# ABOUT  (formerly the sidebar)
# ============================================================

ornament()

with st.expander("About this pipeline", expanded=False):
    st.markdown(
        """
        **1. Vector retrieval** — the query is embedded once and Astra DB collections are searched in parallel.

        **2. OpenRouter reranking** — candidate passages are reranked through the OpenRouter rerank API.

        **3. Document grading** — retrieved passages are checked for relevance.

        **4. Query rewriting** — if retrieval is insufficient, the query is rewritten and searched again.

        **5. Grounded generation** — the final answer is composed only from the selected source context.
        """
    )

    st.markdown(
        """
        **Database:** `ganesa_data` · **Keyspace:** `default_keyspace`

        **Collections:** `puranas`, `research`, `iconography`, `rahasya`, `sahastranaam`, `upanishads`

        **Embedding:** `all-MiniLM-L6-v2` · **Reranking:** OpenRouter `/rerank` API
        """
    )

    rerank_models = [
        m.strip()
        for m in os.getenv(
            "RERANK_MODELS",
            "cohere/rerank-v3.5,voyageai/rerank-2.5-lite,qwen/qwen3-reranker-8b",
        ).split(",")
        if m.strip()
    ]
    st.caption("Reranker models, in order of preference: " + ", ".join(f"`{m}`" for m in rerank_models))
    st.caption("Backend defaults: retrieve_k = 10 · rerank_per_collection = 5 · top_k = 6")
    st.caption(
        "Status: graph initialised ✓" if initialized else "Status: graph failed to initialise ✗"
    )

    if st.button("Reset Streamlit cache", help="Clear cached resources and rerun the app."):
        st.cache_resource.clear()
        st.session_state.pop("last", None)
        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer-mantra">ॐ गं गणपतये नमः</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Ganesh Tattvagyan · six-collection vector retrieval, "
    "OpenRouter reranking and agentic grading"
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

"""Streamlit entry point for the Lab Manual Q&A Agent.

Orchestrates the sidebar (upload + index) and the chat interface.
All business logic lives in src/; this file contains only thin glue code.
"""

import streamlit as st

import config
from src import agent, chunking, pdf_loader, retrieval, vectorstore
from src.utils import get_logger, hash_multiple_files, is_extraction_effectively_empty

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom CSS — premium dark glassmorphism design
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ───────────────────────────────────────────────────── */
:root {
    --bg-base:        #0a0d14;
    --bg-surface:     #111520;
    --bg-elevated:    #161b2e;
    --glass-bg:       rgba(22, 27, 46, 0.75);
    --glass-border:   rgba(99, 120, 255, 0.18);
    --accent-primary: #6378ff;
    --accent-violet:  #8b5cf6;
    --accent-cyan:    #22d3ee;
    --accent-green:   #10b981;
    --accent-amber:   #f59e0b;
    --accent-red:     #ef4444;
    --text-primary:   #f0f2ff;
    --text-secondary: #8b93b8;
    --text-muted:     #4b5280;
    --radius-sm:      8px;
    --radius-md:      14px;
    --radius-lg:      20px;
    --shadow-glow:    0 0 30px rgba(99, 120, 255, 0.12);
    --shadow-card:    0 8px 32px rgba(0, 0, 0, 0.45);
    --transition:     all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global base ──────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 0%, rgba(99,120,255,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(139,92,246,0.06) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stMain"] {
    background: transparent !important;
}

/* ── Streamlit header & sidebar collapse/expand toggle ────────────────── */
[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 100 !important;
}

#MainMenu, footer { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* Sidebar toggle buttons (expand/collapse) */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarHeader"] button {
    visibility: visible !important;
    display: flex !important;
    color: var(--accent-primary) !important;
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-sm) !important;
}

[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="stSidebarHeader"] button:hover {
    background: rgba(99,120,255,0.15) !important;
    border-color: var(--accent-primary) !important;
}

/* ── Main content padding ─────────────────────────────────────────────── */
[data-testid="stMainBlockContainer"] {
    padding: 2rem 2.5rem 6rem !important;
    max-width: 920px !important;
    margin: 0 auto !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--glass-bg) !important;
    border-right: 1px solid var(--glass-border) !important;
    backdrop-filter: blur(20px) !important;
}

[data-testid="stSidebarContent"] {
    padding: 1.5rem 1.25rem !important;
}

/* Sidebar header */
.sidebar-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.75rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--glass-border);
}

.sidebar-logo {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-violet));
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 4px 12px rgba(99,120,255,0.35);
    flex-shrink: 0;
}

.sidebar-title {
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--text-primary);
    line-height: 1.2;
}

.sidebar-subtitle {
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-weight: 500;
}

/* Sidebar section labels */
.sidebar-section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.25rem 0 0.5rem;
}

/* Indexed document pills */
.doc-pill {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: rgba(99,120,255,0.08);
    border: 1px solid rgba(99,120,255,0.15);
    border-radius: var(--radius-sm);
    margin-bottom: 0.4rem;
    font-size: 0.78rem;
    color: var(--text-secondary);
    transition: var(--transition);
}

.doc-pill-icon {
    color: var(--accent-primary);
    font-size: 0.85rem;
    flex-shrink: 0;
}

/* ── Buttons ──────────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.02em !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.55rem 1.1rem !important;
    transition: var(--transition) !important;
    border: 1px solid transparent !important;
}

/* Primary process button */
[data-testid="stButton"]:first-of-type > button {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-violet)) !important;
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(99,120,255,0.35) !important;
}

[data-testid="stButton"]:first-of-type > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,120,255,0.5) !important;
}

[data-testid="stButton"]:first-of-type > button:active {
    transform: translateY(0) !important;
}

/* Secondary / clear button */
[data-testid="stButton"]:not(:first-of-type) > button {
    background: transparent !important;
    border-color: rgba(239,68,68,0.35) !important;
    color: #ef6060 !important;
}

[data-testid="stButton"]:not(:first-of-type) > button:hover {
    background: rgba(239,68,68,0.08) !important;
    border-color: rgba(239,68,68,0.6) !important;
}

/* ── File uploader ────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(99,120,255,0.04) !important;
    border: 1.5px dashed rgba(99,120,255,0.25) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.5rem !important;
    transition: var(--transition) !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,120,255,0.5) !important;
    background: rgba(99,120,255,0.07) !important;
}

[data-testid="stFileUploader"] label {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
}

/* ── Text inputs (API key) ────────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    transition: var(--transition) !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 3px rgba(99,120,255,0.15) !important;
}

[data-testid="stTextInput"] label {
    color: var(--text-secondary) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}

/* ── Page header ──────────────────────────────────────────────────────── */
.page-header {
    text-align: center;
    margin-bottom: 2.5rem;
    padding: 2rem 0 1rem;
}

.page-header-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(99,120,255,0.1);
    border: 1px solid rgba(99,120,255,0.2);
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent-primary);
    margin-bottom: 1rem;
}

.page-header h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.15 !important;
    background: linear-gradient(135deg, #f0f2ff 0%, var(--accent-primary) 50%, var(--accent-violet) 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 0 0.75rem !important;
}

.page-header p {
    color: var(--text-secondary) !important;
    font-size: 0.95rem !important;
    max-width: 480px;
    margin: 0 auto !important;
    line-height: 1.6 !important;
}

/* ── Empty state ──────────────────────────────────────────────────────── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    padding: 3.5rem 2rem;
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-lg);
    backdrop-filter: blur(12px);
    text-align: center;
    box-shadow: var(--shadow-card);
    margin: 1rem 0;
}

.empty-state-icon {
    width: 64px;
    height: 64px;
    background: linear-gradient(135deg, rgba(99,120,255,0.15), rgba(139,92,246,0.15));
    border: 1px solid rgba(99,120,255,0.2);
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
}

.empty-state h3 {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
}

.empty-state p {
    color: var(--text-secondary) !important;
    font-size: 0.88rem !important;
    max-width: 340px;
    margin: 0 !important;
    line-height: 1.6 !important;
}

.empty-state-steps {
    display: flex;
    gap: 0.75rem;
    margin-top: 0.5rem;
    flex-wrap: wrap;
    justify-content: center;
}

.step-badge {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(99,120,255,0.07);
    border: 1px solid rgba(99,120,255,0.14);
    border-radius: 100px;
    padding: 0.35rem 0.85rem;
    font-size: 0.76rem;
    color: var(--text-secondary);
    font-weight: 500;
}

.step-num {
    width: 18px;
    height: 18px;
    background: var(--accent-primary);
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    font-weight: 700;
    color: #fff;
    flex-shrink: 0;
}

/* ── Chat messages ────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
}

/* User message bubble */
[data-testid="stChatMessage"][data-testid*="user"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: transparent !important;
}

.user-bubble {
    background: linear-gradient(135deg, rgba(99,120,255,0.15), rgba(139,92,246,0.12));
    border: 1px solid rgba(99,120,255,0.2);
    border-radius: var(--radius-md);
    border-bottom-right-radius: 4px;
    padding: 0.9rem 1.2rem;
    margin-left: 2rem;
    box-shadow: 0 2px 12px rgba(99,120,255,0.08);
}

/* Assistant message bubble */
.assistant-bubble {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    border-bottom-left-radius: 4px;
    padding: 1rem 1.25rem;
    margin-right: 2rem;
    box-shadow: var(--shadow-card);
    backdrop-filter: blur(12px);
}

[data-testid="stChatMessage"] .stMarkdown p {
    color: var(--text-primary) !important;
    line-height: 1.7 !important;
    font-size: 0.92rem !important;
}

/* Chat message avatars */
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-violet)) !important;
    border-radius: 10px !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-primary)) !important;
    border-radius: 10px !important;
}

/* ── Chat input ───────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: var(--glass-bg) !important;
    border: 1.5px solid var(--glass-border) !important;
    border-radius: var(--radius-md) !important;
    backdrop-filter: blur(16px) !important;
    box-shadow: 0 -4px 24px rgba(0,0,0,0.2), var(--shadow-glow) !important;
    transition: var(--transition) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 -4px 24px rgba(0,0,0,0.2), 0 0 0 3px rgba(99,120,255,0.15) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #8b93b8 !important;
    -webkit-text-fill-color: #8b93b8 !important;
}

/* ── Expander (Sources panel) ─────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(99,120,255,0.04) !important;
    border: 1px solid rgba(99,120,255,0.12) !important;
    border-radius: var(--radius-sm) !important;
    margin-top: 0.75rem !important;
}

[data-testid="stExpander"] summary {
    color: var(--accent-primary) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 0.9rem !important;
    letter-spacing: 0.02em !important;
}

[data-testid="stExpander"] summary:hover {
    background: rgba(99,120,255,0.06) !important;
    border-radius: var(--radius-sm) !important;
}

/* Source chunk cards inside expander */
.source-card {
    background: var(--bg-elevated);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
}

.source-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.4rem;
}

.source-filename {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--accent-primary);
    font-family: 'JetBrains Mono', monospace;
}

.source-page-badge {
    background: rgba(34,211,238,0.1);
    border: 1px solid rgba(34,211,238,0.2);
    color: var(--accent-cyan);
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 100px;
    letter-spacing: 0.04em;
}

.source-excerpt {
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.6;
    font-family: 'Inter', sans-serif;
}

/* ── Alert / info / success messages ─────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border-width: 1px !important;
    font-size: 0.85rem !important;
    font-family: 'Inter', sans-serif !important;
}

/* Info */
[data-testid="stAlert"][kind="info"] {
    background: rgba(99,120,255,0.07) !important;
    border-color: rgba(99,120,255,0.25) !important;
    color: #a5b4fc !important;
}

/* Success */
[data-testid="stAlert"][kind="success"] {
    background: rgba(16,185,129,0.07) !important;
    border-color: rgba(16,185,129,0.25) !important;
    color: #6ee7b7 !important;
}

/* Warning */
[data-testid="stAlert"][kind="warning"] {
    background: rgba(245,158,11,0.07) !important;
    border-color: rgba(245,158,11,0.25) !important;
    color: #fcd34d !important;
}

/* Error */
[data-testid="stAlert"][kind="error"] {
    background: rgba(239,68,68,0.07) !important;
    border-color: rgba(239,68,68,0.25) !important;
    color: #fca5a5 !important;
}

/* ── Spinner ──────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] {
    color: var(--accent-primary) !important;
}

/* ── Divider ──────────────────────────────────────────────────────────── */
hr {
    border-color: var(--glass-border) !important;
    margin: 0.5rem 0 !important;
}

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: rgba(99,120,255,0.25); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,120,255,0.45); }

/* ── Markdown improvements ────────────────────────────────────────────── */
.stMarkdown code {
    background: rgba(99,120,255,0.1) !important;
    border: 1px solid rgba(99,120,255,0.15) !important;
    border-radius: 4px !important;
    color: var(--accent-cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82em !important;
    padding: 0.1em 0.4em !important;
}

.stMarkdown pre {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 1rem !important;
}

.stMarkdown ol, .stMarkdown ul {
    padding-left: 1.4rem !important;
}

.stMarkdown li {
    color: var(--text-primary) !important;
    margin-bottom: 0.3rem !important;
    line-height: 1.65 !important;
}

/* ── Cursor blink animation for streaming ────────────────────────────── */
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}
</style>
"""


# ---------------------------------------------------------------------------
# Cached resource loaders (survive Streamlit reruns without recomputing)
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_embedding_function():
    """Load the sentence-transformer embedding model once per process."""
    return vectorstore.get_embedding_function()


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Ensure required keys exist in st.session_state on first run."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "collection" not in st.session_state:
        st.session_state.collection = None
    if "indexed_documents" not in st.session_state:
        st.session_state.indexed_documents = []
    if "bm25_index" not in st.session_state:
        st.session_state.bm25_index = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = None


# ---------------------------------------------------------------------------
# Handler functions
# ---------------------------------------------------------------------------

def handle_process_manuals(uploaded_files: list) -> None:
    """Wire together pdf_loader -> chunking -> vectorstore for the uploaded files,
    then update st.session_state with the collection and indexed document names.

    Shows a progress spinner. Surfaces specific st.error / st.warning messages
    for: extraction failure, empty/scanned PDF, missing API key. Never lets an
    unhandled exception reach the user as a raw traceback.
    """
    api_key = config.GEMINI_API_KEY or st.session_state.get("session_api_key", "")
    if not api_key:
        st.error(
            "Gemini API key is required. Add GEMINI_API_KEY to your .env file or enter it in the sidebar."
        )
        return

    file_data: list[tuple[bytes, str]] = []
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        file_data.append((file_bytes, uploaded_file.name))

    with st.spinner("Processing manual(s) — extracting, chunking, embedding…"):
        try:
            # --- PDF Extraction ---
            all_pages = pdf_loader.extract_multiple_pdfs(file_data)

            if not all_pages:
                st.error(
                    "No text could be extracted from the uploaded file(s). "
                    "This usually means the PDF contains only scanned images. "
                    "This app requires text-based PDFs (OCR is not supported)."
                )
                return

            page_texts = [p.text for p in all_pages]
            if is_extraction_effectively_empty(page_texts):
                st.warning(
                    "The extracted text is almost empty — the PDF may be scanned or image-only. "
                    "Answers may be poor or unavailable. Consider using a text-based PDF."
                )

            # --- Chunking ---
            chunks = chunking.chunk_pages(
                all_pages,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
            )

            if not chunks:
                st.error("Chunking produced no content. The PDF may be empty or unreadable.")
                return

            # --- Vector Store (reuse cached collection if same files re-uploaded) ---
            collection_name = "col_" + hash_multiple_files(
                [fb for fb, _ in file_data]
            )[:32]  # Chroma collection names must be <=63 chars

            if vectorstore.collection_exists(collection_name):
                st.info("Found existing index for these file(s) — skipping re-embedding.")
                collection = vectorstore.load_vectorstore(collection_name)
            else:
                collection = vectorstore.build_vectorstore(chunks, collection_name)

            # --- BM25 index (Phase 2, always rebuilt in-memory) ---
            bm25_index = retrieval.build_bm25_index(chunks)

            # --- Update session state ---
            st.session_state.collection = collection
            st.session_state.chunks = chunks
            st.session_state.bm25_index = bm25_index
            st.session_state.indexed_documents = [name for _, name in file_data]

            st.success(
                f"Indexed {len(chunks)} chunks from "
                f"{len(st.session_state.indexed_documents)} document(s). Ready to answer questions."
            )

        except Exception as exc:
            logger.exception("Error during manual processing: %s", exc)
            st.error(f"An error occurred while processing the manual: {exc}")


def handle_clear_knowledge_base() -> None:
    """Reset session state to clear the active index.

    Keeps persisted Chroma collections on disk (they are reused automatically
    if the same PDF is re-uploaded), but removes the in-session handles so
    the app returns to its empty state.
    """
    st.session_state.collection = None
    st.session_state.chunks = None
    st.session_state.bm25_index = None
    st.session_state.indexed_documents = []
    st.session_state.messages = []
    logger.info("Knowledge base cleared from session state.")


def handle_user_question(question: str) -> None:
    """Run retrieval + streaming agent call for a new user question.

    Appends both user and assistant messages to st.session_state.messages,
    renders the streamed answer, and shows an expandable Sources section
    with the exact chunks/pages used.
    """
    if st.session_state.collection is None:
        st.warning("Please upload and process a lab manual first.")
        return

    # Append user message to history and render it immediately.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Resolve API key (env or sidebar input).
    api_key = config.GEMINI_API_KEY or st.session_state.get("session_api_key", "")
    if not api_key:
        st.error(
            "Gemini API key is missing. Add GEMINI_API_KEY to your .env file or enter it in the sidebar."
        )
        return

    # Temporarily override config with sidebar key if env key is absent.
    if not config.GEMINI_API_KEY and api_key:
        config.GEMINI_API_KEY = api_key

    try:
        # Retrieve relevant chunks.
        retrieved_chunks = retrieval.retrieve(
            collection=st.session_state.collection,
            bm25_index=st.session_state.bm25_index,
            chunks=st.session_state.chunks or [],
            query=question,
        )

        # Build messages for Gemini (excluding current question, which retrieve adds).
        prior_history = [
            m for m in st.session_state.messages[:-1]  # exclude the question we just appended
            if m["role"] in ("user", "assistant")
        ]
        messages, system_prompt = agent.build_messages(question, retrieved_chunks, prior_history)

        # Stream the response.
        full_response = ""
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            for delta in agent.call_gemini_streaming(messages, system_prompt):
                full_response += delta
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

            # Faithfulness check + expandable sources panel.
            raw_citations = agent.extract_citations(full_response)
            verified_citations, stripped_count = agent.verify_citations(
                raw_citations, retrieved_chunks
            )

            if stripped_count > 0:
                logger.warning(
                    "%d citation(s) stripped by faithfulness check for question: '%s'",
                    stripped_count,
                    question[:80],
                )

            with st.expander("📄 Sources"):
                if retrieved_chunks:
                    for chunk in retrieved_chunks:
                        filename = chunk["source_filename"]
                        page = chunk["page_number"]
                        excerpt = chunk["text"][:300] + ("…" if len(chunk["text"]) > 300 else "")
                        st.markdown(
                            f"""<div class="source-card">
                                <div class="source-card-header">
                                    <span class="source-filename">📄 {filename}</span>
                                    <span class="source-page-badge">Page {page}</span>
                                </div>
                                <div class="source-excerpt">{excerpt}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("No source chunks were retrieved for this query.")

        # Persist assistant message to history.
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    except ValueError as exc:
        # Raised by call_gemini_streaming when the API key is missing.
        logger.error("ValueError in handle_user_question: %s", exc)
        st.error(str(exc))
    except Exception as exc:
        logger.exception("Unexpected error while answering question: %s", exc)
        st.error(f"An unexpected error occurred: {exc}")


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

def render_sidebar() -> list | None:
    """Render the sidebar: file uploader, Process button, indexed document
    list, Clear Knowledge Base button, and a masked API key input shown
    only if config.GEMINI_API_KEY is not set. Returns the list of newly
    uploaded files if the Process button was clicked, else None.
    """
    with st.sidebar:
        # Header
        st.markdown(
            """<div class="sidebar-header">
                <div class="sidebar-logo">🧪</div>
                <div>
                    <div class="sidebar-title">Lab Manual Q&A</div>
                    <div class="sidebar-subtitle">Powered by Gemini</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Upload section
        st.markdown('<div class="sidebar-section-label">Upload Manual</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        process_clicked = st.button(
            "Process Manual(s)",
            disabled=not uploaded_files,
            use_container_width=True,
        )

        # Indexed documents
        if st.session_state.indexed_documents:
            st.markdown('<div class="sidebar-section-label">Indexed Documents</div>', unsafe_allow_html=True)
            for name in st.session_state.indexed_documents:
                # Truncate long filenames
                display_name = name if len(name) <= 28 else name[:25] + "…"
                st.markdown(
                    f"""<div class="doc-pill">
                        <span class="doc-pill-icon">📄</span>
                        <span>{display_name}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Actions
        st.markdown('<div class="sidebar-section-label">Actions</div>', unsafe_allow_html=True)
        st.button("Clear Knowledge Base", on_click=handle_clear_knowledge_base, use_container_width=True)

        # API key input (only if not in .env)
        if not config.GEMINI_API_KEY:
            st.markdown('<div class="sidebar-section-label">API Key</div>', unsafe_allow_html=True)
            st.text_input(
                "Gemini API Key",
                type="password",
                key="session_api_key",
                placeholder="AIza...",
                help="Get a free key at aistudio.google.com/apikey",
            )

        # Footer
        st.markdown(
            """<div style="margin-top:2.5rem;border-top:1px solid rgba(99,120,255,0.12);padding-top:0.9rem;
                    font-size:0.72rem;color:#8b93b8;text-align:center;line-height:1.5;">
                    Answers are grounded strictly in your uploaded PDF.<br>
                    Citations verified by faithfulness check.
            </div>""",
            unsafe_allow_html=True,
        )

    return uploaded_files if process_clicked else None


def render_chat() -> None:
    """Render the page header, existing chat history, then the chat input.
    On new input, call handle_user_question. Show the empty-state when no
    document has been indexed yet.
    """
    # Page header
    st.markdown(
        """<div class="page-header">
            <div class="page-header-badge">✦ AI-Powered</div>
            <h1>Lab Manual Assistant</h1>
            <p>Upload your lab manual and ask questions. Get precise, cited answers grounded in your document.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Empty state
    if not st.session_state.indexed_documents:
        st.markdown(
            """<div class="empty-state">
                <div class="empty-state-icon">📚</div>
                <h3>No manual indexed yet</h3>
                <p>Upload a PDF in the sidebar to get started. Ask about procedures, safety, materials, or anything in your lab manual.</p>
                <div class="empty-state-steps">
                    <div class="step-badge"><span class="step-num">1</span> Upload PDF</div>
                    <div class="step-badge"><span class="step-num">2</span> Click Process</div>
                    <div class="step-badge"><span class="step-num">3</span> Ask anything</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    question = st.chat_input("Ask a question about your lab manual…")
    if question:
        handle_user_question(question)


def main() -> None:
    st.set_page_config(
        page_title="Lab Manual Q&A Agent",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    init_session_state()

    files_to_process = render_sidebar()
    if files_to_process:
        handle_process_manuals(files_to_process)

    render_chat()


if __name__ == "__main__":
    main()

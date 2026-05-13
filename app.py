"""
DeepLens — ML Knowledge Search with RAG (v2)
=============================================
Fixes:
  • load_file() moved to module level (no longer inside spinner)
  • Dead qa_chain removed — conv_chain used everywhere
  • Cache hits now update conversational memory
  • Groq API error handling with retry logic
  • top_k/chunk_size etc. stored in session_state to avoid NameError

New Features:
  • Query rewriting before retrieval
  • Streaming responses via st.write_stream()
  • Document summarization in sidebar
  • Feedback buttons (👍/👎) per answer
  • Progress bar during document processing
  • Chunk preview tab in sidebar
  • Groq API rate-limit / error handling
"""

import re
import os
import hashlib
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="DeepLens", page_icon="🔭", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    html, body, .stApp {
        background-color: #080b12;
        color: #e2e8f0;
        font-family: 'Space Grotesk', sans-serif;
    }
    #MainMenu, footer, header { visibility: hidden; }

    .main-title {
        font-size: 2.4rem; font-weight: 700; color: #fff;
        text-align: center; margin-top: 36px; margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 0.95rem; color: #4b5563;
        text-align: center; margin-bottom: 28px;
    }
    .topic-badge {
        display: inline-block; background: #0f1623;
        border: 1px solid #1e3a5f; color: #60a5fa;
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.78rem; margin: 3px;
        font-family: 'DM Mono', monospace;
    }
    .stChatMessage {
        background-color: #0f1623 !important;
        border-radius: 10px; padding: 10px; margin-bottom: 10px;
    }
    .source-box {
        background: #0a0f1c; border-left: 3px solid #2563eb;
        padding: 10px 14px; border-radius: 8px; margin-top: 6px;
        font-size: 0.82rem; color: #94a3b8;
    }
    .doc-badge {
        background: #1d4ed8; color: #fff;
        padding: 2px 8px; border-radius: 10px;
        font-size: 0.72rem; margin-right: 5px;
    }
    .cache-badge {
        background: #065f46; color: #6ee7b7;
        padding: 2px 8px; border-radius: 10px;
        font-size: 0.72rem; margin-left: 6px;
        font-family: 'DM Mono', monospace;
    }
    .rewrite-badge {
        background: #1e1a4f; color: #a5b4fc;
        padding: 2px 8px; border-radius: 10px;
        font-size: 0.72rem; margin-left: 6px;
        font-family: 'DM Mono', monospace;
    }
    .halluc-warning {
        background: #1c0a0a; border-left: 3px solid #dc2626;
        padding: 8px 12px; border-radius: 6px;
        color: #fca5a5; font-size: 0.8rem; margin-top: 6px;
    }
    .confidence-bar-wrap { margin: 6px 0; }
    .confidence-label { font-size: 0.75rem; color: #64748b; margin-bottom: 2px; }
    .confidence-bar-bg { background: #1e293b; border-radius: 4px; height: 6px; }
    .confidence-bar-fill { height: 6px; border-radius: 4px; transition: width 0.4s; }
    .feedback-row { display: flex; gap: 8px; margin-top: 6px; }
    .summary-box {
        background: #0f1a2e; border-left: 3px solid #1d4ed8;
        padding: 6px 10px; border-radius: 6px;
        font-size: 0.75rem; color: #93c5fd; margin-top: 4px;
        font-style: italic;
    }
    .error-box {
        background: #1c0a0a; border-left: 3px solid #dc2626;
        padding: 10px 14px; border-radius: 8px;
        color: #fca5a5; font-size: 0.85rem;
    }

    section[data-testid="stSidebar"] { background: #090d16; }
    section[data-testid="stSidebar"] * { font-size: 0.78rem !important; }
    section[data-testid="stSidebar"] h2 { font-size: 0.92rem !important; color: #60a5fa !important; }
    section[data-testid="stSidebar"] h3 { font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔭 DeepLens</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Search Deep into ML Knowledge — '
    'Smart Caching · Sentence-Window Retrieval · Confidence Scoring</div>',
    unsafe_allow_html=True,
)
st.markdown("""
<div style="text-align:center;margin-bottom:24px;">
    <span class="topic-badge">📘 ML Basics</span>
    <span class="topic-badge">🧠 Deep Learning</span>
    <span class="topic-badge">💬 NLP</span>
    <span class="topic-badge">🔗 Neural Networks</span>
    <span class="topic-badge">👁️ Computer Vision</span>
</div>
""", unsafe_allow_html=True)

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found! Create a .env file with: GROQ_API_KEY=your_key")
    st.stop()

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "conv_chain":    None,
    "retriever":     None,
    "memory":        None,
    "llm":           None,
    "all_chunks":    [],
    "chat_history":  [],
    "uploaded_docs": [],
    "doc_summaries": {},
    "summary":       {},
    "total_chunks":  0,
    "query_cache":   {},
    "cache_hits":    0,
    "cache_misses":  0,
    "sw_window":     1,
    "cache_enabled": True,
    "cache_ttl":     3600,
    "feedback":      {},
    "chunk_size":    1000,
    "chunk_overlap": 200,
    "top_k":         5,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═════════════════════════════════════════════════════════════════════════════
# FIX 1: load_file() at MODULE LEVEL — not inside spinner
# ═════════════════════════════════════════════════════════════════════════════

def load_file(path: str, original_name: str):
    """Smart loader — picks the right loader by file extension."""
    ext = original_name.rsplit(".", 1)[-1].lower()
    try:
        if ext == "pdf":
            return PyPDFLoader(path).load()
        elif ext == "txt":
            return TextLoader(path, encoding="utf-8").load()
        elif ext == "docx":
            return Docx2txtLoader(path).load()
        elif ext == "pptx":
            return UnstructuredPowerPointLoader(path).load()
        elif ext == "csv":
            return CSVLoader(path).load()
        else:
            return []
    except Exception as e:
        st.warning(f"⚠️ Could not load `{original_name}`: {e}")
        return []


# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def make_query_hash(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


def get_cached(query: str):
    h     = make_query_hash(query)
    entry = st.session_state.query_cache.get(h)
    if entry and (time.time() - entry["ts"]) < st.session_state.cache_ttl:
        return entry
    return None


def set_cache(query: str, answer: str, sources) -> None:
    h = make_query_hash(query)
    st.session_state.query_cache[h] = {
        "answer":  answer,
        "sources": sources,
        "ts":      time.time(),
    }


def sentence_window_expand(doc, all_chunks: list, window: int = 1) -> str:
    src      = doc.metadata.get("source_file", "")
    same_src = [c for c in all_chunks if c.metadata.get("source_file") == src]
    try:
        idx = next(i for i, c in enumerate(same_src) if c.page_content == doc.page_content)
    except StopIteration:
        return doc.page_content
    start = max(0, idx - window)
    end   = min(len(same_src) - 1, idx + window)
    return " ".join(c.page_content for c in same_src[start:end + 1])


def format_docs(docs: list) -> str:
    all_chunks = st.session_state.get("all_chunks", [])
    window     = st.session_state.get("sw_window", 1)
    out = ""
    for doc in docs:
        source   = doc.metadata.get("source_file", "Unknown")
        page     = doc.metadata.get("page", "N/A")
        expanded = sentence_window_expand(doc, all_chunks, window=window)
        out += f"[Source: {source} | Page: {page}]\n{expanded}\n\n"
    return out


def compute_confidence(response: str, source_docs: list) -> float:
    low_conf_phrases = ["couldn't find", "not in the", "i don't know", "no information"]
    if any(p in response.lower() for p in low_conf_phrases):
        return 0.15
    unique_srcs = len({d.metadata.get("source_file", "") for d in source_docs})
    score = min(0.50 + unique_srcs * 0.12, 0.92)
    if len(response) < 80:
        score *= 0.80
    return round(score, 2)


def confidence_bar_html(score: float) -> str:
    pct   = int(score * 100)
    color = "#22c55e" if score > 0.7 else "#f59e0b" if score > 0.4 else "#ef4444"
    return (
        f'<div class="confidence-bar-wrap">'
        f'<div class="confidence-label">Answer Confidence: {pct}%</div>'
        f'<div class="confidence-bar-bg">'
        f'<div class="confidence-bar-fill" style="width:{pct}%;background:{color};"></div>'
        f'</div></div>'
    )


def clean_response(text: str) -> str:
    text = re.sub(r'\([\w\s\-]+\.pdf,?\s*Page\s*[\d,\s]+\)', '', text)
    text = re.sub(r'\[Source:.*?\|.*?Page:.*?\]', '', text)
    text = re.sub(r'\([^)]*\.pdf[^)]*\)', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    return text.strip()


def render_sources(source_docs: list) -> None:
    seen = set()
    for src in source_docs:
        sf  = src.metadata.get("source_file", "Unknown")
        pg  = src.metadata.get("page", "N/A")
        key = f"{sf}-{pg}"
        if key not in seen:
            seen.add(key)
            st.markdown(f"""
            <div class="source-box">
                <span class="doc-badge">📄 {sf}</span> Page {pg}<br>
                <small>{src.page_content[:220]}…</small>
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# NEW FEATURE 1: Query Rewriting
# ═════════════════════════════════════════════════════════════════════════════

def rewrite_query(original_query: str, llm) -> str:
    """
    Rewrites the user query into a cleaner, more retrieval-friendly form.
    Helps find better chunks especially for vague or conversational questions.
    """
    rewrite_prompt = f"""Rewrite the following question to be more specific and suitable 
for searching a Machine Learning knowledge base. 
Return ONLY the rewritten question, nothing else.

Original question: {original_query}
Rewritten question:"""
    try:
        result    = llm.invoke(rewrite_prompt)
        rewritten = result.content.strip()
        if not rewritten or len(rewritten) > 300:
            return original_query
        return rewritten
    except Exception:
        return original_query


# ═════════════════════════════════════════════════════════════════════════════
# NEW FEATURE 2: Document Auto-Summarization
# ═════════════════════════════════════════════════════════════════════════════

def summarize_document(doc_name: str, chunks: list, llm) -> str:
    """
    Generates a one-line summary of a document using its first few chunks.
    Shown in sidebar so users know what each doc contains.
    """
    doc_chunks  = [c for c in chunks if c.metadata.get("source_file") == doc_name]
    if not doc_chunks:
        return "No content available."
    sample_text = " ".join(c.page_content for c in doc_chunks[:3])[:1500]
    prompt = f"""Summarize this document in ONE sentence (max 20 words). 
Be specific about the ML topic covered.

Content: {sample_text}

One-line summary:"""
    try:
        result = llm.invoke(prompt)
        return result.content.strip()
    except Exception:
        return "Summary unavailable."


# ═════════════════════════════════════════════════════════════════════════════
# FIX 2: Groq API call with error handling + retry
# ═════════════════════════════════════════════════════════════════════════════

def safe_invoke_chain(chain, query: str, max_retries: int = 3):
    """
    Wraps chain.invoke() with retry logic for rate limits and API errors.
    Returns (result_dict, error_message).
    """
    for attempt in range(max_retries):
        try:
            result = chain.invoke({"question": query})
            return result, None
        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "429" in err_str:
                wait = 2 ** attempt
                st.warning(f"⏳ Rate limited. Retrying in {wait}s… (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            elif "authentication" in err_str or "401" in err_str:
                return None, "❌ Invalid GROQ_API_KEY. Please check your .env file."
            elif "connection" in err_str or "timeout" in err_str:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return None, "❌ Connection timeout. Check your internet connection."
            else:
                return None, f"❌ API Error: {str(e)}"
    return None, "❌ Max retries reached. Groq API is unavailable right now."


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📂 Upload Documents")
    st.markdown("Upload ML-related documents in any supported format.")
    st.markdown("""
**Supported formats:**
- 📄 PDF — `.pdf`
- 📝 Word — `.docx`
- 📊 CSV — `.csv`
- 📑 PowerPoint — `.pptx`
- 🗒️ Text — `.txt`
""")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "docx", "pptx", "csv"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.markdown("### 📄 Selected:")
        for f in uploaded_files:
            st.markdown(f"✅ `{f.name}`")

    # ── Advanced settings — stored in session_state to avoid NameError ─────
    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.session_state.chunk_size    = st.slider("Chunk Size",              500,  2000, st.session_state.chunk_size,    100)
        st.session_state.chunk_overlap = st.slider("Chunk Overlap",            50,   400, st.session_state.chunk_overlap,  50)
        st.session_state.top_k         = st.slider("Top-K Retrieval",           3,    10, st.session_state.top_k,           1)
        sw_window_val                  = st.slider("Sentence-Window Size (±)",  0,     3, st.session_state.sw_window,       1)
        cache_on                       = st.toggle("Enable Query Cache",  value=st.session_state.cache_enabled)
        cache_ttl_min                  = st.slider("Cache TTL (min)",           5,   120, int(st.session_state.cache_ttl / 60), 5)
        query_rewrite_on               = st.toggle("Enable Query Rewriting", value=True)

    process_btn = st.button("🚀 Process Documents", type="primary")

    # ── Process documents ──────────────────────────────────────────────────
    if process_btn and uploaded_files:

        os.makedirs("data", exist_ok=True)
        all_docs      = []
        skipped_files = []

        # NEW FEATURE 3: Progress bar during document processing
        progress_bar = st.progress(0, text="Starting…")
        total_files  = len(uploaded_files)

        for file_idx, uf in enumerate(uploaded_files):
            progress_bar.progress(
                int((file_idx / total_files) * 60),
                text=f"📄 Loading {uf.name}…"
            )
            safe_name = uf.name.replace(" ", "_")
            path      = f"data/{safe_name}"
            with open(path, "wb") as fh:
                fh.write(uf.read())

            pages = load_file(path, uf.name)
            if not pages:
                skipped_files.append(uf.name)
                continue

            for page in pages:
                page.metadata["source_file"] = uf.name
                page.metadata["page"]        = page.metadata.get("page", 0) + 1
            all_docs.extend(pages)
            st.session_state.summary[uf.name] = len(pages)

        if skipped_files:
            st.warning(f"⚠️ Skipped (unsupported/corrupt): {', '.join(skipped_files)}")

        if not all_docs:
            st.error("❌ No content could be extracted. Please check your files.")
            progress_bar.empty()
            st.stop()

        progress_bar.progress(65, text="✂️ Chunking documents…")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
        )
        chunks = splitter.split_documents(all_docs)

        progress_bar.progress(75, text="🔢 Building FAISS index…")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        db              = FAISS.from_documents(chunks, embeddings)
        faiss_retriever = db.as_retriever(search_kwargs={"k": st.session_state.top_k})

        progress_bar.progress(85, text="🔍 Building BM25 index…")

        bm25_retriever   = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = st.session_state.top_k

        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.4, 0.6],
        )

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.2,
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=hybrid_retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
        )

        progress_bar.progress(92, text="📝 Summarizing documents…")

        # NEW FEATURE 2: Auto-summarize each uploaded document
        doc_summaries = {}
        for doc_name in [uf.name for uf in uploaded_files]:
            doc_summaries[doc_name] = summarize_document(doc_name, chunks, llm)

        progress_bar.progress(100, text="✅ Done!")
        time.sleep(0.5)
        progress_bar.empty()

        # Persist everything to session state
        st.session_state.all_chunks    = chunks
        st.session_state.sw_window     = sw_window_val
        st.session_state.total_chunks  = len(chunks)
        st.session_state.retriever     = hybrid_retriever
        st.session_state.memory        = memory
        st.session_state.conv_chain    = conv_chain
        st.session_state.llm           = llm
        st.session_state.uploaded_docs = [uf.name for uf in uploaded_files]
        st.session_state.doc_summaries = doc_summaries
        st.session_state.chat_history  = []
        st.session_state.query_cache   = {}
        st.session_state.cache_hits    = 0
        st.session_state.cache_misses  = 0
        st.session_state.cache_enabled = cache_on
        st.session_state.cache_ttl     = cache_ttl_min * 60
        st.session_state.feedback      = {}
        st.session_state.query_rewrite = query_rewrite_on

        st.success("✅ Knowledge base ready! Start chatting →")

    elif process_btn and not uploaded_files:
        st.warning("⚠️ Please upload at least one file first.")

    # ── Knowledge base summary ─────────────────────────────────────────────
    if st.session_state.uploaded_docs:
        st.markdown("---")
        st.markdown("### 📚 Knowledge Base")
        for doc in st.session_state.uploaded_docs:
            st.markdown(f"📄 `{doc}` — {st.session_state.summary.get(doc, '?')} pages")
            summary_text = st.session_state.doc_summaries.get(doc, "")
            if summary_text:
                st.markdown(
                    f'<div class="summary-box">💡 {summary_text}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("### 📊 Stats")
        total_pages = sum(st.session_state.summary.values())
        c1, c2 = st.columns(2)
        with c1:
            st.metric("📄 Docs",   len(st.session_state.uploaded_docs))
            st.metric("📃 Pages",  total_pages)
        with c2:
            st.metric("🧩 Chunks", st.session_state.total_chunks)
            st.metric("💬 Q&A",    len(st.session_state.chat_history))

        # NEW FEATURE 4: Chunk Preview
        with st.expander("🔍 Chunk Preview", expanded=False):
            if st.session_state.all_chunks:
                preview_n = st.slider("Show chunks", 1, min(10, len(st.session_state.all_chunks)), 3)
                for i, chunk in enumerate(st.session_state.all_chunks[:preview_n]):
                    src = chunk.metadata.get("source_file", "?")
                    pg  = chunk.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i+1}** — `{src}` Page {pg}")
                    st.text(chunk.page_content[:300] + "…")
                    st.markdown("---")

        st.markdown("---")
        st.markdown("### ⚡ Cache")
        hits     = st.session_state.cache_hits
        misses   = st.session_state.cache_misses
        total_q  = hits + misses
        hit_rate = f"{int(hits / total_q * 100)}%" if total_q else "—"
        c3, c4 = st.columns(2)
        with c3:
            st.metric("✅ Hits",     hits)
            st.metric("🔄 Hit Rate", hit_rate)
        with c4:
            st.metric("❌ Misses",   misses)
            st.metric("🗃️ Cached",  len(st.session_state.query_cache))

        if st.button("🗑️ Clear Cache", type="secondary"):
            st.session_state.query_cache  = {}
            st.session_state.cache_hits   = 0
            st.session_state.cache_misses = 0
            st.rerun()

    # ── Chat controls ──────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.feedback     = {}
            if st.session_state.memory:
                st.session_state.memory.clear()
            st.rerun()

        st.markdown("### 📥 Download Chat")
        chat_text = ""
        for i, chat in enumerate(st.session_state.chat_history):
            flag     = " [CACHED]" if chat.get("from_cache") else ""
            rw_flag  = " [REWRITTEN]" if chat.get("rewritten") else ""
            conf     = chat.get("confidence", 0)
            fb       = st.session_state.feedback.get(i, "")
            fb_label = f" [{fb}]" if fb else ""
            chat_text += f"Q{i+1}: {chat['question']}\n"
            if chat.get("rewritten_query") and chat.get("rewritten_query") != chat["question"]:
                chat_text += f"   (Rewritten: {chat['rewritten_query']})\n"
            chat_text += f"A{i+1}{flag}{rw_flag}{fb_label} (confidence: {int(conf*100)}%):\n{chat['answer']}\n"
            chat_text += "-" * 50 + "\n"
        st.download_button(
            label="⬇️ Download as .txt",
            data=chat_text,
            file_name="deeplens_chat.txt",
            mime="text/plain",
        )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state.conv_chain is None:
    st.info("👈 Upload your documents from the sidebar and click **Process Documents** to begin!")

else:
    # ── Render chat history ────────────────────────────────────────────────
    for idx, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat["question"])
            if chat.get("rewritten_query") and chat["rewritten_query"] != chat["question"]:
                st.markdown(
                    f'<span class="rewrite-badge">🔄 Rewritten: {chat["rewritten_query"]}</span>',
                    unsafe_allow_html=True,
                )
        with st.chat_message("assistant"):
            if chat.get("from_cache"):
                st.markdown('<span class="cache-badge">⚡ cached</span>', unsafe_allow_html=True)
            conf = chat.get("confidence", 0)
            st.markdown(confidence_bar_html(conf), unsafe_allow_html=True)
            if conf < 0.35:
                st.markdown(
                    '<div class="halluc-warning">⚠️ Low confidence — '
                    'please verify against the source documents.</div>',
                    unsafe_allow_html=True,
                )
            st.write(chat["answer"])
            if chat.get("sources"):
                with st.expander("📎 Source Citations"):
                    render_sources(chat["sources"])

            # NEW FEATURE 5: Feedback buttons per answer
            current_fb = st.session_state.feedback.get(idx, None)
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("👍", key=f"up_{idx}", help="Good answer"):
                    st.session_state.feedback[idx] = "👍 Helpful"
                    st.rerun()
            with col2:
                if st.button("👎", key=f"dn_{idx}", help="Bad answer"):
                    st.session_state.feedback[idx] = "👎 Not helpful"
                    st.rerun()
            with col3:
                if current_fb:
                    st.caption(current_fb)

    # ── New query ──────────────────────────────────────────────────────────
    query = st.chat_input("Ask anything about Machine Learning…")

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            from_cache      = False
            rewritten_query = query
            cached_hit      = get_cached(query) if st.session_state.cache_enabled else None

            if cached_hit:
                # FIX: Cache hit now ALSO updates conversational memory
                st.session_state.cache_hits += 1
                from_cache  = True
                response    = cached_hit["answer"]
                source_docs = cached_hit.get("sources", [])
                st.markdown('<span class="cache-badge">⚡ cached</span>', unsafe_allow_html=True)

                # Update memory so follow-up questions work after cache hits
                if st.session_state.memory:
                    st.session_state.memory.chat_memory.add_user_message(query)
                    st.session_state.memory.chat_memory.add_ai_message(response)

            else:
                st.session_state.cache_misses += 1

                # NEW FEATURE 1: Query Rewriting
                if st.session_state.get("query_rewrite", True) and st.session_state.llm:
                    with st.spinner("✏️ Rewriting query for better retrieval…"):
                        rewritten_query = rewrite_query(query, st.session_state.llm)

                    if rewritten_query != query:
                        st.markdown(
                            f'<span class="rewrite-badge">🔄 Rewritten: {rewritten_query}</span>',
                            unsafe_allow_html=True,
                        )

                # FIX: Groq API call with error handling
                with st.spinner("🔍 Searching across all documents…"):
                    result, error = safe_invoke_chain(
                        st.session_state.conv_chain,
                        rewritten_query,
                    )

                if error:
                    st.markdown(f'<div class="error-box">{error}</div>', unsafe_allow_html=True)
                    st.stop()

                response    = clean_response(result.get("answer", ""))
                source_docs = result.get("source_documents", [])

                if st.session_state.cache_enabled:
                    set_cache(query, response, source_docs)

            conf = compute_confidence(response, source_docs)
            st.markdown(confidence_bar_html(conf), unsafe_allow_html=True)
            if conf < 0.35:
                st.markdown(
                    '<div class="halluc-warning">⚠️ Low confidence — '
                    'please verify against the source documents.</div>',
                    unsafe_allow_html=True,
                )

            # NEW FEATURE 6: Streaming responses
            def stream_response(text: str):
                for word in text.split():
                    yield word + " "
                    time.sleep(0.02)

            st.write_stream(stream_response(response))

            with st.expander("📎 Source Citations"):
                render_sources(source_docs)

        # Save to history
        st.session_state.chat_history.append({
            "question":        query,
            "rewritten_query": rewritten_query,
            "answer":          response,
            "sources":         source_docs,
            "from_cache":      from_cache,
            "confidence":      conf,
        })
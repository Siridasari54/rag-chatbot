# app.py — FULL REPLACEMENT

from ui.styles import apply_styles
apply_styles()

import streamlit as st
from config.settings import init_session_state, groq_api_key
from ui.sidebar import render_sidebar
from ui.chat_ui import render_chat
from utils.persistence import (
    init_db, load_chat_history, load_retriever,
    load_doc_meta, retriever_exists
)
from rag.chains import build_conv_chain
from services.groq_services import build_llm

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔭 DeepLens</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Search Deep into ML Knowledge — '
    'RAGAS Evaluation · Smart Caching · Hybrid Retrieval · Confidence Scoring</div>',
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
    st.error(
        "❌ GROQ_API_KEY not found! "
        "Create a `.env` file in the project root with:\n\n"
        "```\nGROQ_API_KEY=your_key_here\n```"
    )
    st.stop()

# ── Init session + DB ─────────────────────────────────────────────────────────
init_session_state()
init_db()

# ── Auto-restore persisted state on first load ────────────────────────────────
if not st.session_state.get("_restored"):
    st.session_state._restored = True

    # Restore chat history
    if not st.session_state.chat_history:
        st.session_state.chat_history = load_chat_history()

    # Restore retriever + chunks + chain
    if st.session_state.conv_chain is None and retriever_exists():
        retriever, chunks = load_retriever()
        if retriever and chunks:
            llm        = build_llm()
            conv_chain, memory = build_conv_chain(llm, retriever)
            st.session_state.retriever    = retriever
            st.session_state.all_chunks   = chunks
            st.session_state.total_chunks = len(chunks)
            st.session_state.conv_chain   = conv_chain
            st.session_state.memory       = memory
            st.session_state.llm          = llm

    # Restore doc metadata
    if not st.session_state.uploaded_docs:
        docs = load_doc_meta()
        if docs:
            st.session_state.uploaded_docs = [d["name"] for d in docs]
            st.session_state.summary       = {d["name"]: d["pages"]   for d in docs}
            st.session_state.doc_summaries = {d["name"]: d["summary"] for d in docs}

# ── Render ────────────────────────────────────────────────────────────────────
render_sidebar()
render_chat()
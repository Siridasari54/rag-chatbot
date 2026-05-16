import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

DEFAULTS = {
    # RAG pipeline
    "conv_chain":    None,
    "retriever":     None,
    "memory":        None,
    "llm":           None,
    "all_chunks":    [],
    "uploaded_docs": [],
    "doc_summaries": {},
    "summary":       {},
    "total_chunks":  0,
    "sw_window":     1,

    # Chat
    "chat_history":  [],   # list of dicts with 'timestamp' key
    "feedback":      {},

    # Cache
    "query_cache":   {},
    "cache_hits":    0,
    "cache_misses":  0,
    "cache_enabled": True,
    "cache_ttl":     3600,

    # Chunking / retrieval
    "chunk_size":    300,
    "chunk_overlap": 100,
    "top_k":         5,

    # Feature flags
    "query_rewrite":    True,
    "ragas_enabled":    True,   # ★ NEW: toggle RAGAS evaluation

    # Misc
    "suggested_query":  None,
    "total_tokens_used":   0,
    "tokens_this_session": [],
}


def init_session_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
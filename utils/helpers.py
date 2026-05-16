import streamlit as st

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
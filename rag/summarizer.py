def summarize_document(doc_name: str, chunks: list, llm) -> str:
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
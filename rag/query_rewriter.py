# rag/query_rewriter.py — FULL REPLACEMENT

EXPANSIONS = {
    "dl":              "deep learning",
    "ml":              "machine learning",
    "nlp":             "natural language processing",
    "cv":              "computer vision",
    "nn":              "neural network",
    "rl":              "reinforcement learning",
    "llm":             "large language model",
    "mlp":             "multilayer perceptron",
    "bp":              "backpropagation",
    "backpropogation": "backpropagation",
    "cnn":             "convolutional neural network",
    "rnn":             "recurrent neural network",
    "svm":             "support vector machine",
    "knn":             "k nearest neighbors",
    "pca":             "principal component analysis",
}

def rewrite_query(original_query: str, llm) -> str:
    lower = original_query.strip().lower()

    # ★ Step 1 — exact match ("what is ml" won't hit this, only bare "ml")
    if lower in EXPANSIONS:
        return EXPANSIONS[lower]

    # ★ Step 2 — word-level expansion ("what is ml" → "what is machine learning")
    words = lower.split()
    expanded_words = [EXPANSIONS.get(w, w) for w in words]
    expanded = " ".join(expanded_words)
    if expanded != lower:
        return expanded          # ← this is what was missing before

    # ★ Step 3 — skip LLM rewrite if already short and specific
    if len(words) <= 4:
        return original_query

    # ★ Step 4 — LLM rewrite for longer vague queries
    prompt = f"""Rewrite this question to improve document retrieval. Rules:
- Keep it SHORT (under 10 words)
- Use specific technical terms likely to appear in ML documents
- Remove filler words like "fundamental", "concepts of", "explain"
- If already short and specific, return it UNCHANGED

Original: {original_query}
Rewritten (short, keyword-focused):"""
    try:
        result    = llm.invoke(prompt)
        rewritten = result.content.strip()
        if not rewritten or len(rewritten) > 300:
            return original_query
        return rewritten
    except Exception:
        return original_query
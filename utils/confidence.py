# utils/confidence.py — FULL REPLACEMENT

def compute_relevance_score(response: str, source_docs: list) -> float:
    """
    Renamed from compute_confidence — this is a relevance heuristic,
    not true calibrated confidence.
    """
    not_found_phrases = [
        "couldn't find",
        "not in the",
        "i don't know",
        "no information",
        "i'm not sure",
        "unclear",
        "cannot determine",
        "isn't covered in the uploaded",
        "not covered in",
        "not available",
    ]
    if any(p in response.lower() for p in not_found_phrases):
        return 0.10

    score = 0.5

    unique_srcs = len({d.metadata.get("source_file", "") for d in source_docs})
    score += min(unique_srcs * 0.10, 0.25)

    word_count = len(response.split())
    if word_count < 20:
        score *= 0.6
    elif word_count > 80:
        score += 0.05

    if source_docs:
        answer_words   = set(response.lower().split())
        overlap_scores = []
        for doc in source_docs:
            doc_words = set(doc.page_content.lower().split())
            overlap   = len(answer_words & doc_words) / max(len(answer_words), 1)
            overlap_scores.append(overlap)
        avg_overlap = sum(overlap_scores) / len(overlap_scores)
        score += min(avg_overlap * 0.3, 0.15)

    return round(min(score, 0.95), 2)


def relevance_bar_html(score: float) -> str:
    """Renamed from confidence_bar_html — honest labeling."""
    percentage = int(score * 100)

    if score >= 0.75:
        color = "#2ecc71"
    elif score >= 0.45:
        color = "#f1c40f"
    else:
        color = "#e74c3c"

    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 12px 0; width: 100%; max-width: 500px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 14px; font-weight: 600; color: #333333;">
            <span>Relevance Score</span>
            <span style="color: {color};">{percentage}%</span>
        </div>
        <div style="background-color: #e0e0e0; border-radius: 10px; width: 100%; height: 10px; overflow: hidden;">
            <div style="background-color: {color}; width: {percentage}%; height: 100%; border-radius: 10px; transition: width 0.4s ease-out;"></div>
        </div>
        <div style="font-size: 11px; color: #94a3b8; margin-top: 3px;">
            Heuristic score based on source overlap — not calibrated confidence
        </div>
    </div>
    """


# Keep old names as aliases so nothing else breaks
compute_confidence   = compute_relevance_score
confidence_bar_html  = relevance_bar_html
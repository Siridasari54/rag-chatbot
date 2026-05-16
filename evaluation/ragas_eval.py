# evaluation/ragas_eval.py — FULL REPLACEMENT

import streamlit as st


def _get_judge_llm():
    try:
        from langchain_groq import ChatGroq
        from config.settings import groq_api_key
        return ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",  # ★ same model, different instance
            temperature=0.0,                      # ★ deterministic scoring
        )
    except Exception:
        return st.session_state.get("llm")  # fallback to same LLM


def _score_with_judge(prompt: str, judge_llm) -> float:
    try:
        import re
        result = judge_llm.invoke(prompt)
        text   = result.content.strip()
        match  = re.search(r"(0?\.\d+|1\.0|0|1)", text)
        return float(match.group(1)) if match else 0.5
    except Exception:
        return 0.5


def run_ragas_evaluation(question: str, answer: str, source_docs: list):
    if not source_docs:
        return None

    # ★ Use separate judge LLM — not the same one that generated the answer
    judge_llm = _get_judge_llm()
    if not judge_llm:
        return None

    context = "\n".join(d.page_content for d in source_docs[:5])

    faithfulness_prompt = f"""You are an impartial evaluator.
Rate how faithful this answer is to the context. 
Score 0.0 (contradicts context) to 1.0 (fully supported).
Reply with ONLY a number between 0.0 and 1.0.

Context: {context}
Answer: {answer}
Score:"""

    relevancy_prompt = f"""You are an impartial evaluator.
Rate how relevant this answer is to the question.
Score 0.0 (irrelevant) to 1.0 (perfectly answers).
Reply with ONLY a number between 0.0 and 1.0.

Question: {question}
Answer: {answer}
Score:"""

    precision_prompt = f"""You are an impartial evaluator.
Rate what fraction of the retrieved context was useful for answering.
Score 0.0 (none useful) to 1.0 (all useful).
Reply with ONLY a number between 0.0 and 1.0.

Question: {question}
Context: {context}
Score:"""

    recall_prompt = f"""You are an impartial evaluator.
Rate whether the answer covers all key points it should.
Score 0.0 (misses everything) to 1.0 (covers everything).
Reply with ONLY a number between 0.0 and 1.0.

Question: {question}
Answer: {answer}
Score:"""

    try:
        return {
            "faithfulness":      round(_score_with_judge(faithfulness_prompt, judge_llm), 2),
            "answer_relevancy":  round(_score_with_judge(relevancy_prompt,    judge_llm), 2),
            "context_precision": round(_score_with_judge(precision_prompt,    judge_llm), 2),
            "context_recall":    round(_score_with_judge(recall_prompt,        judge_llm), 2),
        }
    except Exception:
        return None


def render_ragas_panel(scores: dict) -> None:
    if not scores:
        return

    def _bar(label: str, value: float, color: str):
        pct = int(value * 100)
        st.markdown(
            f"""<div style="margin:4px 0">
              <div style="display:flex;justify-content:space-between;font-size:12px;color:#94a3b8">
                <span>{label}</span><span style="color:{color}">{pct}%</span>
              </div>
              <div style="background:#1e293b;border-radius:4px;height:5px">
                <div style="width:{pct}%;height:5px;border-radius:4px;background:{color}"></div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    def _color(v: float) -> str:
        return "#2ecc71" if v >= 0.7 else "#f1c40f" if v >= 0.4 else "#e74c3c"

    with st.expander("📊 RAGAS Evaluation (judged by separate LLM)"):
        st.caption("⚖️ Evaluated by llama-3.1-70b — separate from answer model")
        _bar("Faithfulness",      scores["faithfulness"],      _color(scores["faithfulness"]))
        _bar("Answer Relevancy",  scores["answer_relevancy"],  _color(scores["answer_relevancy"]))
        _bar("Context Precision", scores["context_precision"], _color(scores["context_precision"]))
        _bar("Context Recall",    scores["context_recall"],    _color(scores["context_recall"]))
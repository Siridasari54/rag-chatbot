# ui/chat_ui.py — FULL REPLACEMENT

import time
import hashlib
import streamlit as st
from rag.query_rewriter import rewrite_query
from rag.chains import safe_invoke_chain
from utils.cache import get_cached, set_cache
from utils.confidence import compute_confidence, confidence_bar_html
from utils.helpers import render_sources
from utils.text_cleaner import clean_response
from evaluation.ragas_eval import run_ragas_evaluation, render_ragas_panel
from utils.persistence import save_chat

def _stable_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _suggest_followups(answer: str, llm) -> list:
    prompt = f"""Based on this ML answer, generate 3 short follow-up questions.
Return ONLY a JSON list like: ["q1", "q2", "q3"]

Answer: {answer}"""
    try:
        import json
        result   = llm.invoke(prompt)
        raw      = result.content.strip()
        parsed   = json.loads(raw)
        if not isinstance(parsed, list):
            return []
        # ★ validate each item is a non-empty string
        return [q for q in parsed if isinstance(q, str) and q.strip()][:3]
    except json.JSONDecodeError:
        # ★ log it instead of silently swallowing
        print(f"[WARN] _suggest_followups: invalid JSON returned: {result.content[:100]}")
        return []
    except Exception as e:
        print(f"[WARN] _suggest_followups failed: {e}")
        return []


def _stream_response(text: str):
    for word in text.split():
        yield word + " "
        time.sleep(0.02)


def _render_history():
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
                    '<div class="halluc-warning">⚠️ Low confidence — verify against source documents.</div>',
                    unsafe_allow_html=True,
                )

            st.write(chat["answer"])

            # ★ show stored token usage from history
            token_usage = chat.get("token_usage", {})
            if token_usage:
                st.caption(
                    f"🔢 Prompt: {token_usage.get('prompt_tokens', 0)} | "
                    f"Completion: {token_usage.get('completion_tokens', 0)} | "
                    f"Total: {token_usage.get('total_tokens', 0)}"
                )

            if chat.get("sources"):
                with st.expander("📎 Source Citations"):
                    render_sources(chat["sources"])

            if chat.get("ragas_scores"):
                render_ragas_panel(chat["ragas_scores"])

            current_fb = st.session_state.feedback.get(idx)
            c1, c2, c3 = st.columns([1, 1, 8])
            with c1:
                if st.button("👍", key=f"up_{idx}", help="Good answer"):
                    st.session_state.feedback[idx] = "👍 Helpful"
                    st.rerun()
            with c2:
                if st.button("👎", key=f"dn_{idx}", help="Bad answer"):
                    st.session_state.feedback[idx] = "👎 Not helpful"
                    st.rerun()
            with c3:
                if current_fb:
                    st.caption(current_fb)


def render_chat():
    if st.session_state.conv_chain is None:
        st.info("👈 Upload your documents from the sidebar and click **Process Documents** to begin!")
        return

    if len(st.session_state.uploaded_docs) >= 2:
        compare_mode = st.toggle("⚖️ Compare across documents")
        if compare_mode:
            col1, col2 = st.columns(2)
            with col1:
                doc1 = st.selectbox("Doc 1", st.session_state.uploaded_docs, key="d1")
            with col2:
                doc2 = st.selectbox("Doc 2", st.session_state.uploaded_docs, key="d2")
            query = st.chat_input("Ask to compare both documents…")
            if query:
                prompt = f"Compare what '{doc1}' and '{doc2}' say about: {query}"
                with st.spinner("⚖️ Comparing…"):
                    result, error = safe_invoke_chain(st.session_state.conv_chain, prompt)
                if error:
                    st.error(error)
                else:
                    st.write(result.get("answer", ""))
            return

    _render_history()

    if st.session_state.get("suggested_query"):
        query = st.session_state.suggested_query
        st.session_state.suggested_query = None
    else:
        query = st.chat_input("Ask anything about Machine Learning…")

    if not query:
        return

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        from_cache      = False
        rewritten_query = query
        token_usage     = {}
        cached_hit      = get_cached(query) if st.session_state.cache_enabled else None

        if cached_hit:
            st.session_state.cache_hits += 1
            from_cache  = True
            response    = cached_hit["answer"]
            source_docs = cached_hit.get("sources", [])
            st.markdown('<span class="cache-badge">⚡ cached</span>', unsafe_allow_html=True)
        else:
            st.session_state.cache_misses += 1

            if st.session_state.get("query_rewrite", True) and st.session_state.llm:
                with st.spinner("✏️ Rewriting query…"):
                    rewritten_query = rewrite_query(query, st.session_state.llm)
                if rewritten_query != query:
                    st.markdown(
                        f'<span class="rewrite-badge">🔄 Rewritten: {rewritten_query}</span>',
                        unsafe_allow_html=True,
                    )

            with st.spinner("🔍 Searching across all documents…"):
                result, error = safe_invoke_chain(st.session_state.conv_chain, rewritten_query)

            if error:
                st.markdown(f'<div class="error-box">{error}</div>', unsafe_allow_html=True)
                st.stop()

            response    = clean_response(result.get("answer", ""))
            source_docs = result.get("source_documents", [])

            # ★ extract token usage from chain result
            token_usage = result.get("token_usage", {})

            if st.session_state.cache_enabled:
                set_cache(query, response, source_docs)

        conf = compute_confidence(response, source_docs)
        st.markdown(confidence_bar_html(conf), unsafe_allow_html=True)
        if conf < 0.35:
            st.markdown(
                '<div class="halluc-warning">⚠️ Low confidence — verify against source documents.</div>',
                unsafe_allow_html=True,
            )

        st.write_stream(_stream_response(response))

        # ★ Token usage display
        if token_usage:
            total_q = token_usage.get("total_tokens", 0)
            st.session_state.total_tokens_used += total_q
            st.session_state.tokens_this_session.append(total_q)
            st.caption(
                f"🔢 Prompt: {token_usage.get('prompt_tokens', 0)} | "
                f"Completion: {token_usage.get('completion_tokens', 0)} | "
                f"This query: {total_q} | "
                f"Session total: {st.session_state.total_tokens_used}"
            )
        elif from_cache:
            st.caption("⚡ Served from cache — no tokens used")

        with st.expander("📎 Source Citations"):
            render_sources(source_docs)

        ragas_scores = None
        if st.session_state.get("ragas_enabled", True) and not from_cache:
            ragas_scores = run_ragas_evaluation(
                question=rewritten_query,
                answer=response,
                source_docs=source_docs,
            )
            if ragas_scores:
                render_ragas_panel(ragas_scores)
            else:
                st.info("📊 RAGAS evaluation disabled — enable in Advanced Settings once API quota is resolved.")

        if st.session_state.llm:
            followups = _suggest_followups(response, st.session_state.llm)
            if followups:
                st.markdown("**💡 You might also ask:**")
                for fq in followups:
                    btn_key = f"fq_{_stable_key(query + fq)}"
                    if st.button(fq, key=btn_key):
                        st.session_state.suggested_query = fq
                        st.rerun()

    # ★ save token_usage to history so _render_history() can display it
    new_chat = {
        "question":        query,
        "rewritten_query": rewritten_query,
        "answer":          response,
        "sources":         source_docs,
        "from_cache":      from_cache,
        "confidence":      conf,
        "ragas_scores":    ragas_scores,
        "token_usage":     token_usage,
        "timestamp":       time.time(),
    }
    st.session_state.chat_history.append(new_chat)
    save_chat(new_chat)   # ★ persist to SQLite
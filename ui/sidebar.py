"""
★ Upgraded Sidebar — Claude-style chat history
================================================
Features:
  • ✏️ New Chat button (top-right, always visible)
  • Chat history: Starred + Recents (last 10 only)
  • Star/unstar questions
  • Clear Chat + Download (.txt / PDF) at the bottom
  • RAGAS toggle in Advanced Settings
  • Loaded Documents with one-line AI summaries
  • Token usage panel
  • Ground Truth Evaluation
"""

import os
import time
import datetime
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

from loaders.file_loader import load_file
from rag.retriever import build_hybrid_retriever
from rag.chains import build_conv_chain
from rag.summarizer import summarize_document
from services.groq_services import build_llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.persistence import (
    save_retriever, save_doc_meta,
    clear_chat_history, clear_doc_meta
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize_for_pdf(text: str) -> str:
    replacements = {
        "\u2022": "-",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u2026": "...",
        "\u2192": "->",
        "\u2713": "ok",
        "\u00b7": "-",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _summarize_one(uf_name, chunks, llm):
    return uf_name, summarize_document(uf_name, chunks, llm)


# ── Main render ───────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:

        # ── Header row ───────────────────────────────────────────────────────
        col_title, col_new = st.columns([3, 1])
        with col_title:
            st.markdown("## 🔭 DeepLens")
        with col_new:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✏️", help="New Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.feedback     = {}
                st.session_state.query_cache  = {}
                st.session_state.cache_hits   = 0
                st.session_state.cache_misses = 0
                if st.session_state.get("memory"):
                    st.session_state.memory.clear()
                st.rerun()

        st.markdown("---")

        # ── Upload section ───────────────────────────────────────────────────
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

        # ── Advanced Settings ─────────────────────────────────────────────────
        with st.expander("⚙️ Advanced Settings", expanded=False):
            st.session_state.chunk_size    = st.slider(
                "Chunk Size", 500, 2000, st.session_state.chunk_size, 100
            )
            st.session_state.chunk_overlap = st.slider(
                "Chunk Overlap", 50, 400, st.session_state.chunk_overlap, 50
            )
            st.session_state.top_k         = st.slider(
                "Top-K Retrieval", 3, 10, st.session_state.top_k, 1
            )
            cache_on         = st.toggle("Enable Query Cache",     value=st.session_state.cache_enabled)
            query_rewrite_on = st.toggle("Enable Query Rewriting", value=st.session_state.query_rewrite)
            ragas_on         = st.toggle(
                "Enable RAGAS Evaluation",
                value=st.session_state.ragas_enabled,
                help="Runs LLM-as-judge evaluation per answer. Adds ~2–5s latency.",
            )

        # ── Process button ────────────────────────────────────────────────────
        process_btn = st.button("🚀 Process Documents", type="primary")

        if process_btn and uploaded_files:
            os.makedirs("data", exist_ok=True)
            all_docs, skipped_files = [], []
            progress_bar = st.progress(0, text="Starting…")
            total_files  = len(uploaded_files)

            for file_idx, uf in enumerate(uploaded_files):
                progress_bar.progress(
                    int((file_idx / total_files) * 60),
                    text=f"📄 Loading {uf.name}…",
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
                st.warning(f"⚠️ Skipped: {', '.join(skipped_files)}")
            if not all_docs:
                st.error("❌ No content extracted.")
                progress_bar.empty()
                st.stop()

            progress_bar.progress(65, text="✂️ Chunking documents…")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
            )
            chunks = splitter.split_documents(all_docs)

            progress_bar.progress(75, text="🔢 Building FAISS + BM25…")
            hybrid_retriever = build_hybrid_retriever(chunks, st.session_state.top_k)

            progress_bar.progress(85, text="🤖 Loading Groq LLM…")
            llm = build_llm()

            conv_chain, memory = build_conv_chain(llm, hybrid_retriever)

            progress_bar.progress(92, text="📝 Summarising documents…")
            doc_summaries = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(_summarize_one, uf.name, chunks, llm): uf.name
                    for uf in uploaded_files
                }
                for future in as_completed(futures):
                    name, summary = future.result()
                    doc_summaries[name] = summary

            progress_bar.progress(100, text="✅ Done!")
            time.sleep(0.5)
            progress_bar.empty()

            # ★ Persist retriever and doc metadata to disk
            save_retriever(hybrid_retriever, chunks)
            for uf in uploaded_files:
                save_doc_meta(
                    uf.name,
                    st.session_state.summary.get(uf.name, 0),
                    doc_summaries.get(uf.name, "")
                )

            st.session_state.update({
                "all_chunks":          chunks,
                "sw_window":           1,
                "total_chunks":        len(chunks),
                "retriever":           hybrid_retriever,
                "memory":              memory,
                "conv_chain":          conv_chain,
                "llm":                 llm,
                "uploaded_docs":       [uf.name for uf in uploaded_files],
                "doc_summaries":       doc_summaries,
                "chat_history":        [],
                "query_cache":         {},
                "cache_hits":          0,
                "cache_misses":        0,
                "cache_enabled":       cache_on,
                "cache_ttl":           3600,
                "feedback":            {},
                "query_rewrite":       query_rewrite_on,
                "ragas_enabled":       ragas_on,
                "total_tokens_used":   0,
                "tokens_this_session": [],
            })
            st.success("✅ Knowledge base ready! Start chatting →")

        elif process_btn and not uploaded_files:
            st.warning("⚠️ Please upload at least one file first.")

        # ── Loaded Documents ──────────────────────────────────────────────────
        if st.session_state.uploaded_docs:
            st.markdown("---")
            st.markdown("### 📚 Loaded Documents")
            for doc in st.session_state.uploaded_docs:
                pages   = st.session_state.summary.get(doc, "?")
                summary = st.session_state.doc_summaries.get(doc, "")
                st.markdown(f"📄 `{doc}` — {pages} pages")
                if summary:
                    st.markdown(
                        f'<div class="summary-box">💡 {summary}</div>',
                        unsafe_allow_html=True,
                    )

            if st.button("🗑️ Clear Cache", type="secondary"):
                st.session_state.query_cache  = {}
                st.session_state.cache_hits   = 0
                st.session_state.cache_misses = 0
                st.rerun()

        # ── Token Usage ───────────────────────────────────────────────────────
        if st.session_state.get("total_tokens_used", 0) > 0:
            st.markdown("---")
            st.markdown("### 🔢 Token Usage")
            total = st.session_state.total_tokens_used
            calls = len(st.session_state.get("tokens_this_session", []))
            avg   = int(total / calls) if calls > 0 else 0
            st.markdown(f"**Session total:** {total:,} tokens")
            st.markdown(f"**Queries made:** {calls}")
            st.markdown(f"**Avg per query:** {avg} tokens")

        # ── Ground Truth Evaluation ───────────────────────────────────────────
        if st.session_state.conv_chain:
            st.markdown("---")
            st.markdown("### 🧪 Ground Truth Evaluation")
            st.caption("Tests 10 preset questions against your documents")
            if st.button("▶️ Run Eval Dataset", type="secondary"):
                from evaluation.eval_dataset import run_ground_truth_eval
                with st.spinner("Running 10 eval questions…"):
                    results = run_ground_truth_eval(
                        st.session_state.conv_chain,
                        st.session_state.llm,
                    )
                total     = len(results)
                retrieved = sum(1 for r in results if r["retrieved"])
                avg_score = sum(r["overlap_score"] for r in results) / total

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Retrieved", f"{retrieved}/{total}")
                with col2:
                    st.metric("Avg Score", f"{avg_score:.0%}")

                with st.expander("📋 Full Eval Results"):
                    for r in results:
                        icon = "✅" if r["retrieved"] else "❌"
                        st.markdown(f"{icon} **{r['question']}**")
                        st.caption(
                            f"Score: {r['overlap_score']:.0%} | "
                            f"Answer: {r['answer'][:100]}…"
                        )

        # ── ★ Chat History — Claude style: Starred + Recents ─────────────────
        if st.session_state.chat_history:
            st.markdown("---")

            # Starred section
            starred = [c for c in st.session_state.chat_history if c.get("starred")]
            if starred:
                st.markdown(
                    "<p style='font-size:11px;color:#64748b;font-weight:700;"
                    "text-transform:uppercase;letter-spacing:0.10em;"
                    "margin:8px 0 4px 4px'>⭐ Starred</p>",
                    unsafe_allow_html=True,
                )
                for i, chat in enumerate(starred):
                    words = chat["question"].split()
                    title = (" ".join(words[:5]) + "…") if len(words) > 5 else chat["question"]
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        if st.button(title, key=f"starred_{i}", use_container_width=True):
                            st.session_state.suggested_query = chat["question"]
                            st.rerun()
                    with col2:
                        if st.button("★", key=f"unstar_{i}", help="Unstar"):
                            idx = next(
                                j for j, c in enumerate(st.session_state.chat_history)
                                if c["question"] == chat["question"]
                            )
                            st.session_state.chat_history[idx]["starred"] = False
                            st.rerun()

            # Recents section — last 10 only, newest first
            st.markdown(
                "<p style='font-size:11px;color:#64748b;font-weight:700;"
                "text-transform:uppercase;letter-spacing:0.10em;"
                "margin:14px 0 4px 4px'>Recents</p>",
                unsafe_allow_html=True,
            )
            recent_chats = list(reversed(st.session_state.chat_history[-10:]))
            for i, chat in enumerate(recent_chats):
                words = chat["question"].split()
                title = (" ".join(words[:5]) + "…") if len(words) > 5 else chat["question"]
                col1, col2 = st.columns([5, 1])
                with col1:
                    if st.button(title, key=f"recent_{i}", use_container_width=True):
                        st.session_state.suggested_query = chat["question"]
                        st.rerun()
                with col2:
                    if st.button("☆", key=f"star_btn_{i}", help="Star this"):
                        idx = next(
                            j for j, c in enumerate(st.session_state.chat_history)
                            if c["question"] == chat["question"]
                        )
                        st.session_state.chat_history[idx]["starred"] = True
                        st.rerun()

            st.markdown("---")

            if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.feedback     = {}
                clear_chat_history()
                if st.session_state.get("memory"):
                    st.session_state.memory.clear()
                st.rerun()

            # ── Download options ──────────────────────────────────────────────
            st.markdown("### 📥 Download Chat")

            chat_text = ""
            for i, chat in enumerate(st.session_state.chat_history):
                flag   = " [CACHED]" if chat.get("from_cache") else ""
                conf   = chat.get("confidence", 0)
                fb     = st.session_state.feedback.get(i, "")
                fb_lbl = f" [{fb}]" if fb else ""
                rags   = chat.get("ragas_scores")
                tokens = chat.get("token_usage", {})

                chat_text += f"Q{i+1}: {chat['question']}\n"
                if chat.get("rewritten_query") and chat["rewritten_query"] != chat["question"]:
                    chat_text += f"   (Rewritten: {chat['rewritten_query']})\n"
                chat_text += f"A{i+1}{flag}{fb_lbl} (relevance: {int(conf*100)}%):\n{chat['answer']}\n"
                if tokens:
                    chat_text += (
                        f"   Tokens — Prompt:{tokens.get('prompt_tokens',0)} "
                        f"Completion:{tokens.get('completion_tokens',0)} "
                        f"Total:{tokens.get('total_tokens',0)}\n"
                    )
                if rags:
                    chat_text += (
                        f"   RAGAS — Faithfulness:{rags['faithfulness']} "
                        f"Relevancy:{rags['answer_relevancy']} "
                        f"Precision:{rags['context_precision']} "
                        f"Recall:{rags['context_recall']}\n"
                    )
                chat_text += "-" * 60 + "\n"

            st.download_button(
                label="⬇️ Download .txt",
                data=chat_text,
                file_name="deeplens_chat.txt",
                mime="text/plain",
            )

            # ── PDF export ────────────────────────────────────────────────────
            try:
                from fpdf import FPDF

                def _export_pdf():
                    pdf = FPDF()
                    pdf.add_page()

                    for i, chat in enumerate(st.session_state.chat_history):
                        pdf.set_font("Arial", "B", 12)
                        pdf.multi_cell(
                            0, 10,
                            _sanitize_for_pdf(f"Q{i+1}: {chat['question']}")
                        )
                        if chat.get("rewritten_query") and chat["rewritten_query"] != chat["question"]:
                            pdf.set_font("Arial", "I", 9)
                            pdf.multi_cell(
                                0, 6,
                                _sanitize_for_pdf(f"  Rewritten: {chat['rewritten_query']}")
                            )
                        pdf.set_font("Arial", size=10)
                        conf   = chat.get("confidence", 0)
                        flag   = " [CACHED]" if chat.get("from_cache") else ""
                        fb     = st.session_state.feedback.get(i, "")
                        fb_lbl = f" [{fb}]" if fb else ""
                        pdf.multi_cell(
                            0, 8,
                            _sanitize_for_pdf(
                                f"A{i+1}{flag}{fb_lbl} (relevance: {int(conf*100)}%):\n{chat['answer']}"
                            )
                        )
                        tokens = chat.get("token_usage", {})
                        if tokens:
                            pdf.set_font("Arial", "I", 8)
                            pdf.multi_cell(
                                0, 6,
                                _sanitize_for_pdf(
                                    f"Tokens — Prompt:{tokens.get('prompt_tokens',0)} "
                                    f"Completion:{tokens.get('completion_tokens',0)} "
                                    f"Total:{tokens.get('total_tokens',0)}"
                                )
                            )
                        rags = chat.get("ragas_scores")
                        if rags:
                            pdf.set_font("Arial", "I", 8)
                            pdf.multi_cell(
                                0, 6,
                                _sanitize_for_pdf(
                                    f"RAGAS - Faithfulness:{rags['faithfulness']} "
                                    f"Relevancy:{rags['answer_relevancy']} "
                                    f"Precision:{rags['context_precision']} "
                                    f"Recall:{rags['context_recall']}"
                                )
                            )
                        pdf.ln(4)
                        pdf.set_draw_color(200, 200, 200)
                        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                        pdf.ln(2)

                    return pdf.output(dest="S").encode("latin-1")

                st.download_button(
                    label="⬇️ Download PDF",
                    data=_export_pdf(),
                    file_name="deeplens_chat.pdf",
                    mime="application/pdf",
                )

            except ImportError:
                st.caption("Install `fpdf` for PDF export: `pip install fpdf`")
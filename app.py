import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="DeepLens",
    page_icon="🔭",
    layout="centered"
)

st.markdown("""
    <style>
        .stApp { background-color: #0e1117; color: white; }
        #MainMenu, footer, header { visibility: hidden; }
        section[data-testid="stSidebar"] * { font-size: 0.78rem !important; }
        section[data-testid="stSidebar"] h2 { font-size: 0.95rem !important; }
        section[data-testid="stSidebar"] h3 { font-size: 0.85rem !important; }
        .main-title {
            font-size: 2rem;
            font-weight: 800;
            color: white;
            text-align: center;
            margin-top: 40px;
            margin-bottom: 5px;
        }
        .sub-title {
            font-size: 1rem;
            color: #888;
            text-align: center;
            margin-bottom: 20px;
        }
        .source-box {
            background-color: #1e2130;
            border-left: 3px solid #4a90e2;
            padding: 10px 15px;
            border-radius: 8px;
            margin-top: 5px;
            font-size: 0.85rem;
            color: #aaa;
        }
        .doc-badge {
            background-color: #4a90e2;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.75rem;
            margin-right: 5px;
        }
        .stChatMessage {
            background-color: #1e2130 !important;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .topic-badge {
            display: inline-block;
            background-color: #1e2130;
            border: 1px solid #4a90e2;
            color: #4a90e2;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.8rem;
            margin: 3px;
        }
        .stat-box {
            background-color: #1e2130;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            margin: 5px 0;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔭 DeepLens</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Search Deep into ML Knowledge</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-bottom:20px;">
    <span class="topic-badge">📘 ML Basics</span>
    <span class="topic-badge">🧠 Deep Learning</span>
    <span class="topic-badge">💬 NLP</span>
    <span class="topic-badge">🔗 Neural Networks</span>
    <span class="topic-badge">👁️ Computer Vision</span>
</div>
""", unsafe_allow_html=True)

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found!")
    st.stop()

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "summary" not in st.session_state:
    st.session_state.summary = {}
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

with st.sidebar:
    st.markdown("## 📂 Upload ML Documents")
    st.markdown("Upload any ML related PDFs")
    st.markdown("""
    **Suggested files:**
    - ML_Basics.pdf
    - Deep_Learning.pdf
    - NLP.pdf
    - Neural_Networks.pdf
    - Computer_Vision.pdf
    """)

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        st.markdown("### 📄 Selected Files:")
        for f in uploaded_files:
            st.markdown(f"✅ `{f.name}`")

    process_btn = st.button("🚀 Process Documents", type="primary")

    if process_btn and uploaded_files:
        with st.spinner("Processing all ML documents..."):

            os.makedirs("data", exist_ok=True)
            all_docs = []

            for uploaded_file in uploaded_files:
                safe_name = uploaded_file.name.replace(" ", "_")
                file_path = f"data/{safe_name}"

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                loader = PyPDFLoader(file_path)
                pages = loader.load()

                for page in pages:
                    page.metadata["source_file"] = uploaded_file.name
                    page.metadata["page"] = page.metadata.get("page", 0) + 1

                all_docs.extend(pages)
                st.session_state.summary[uploaded_file.name] = len(pages)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(all_docs)
            st.session_state.total_chunks = len(chunks)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            db = FAISS.from_documents(chunks, embeddings)
            retriever = db.as_retriever(search_kwargs={"k": 5})

            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama-3.1-8b-instant",
                temperature=0.2
            )

            prompt = PromptTemplate.from_template("""
You are an expert Machine Learning assistant.
Answer the question based only on the context provided from ML documents.
Always mention which document and page number the information comes from.
If the answer spans multiple documents mention all of them.
If the answer is not in the context, say "I couldn't find this in the uploaded ML documents."

Context:
{context}

Question: {question}

Answer (mention source document name and page number):
""")

            def format_docs(docs):
                formatted = ""
                for doc in docs:
                    source = doc.metadata.get("source_file", "Unknown")
                    page = doc.metadata.get("page", "N/A")
                    formatted += f"[Source: {source} | Page: {page}]\n{doc.page_content}\n\n"
                return formatted

            st.session_state.retriever = retriever
            st.session_state.uploaded_docs = [f.name for f in uploaded_files]
            st.session_state.chat_history = []
            st.session_state.qa_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

        st.success("✅ All ML documents processed!")

    if st.session_state.uploaded_docs:
        st.markdown("---")
        st.markdown("### 📚 Knowledge Base:")
        for doc in st.session_state.uploaded_docs:
            pages_count = st.session_state.summary.get(doc, "?")
            st.markdown(f"📄 `{doc}` — {pages_count} pages")

    if st.session_state.uploaded_docs:
        st.markdown("---")
        st.markdown("### 📊 Document Statistics:")

        total_pages = sum(st.session_state.summary.values())
        total_docs = len(st.session_state.uploaded_docs)
        total_chunks = st.session_state.total_chunks
        total_questions = len(st.session_state.chat_history)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Total Docs", total_docs)
            st.metric("📃 Total Pages", total_pages)
        with col2:
            st.metric("🧩 Chunks", total_chunks)
            st.metric("💬 Questions", total_questions)

    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📥 Download Chat History:")

        chat_text = ""
        for i, chat in enumerate(st.session_state.chat_history):
            chat_text += f"Q{i+1}: {chat['question']}\n"
            chat_text += f"A{i+1}: {chat['answer']}\n"
            chat_text += "-" * 50 + "\n"

        st.download_button(
            label="⬇️ Download as .txt",
            data=chat_text,
            file_name="ml_chat_history.txt",
            mime="text/plain"
        )

if st.session_state.qa_chain is None:
    st.info("👈 Upload your ML PDF documents from the sidebar and click **Process Documents** to start chatting!")

else:
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            if chat.get("sources"):
                with st.expander("📎 Source Citations"):
                    seen = set()
                    for src in chat["sources"]:
                        source_file = src.metadata.get("source_file", "Unknown")
                        page = src.metadata.get("page", "N/A")
                        key = f"{source_file}-{page}"
                        if key not in seen:
                            seen.add(key)
                            st.markdown(f"""
                            <div class="source-box">
                                <span class="doc-badge">📄 {source_file}</span>
                                Page {page}<br>
                                <small>{src.page_content[:200]}...</small>
                            </div>
                            """, unsafe_allow_html=True)

    query = st.chat_input("Ask anything about Machine Learning...")

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching across all ML documents..."):
                response = st.session_state.qa_chain.invoke(query)
                source_docs = st.session_state.retriever.invoke(query)

            st.write(response)

            with st.expander("📎 Source Citations"):
                seen = set()
                for src in source_docs:
                    source_file = src.metadata.get("source_file", "Unknown")
                    page = src.metadata.get("page", "N/A")
                    key = f"{source_file}-{page}"
                    if key not in seen:
                        seen.add(key)
                        st.markdown(f"""
                        <div class="source-box">
                            <span class="doc-badge">📄 {source_file}</span>
                            Page {page}<br>
                            <small>{src.page_content[:200]}...</small>
                        </div>
                        """, unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "question": query,
            "answer": response,
            "sources": source_docs
        })
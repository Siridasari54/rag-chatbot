import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="ML Chatbot",
    page_icon="🤖",
    layout="centered"
)

# =====================
# CUSTOM CSS — Dark Theme like HR Assistant
# =====================
st.markdown("""
    <style>
        /* Dark background */
        .stApp {
            background-color: #0e1117;
            color: white;
        }

        /* Hide streamlit default header/footer */
        #MainMenu, footer, header {visibility: hidden;}

        /* Title styling */
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            color: white;
            text-align: center;
            margin-top: 60px;
            margin-bottom: 5px;
        }

        .sub-title {
            font-size: 1rem;
            color: #888;
            text-align: center;
            margin-bottom: 30px;
        }

        /* Chat messages */
        .stChatMessage {
            background-color: #1e2130 !important;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }

        /* Chat input box */
        .stChatInput {
            background-color: #1e2130 !important;
            border-radius: 12px;
        }

        /* Spinner */
        .stSpinner {
            color: #4a90e2 !important;
        }

        /* Info box */
        .stAlert {
            background-color: #1a2a3a !important;
            color: #4a90e2 !important;
            border: none !important;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# =====================
# TITLE
# =====================
st.markdown('<div class="main-title">🤖 Machine Learning Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask me anything about Machine Learning concepts, algorithms, and techniques.</div>', unsafe_allow_html=True)

# =====================
# VALIDATE API KEY
# =====================
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found! Please check your .env file.")
    st.stop()

# =====================
# SESSION STATE INIT
# =====================
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================
# AUTO LOAD PDF (only once)
# =====================
PDF_PATH = "data/machine learning.pdf"

if st.session_state.qa_chain is None:

    if not os.path.exists(PDF_PATH):
        st.error(f"❌ PDF not found at: {PDF_PATH}")
        st.stop()

    with st.spinner("Setting up knowledge base for first time..."):

        loader = PyPDFLoader(PDF_PATH)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = splitter.split_documents(pages)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory="vectorstore"
        )

        retriever = db.as_retriever(search_kwargs={"k": 3})

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.2
        )

        prompt = PromptTemplate.from_template("""
You are a helpful Machine Learning assistant. Use the following context from the ML PDF to answer the question clearly.
If the answer is not in the context, say "I don't know based on the provided ML document."

Context:
{context}

Question: {question}

Answer:
""")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        st.session_state.retriever = retriever
        st.session_state.qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

# =====================
# CHAT HISTORY DISPLAY
# =====================
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])

# =====================
# CHAT INPUT
# =====================
query = st.chat_input("Ask your ML question...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke(query)
        st.write(response)

    st.session_state.chat_history.append({
        "question": query,
        "answer": response
    })
import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
)

def load_file(path: str, original_name: str):
    ext = original_name.rsplit(".", 1)[-1].lower()
    try:
        if ext == "pdf":
            return PyPDFLoader(path).load()
        elif ext == "txt":
            return TextLoader(path, encoding="utf-8").load()
        elif ext == "docx":
            return Docx2txtLoader(path).load()
        elif ext == "pptx":
            return UnstructuredPowerPointLoader(path).load()
        elif ext == "csv":
            return CSVLoader(path).load()
        else:
            return []
    except Exception as e:
        st.warning(f"⚠️ Could not load `{original_name}`: {e}")
        return []
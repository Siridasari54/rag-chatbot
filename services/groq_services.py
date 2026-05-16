from langchain_groq import ChatGroq
from config.settings import groq_api_key

def build_llm():
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
    )
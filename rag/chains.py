# rag/chains.py — FULL REPLACEMENT

import time
import streamlit as st
from langchain_community.callbacks import get_openai_callback

try:
    from utils.token_counter import count_tokens_in_messages
except ImportError:
    def count_tokens_in_messages(prompt: str, answer: str) -> dict:
        prompt_tokens     = len(prompt.split())
        completion_tokens = len(answer.split())
        return {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
            "method":            "word_count_fallback",
        }

_MAX_HISTORY_CHARS = 1200
_MAX_CONTEXT_CHARS = 3000


class SimpleConvChain:
    def __init__(self, llm, retriever):
        self.llm       = llm
        self.retriever = retriever
        self.history   = []

    def _get_docs(self, query: str):
        try:
            return self.retriever.invoke(query)
        except AttributeError:
            return self.retriever.get_relevant_documents(query)

    def _build_history_str(self) -> str:
        lines = []
        total = 0
        for h in reversed(self.history[-8:]):
            chunk = f"Human: {h['question']}\nAssistant: {h['answer']}\n"
            if total + len(chunk) > _MAX_HISTORY_CHARS:
                break
            lines.append(chunk)
            total += len(chunk)
        return "".join(reversed(lines))

    def invoke(self, inputs: dict) -> dict:
        question    = inputs["question"]
        source_docs = self._get_docs(question)

        raw_context = "\n\n".join(doc.page_content for doc in source_docs)
        context     = raw_context[:_MAX_CONTEXT_CHARS]
        if len(raw_context) > _MAX_CONTEXT_CHARS:
            context += "\n[…context truncated…]"

        history_str = self._build_history_str()

        prompt = f"""You are a strict document Q&A assistant.

RULES — follow exactly, no exceptions:
1. Answer ONLY using the text in "Retrieved Context" below.
2. Copy key phrases directly from the context — do not paraphrase.
3. If the context does not contain the answer, respond with ONLY this exact sentence and nothing else:
   "This topic isn't covered in the uploaded documents."
   DO NOT add any extra information after this sentence.
4. Do NOT use outside knowledge. Do NOT guess. Do NOT say "however".
5. Use bullet points for multiple facts. Max 150 words.

Retrieved Context:
{context}

Previous Conversation (for reference only):
{history_str}

Question: {question}

Answer (using only the Retrieved Context above):"""

        try:
            with get_openai_callback() as cb:
                response = self.llm.invoke(prompt)
                answer   = response.content.strip()

            if cb.total_tokens > 0:
                token_usage = {
                    "prompt_tokens":     cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens":      cb.total_tokens,
                    "method":            "langchain_callback",
                }
            else:
                token_usage = count_tokens_in_messages(prompt, answer)

        except Exception as e:
            print(f"[WARN] chains.py callback failed: {e}")
            response    = self.llm.invoke(prompt)
            answer      = response.content.strip()
            token_usage = count_tokens_in_messages(prompt, answer)

        self.history.append({"question": question, "answer": answer})

        return {
            "answer":           answer,
            "source_documents": source_docs,
            "token_usage":      token_usage,
        }


class SimpleMemory:
    def __init__(self, chain):
        self.chain = chain

    def clear(self):
        self.chain.history = []


def build_conv_chain(llm, retriever):
    chain  = SimpleConvChain(llm, retriever)
    memory = SimpleMemory(chain)
    return chain, memory


def safe_invoke_chain(chain, query: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            result = chain.invoke({"question": query})
            return result, None
        except Exception as e:
            err = str(e).lower()
            if "rate limit" in err or "429" in err:
                wait = 2 ** attempt
                st.warning(f"⏳ Rate limited — retrying in {wait}s… ({attempt+1}/{max_retries})")
                time.sleep(wait)
            elif "authentication" in err or "401" in err:
                return None, "❌ Invalid GROQ_API_KEY. Please check your .env file."
            elif "connection" in err or "timeout" in err:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return None, "❌ Connection timeout. Check your internet connection."
            else:
                return None, f"❌ API Error: {e}"
    return None, "❌ Max retries reached. Groq API is unavailable."
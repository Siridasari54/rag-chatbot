from langchain_community.retrievers import BM25Retriever
from rag.embeddings import build_faiss_retriever


class SimpleEnsembleRetriever:
    def __init__(self, retrievers, weights):
        self.retrievers = retrievers
        self.weights    = weights

    def _get_docs(self, retriever, query: str):
        try:
            return retriever.invoke(query)
        except AttributeError:
            return retriever.get_relevant_documents(query)

    def get_relevant_documents(self, query: str):
        seen, results = set(), []
        for retriever in self.retrievers:
            docs = self._get_docs(retriever, query)
            for doc in docs:
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    results.append(doc)
        return results

    def invoke(self, query: str):
        return self.get_relevant_documents(query)


def build_hybrid_retriever(chunks, top_k: int):
    faiss_retriever = build_faiss_retriever(chunks, top_k)

    bm25_retriever   = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_k

    return SimpleEnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6],
    )
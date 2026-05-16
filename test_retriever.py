import sys
sys.path.append(".")

from loaders.file_loader import load_file
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.retriever import build_hybrid_retriever

pages = load_file("data/machine_learning.pdf", "machine_learning.pdf")
print(f"Pages loaded: {len(pages)}")
print(f"\nFirst page sample:\n{pages[0].page_content[:500]}")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(pages)
print(f"\nTotal chunks: {len(chunks)}")

retriever = build_hybrid_retriever(chunks, top_k=5)
results = retriever.invoke("what is machine learning")
print(f"\nRetrieved {len(results)} docs")
for i, r in enumerate(results):
    print(f"\n--- Doc {i} ---\n{r.page_content[:200]}")
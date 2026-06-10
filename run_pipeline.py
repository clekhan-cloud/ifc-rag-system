from src.ingestion.pdf_parser import *
from src.chunking.text_chunker import *
from src.embeddings.text_embeddings import *
from src.vectorstores.faiss_store import *
from src.retrieval.retriever import *
from src.llm.gemini_client import *

pages = extract_pdf_text(
    "data/pdf/IFC_Annual_Report_2024.pdf"
)

chunks = chunk_documents(
    pages
)

embeddings = get_embeddings(
    [c["content"] for c in chunks]
)

dim = embeddings.shape[1]

store = FAISSStore(dim)

store.add(embeddings)

retriever = Retriever(
    chunks,
    store
)

docs = retriever.retrieve(
    "What is IFC mission?"
)
for doc in docs:
    print("=" * 80)
    print(doc["content"][:500])


answer = generate_answer(
    "What is IFC mission?",
    [d["content"] for d in docs]
)

print(answer)
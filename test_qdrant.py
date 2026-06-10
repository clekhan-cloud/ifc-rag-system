from src.ingestion.pdf_parser import (
    extract_pdf_text
)

from src.chunking.text_chunker import (
    chunk_documents
)

from src.embeddings.text_embeddings import (
    get_embeddings
)

from src.vectorstores.qdrant_store import (
    QdrantStore
)

pages = extract_pdf_text(
    "data/pdf/IFC_Annual_Report_2024.pdf"
)

chunks = chunk_documents(
    pages
)

embeddings = get_embeddings(
    [
        c["content"]
        for c in chunks
    ]
)

store = QdrantStore(
    "ifc_report",
    embeddings.shape[1]
)

store.add(
    embeddings,
    chunks
)

print(
    "Qdrant indexing complete."
)
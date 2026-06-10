from src.vectorstores.qdrant_store import QdrantStore
from src.retrieval.qdrant_retriever import QdrantRetriever

store = QdrantStore(
    collection_name="ifc_report",
    dim=384
)

retriever = QdrantRetriever(store)

docs = retriever.retrieve(
    "What is IFC mission?"
)

for doc in docs:
    print(doc)
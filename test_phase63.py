from src.retrieval.colpali_retriever import (
    ColPaliRetriever
)

retriever = ColPaliRetriever(
    "data/embeddings/patch_embeddings_small.pkl"
)

results = retriever.search(
    query="What does the IFC net income chart show?",
    top_k=5
)

print("\nTOP RESULTS\n")

for r in results:

    print(
        f"Patch ID : {r['patch_id']}"
    )

    print(
        f"Score    : {r['score']:.4f}"
    )

    print(
        f"Path     : {r['path']}"
    )

    print("-" * 50)
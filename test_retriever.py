from src.pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(
    "data/pdf/IFC_Annual_Report_2024.pdf"
)

docs = pipeline.retriever.retrieve(
    "What is IFC mission?",
    k=10
)

for doc in docs:
    print(
        f"Page: {doc['page']} | Score: {doc['score']:.4f}"
    )
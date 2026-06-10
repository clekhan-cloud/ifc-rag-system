from src.pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(
    "data/pdf/IFC_Annual_Report_2024.pdf"
)

questions = [
    "What is IFC mission?",
    "What was IFC net income in FY24?",
    "What are advisory services?"
]

for q in questions:

    print("\n" + "=" * 80)
    print("QUESTION:")
    print(q)

    answer, docs = pipeline.ask(q)

    print("\nANSWER:")
    print(answer)

    print("\nRETRIEVED PAGES:")

    for doc in docs:
        print(
            f"Page: {doc['page']} | Score: {doc['score']:.4f}"
        )
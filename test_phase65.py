from src.multimodal.multimodal_rag import (
    MultimodalRAG
)

rag = MultimodalRAG()

result = rag.answer(
    query="What is IFC's mission?"
)

print("\nANSWER:\n")
print(
    result["answer"]
)

print("\nRETRIEVED:\n")

for image in result["retrieved_images"]:

    print(image)
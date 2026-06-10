from src.generation.visual_generator import (
    VisualGenerator
)

generator = VisualGenerator()

answer = generator.generate_answer(
    question="What information is shown in this image?",
    image_path="data/page_images/page_1.png"
)

print("\nANSWER:\n")
print(answer)
# test_single_patch.py

from src.embeddings.colpali_embeddings import (
    ColPaliEmbeddings
)

embedder = ColPaliEmbeddings()

embedding = embedder.embed_image(
    "data/patches/patch_0.png"
)

print(type(embedding))

try:
    print(
        "Shape:",
        embedding.shape
    )
except:
    print(
        "No shape attribute"
    )
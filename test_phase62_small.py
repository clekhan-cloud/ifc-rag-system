from src.embeddings.colpali_embeddings import (
    ColPaliEmbeddings
)

import os
import pickle

PATCH_DIR = "data/patches"

OUTPUT_FILE = (
    "data/embeddings/"
    "patch_embeddings_small.pkl"
)

embedder = ColPaliEmbeddings()

patch_files = sorted(
    [
        f
        for f in os.listdir(PATCH_DIR)
        if f.endswith(".png")
    ]
)

# First 20 patches only

patch_files = patch_files[:20]

patch_store = []

print(
    f"Embedding {len(patch_files)} patches..."
)

for idx, patch_file in enumerate(
    patch_files,
    start=1
):

    patch_path = os.path.join(
        PATCH_DIR,
        patch_file
    )

    print(
        f"[{idx}/{len(patch_files)}] "
        f"{patch_file}"
    )

    embedding = embedder.embed_image(
        patch_path
    )

    patch_store.append(
        {
            "patch_id": idx,
            "path": patch_path,
            "embedding": embedding
        }
    )

os.makedirs(
    "data/embeddings",
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "wb"
) as f:

    pickle.dump(
        patch_store,
        f
    )

print(
    "\nSaved:",
    OUTPUT_FILE
)

print(
    "Total embeddings:",
    len(patch_store)
)

print(
    "Embedding shape:",
    patch_store[0]["embedding"].shape
)
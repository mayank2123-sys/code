import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Load FAISS
index = faiss.read_index("faiss_index.index")

# Load metadata
with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# User query
query = input("Ask Question: ")

# BGE instruction
query = "Represent this sentence for searching relevant passages: " + query

# Query embedding
query_embedding = model.encode(
    query,
    normalize_embeddings=True
)

query_embedding = np.array([query_embedding]).astype("float32")

# 🔥 ONLY BEST MATCH
scores, indices = index.search(query_embedding, 1)

print("\n" + "=" * 100)

for rank, idx in enumerate(indices[0]):

    # similarity threshold
    if scores[0][rank] < 0.65:
        print("No strong match found")
        continue

    result = metadata[idx]

    print("\nURL:")
    print(result["url"])

    print("\nCHUNK:")
    print(result["chunk"])

    print("\nSIMILARITY SCORE:")
    print(scores[0][rank])

print("\n" + "=" * 100)
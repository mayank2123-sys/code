import json
import faiss
import numpy as np

# Load embedding JSON
with open("embedding1_output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Store vectors
vectors = []

# Store metadata
metadata = []

# Loop through all URLs
for item in data:

    section = item.get("section")
    url = item.get("url")
    headings = item.get("headings", [])

    chunk_embeddings = item.get("chunk_embeddings", [])

    # Loop through chunks
    for ce in chunk_embeddings:

        chunk = ce["chunk"]
        embedding = ce["embedding"]

        # Add vector
        vectors.append(embedding)

        # Add metadata
        metadata.append({
            "section": section,
            "url": url,
            "headings": headings,
            "chunk": chunk
        })

# Convert vectors to numpy array
vectors = np.array(vectors).astype("float32")

# Get embedding dimension
dimension = vectors.shape[1]

# Create FAISS index
index = faiss.IndexFlatIP(dimension)

# Add vectors
index.add(vectors)

# Save FAISS index
faiss.write_index(index, "faiss_index.index")

# Save metadata
with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("✅ VECTOR DATABASE CREATED!")

print(f"📦 Total vectors stored: {index.ntotal}")
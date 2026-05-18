import json
from sentence_transformers import SentenceTransformer


# Load embedding model

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Load chunked file

with open("chunked2_output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Final output

final_output = []

# Total chunk counter

total_chunks = 0

# Generate embeddings

for item in data:

    section = item.get("section")
    url = item.get("url")
    headings = item.get("headings", [])
    chunks = item.get("chunks", [])

    # Count chunks per URL
    chunk_count = len(chunks)

    # Add to total
    total_chunks += chunk_count

    # Store all embeddings
    embeddings = []

    for chunk in chunks:

        # Generate embedding
        embedding = model.encode(chunk).tolist()

        embeddings.append({
            "chunk": chunk,
            "embedding": embedding
        })

    # Save complete structure
    final_output.append({
        "section": section,
        "url": url,
        "headings": headings,
        "total_chunks": chunk_count,
        "chunk_embeddings": embeddings
    })


# Save embeddings JSON

with open("embedding1_output.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

print(" EMBEDDINGS GENERATED")


# Print output

for item in final_output:

    print("\n" + "=" * 100)

    print("\nURL:")
    print(item["url"])

    # Print chunk count
    print("\nTOTAL CHUNKS:", item["total_chunks"])

    print("\nCHUNKS + EMBEDDINGS:\n")

    for ce in item["chunk_embeddings"]:

        print("CHUNK:\n")
        print(ce["chunk"])

        print("\nEMBEDDING:\n")
        print(ce["embedding"])

        print("\n" + "-" * 80)

# Final total chunks

print("\n" + "=" * 100)
print("TOTAL CHUNKS IN WHOLE FILE:", total_chunks)
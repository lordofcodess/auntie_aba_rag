"""Check why admissions data isn't being retrieved."""
import chromadb
from chromadb.utils import embedding_functions

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="handbooks", embedding_function=embedding_function)

all_data = collection.get(include=["metadatas", "documents"])

# Find admissions guide chunks
adm = []
for meta, doc in zip(all_data["metadatas"], all_data["documents"]):
    if meta.get("source_file") == "UG_Admissions_Complete_Guide.md":
        adm.append((meta, doc))

print(f"Total admissions guide chunks: {len(adm)}")
print(f"Sample metadata keys: {list(adm[0][0].keys()) if adm else 'none'}")
print(f"Sample metadata: {adm[0][0] if adm else 'none'}")

# Check which ones mention 'computer engineering'
matches = [(m, d) for m, d in adm if "computer engineering" in d.lower()]
print(f"\nChunks mentioning 'computer engineering': {len(matches)}")
for m, d in matches[:3]:
    print(f"\n--- {m.get('content_type')} ---")
    print(f"dept field: {m.get('department')} | school: {m.get('school')}")
    print(f"heading_path: {m.get('heading_path')}")
    print(f"text preview: {d[:300]}")

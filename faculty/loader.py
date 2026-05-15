"""
faculty/loader.py — Embed and load faculty chunks into a dedicated Chroma collection.

Run:
    python -m faculty.loader
"""

import json
import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

CHUNKS_PATH    = Path("data/processed_faculty/faculty_chunks.jsonl")
CHROMA_DIR     = "chroma_db"
COLLECTION_NAME = "faculty"
EMBED_MODEL    = "all-mpnet-base-v2"
BATCH_SIZE     = 64   # embed in batches to avoid memory issues


def get_faculty_collection(chroma_dir: str = CHROMA_DIR):
    """Return (or create) the faculty Chroma collection."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def load_chunks(chunks_path: Path = CHUNKS_PATH) -> list[dict]:
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Faculty chunks not found: {chunks_path}\n"
            "Run chunker first: python -m faculty.chunker"
        )
    chunks = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)
    return chunks


def upsert_to_chroma(collection, chunks: list[dict]):
    """Upsert chunks in batches. Uses upsert so re-running is safe."""
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        logger.info("Upserted batch %d-%d / %d", i + 1, min(i + BATCH_SIZE, total), total)
    logger.info("Finished upserting %d chunks into '%s'", total, collection.name)


def run(chroma_dir: str = CHROMA_DIR):
    chunks     = load_chunks()
    collection = get_faculty_collection(chroma_dir)

    # Optional: clear existing data before reload for a clean refresh
    existing = collection.count()
    if existing > 0:
        logger.info("Collection '%s' has %d existing records — upserting (will update changed records)",
                    COLLECTION_NAME, existing)

    upsert_to_chroma(collection, chunks)

    final_count = collection.count()
    logger.info("Faculty collection now contains %d records", final_count)
    return final_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    count = run()
    print(f"\nDone. Faculty collection contains {count} records in Chroma.")

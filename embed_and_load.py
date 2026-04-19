"""
Embed chunks from JSONL files and load into Chroma vector database.

Usage:
    source venv/bin/activate
    python embed_and_load.py chunks_out/*.jsonl --db chroma_db/
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Generator

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer


def load_chunks(jsonl_files: list[str]) -> Generator[dict, None, None]:
    """Load chunks from JSONL files."""
    for file_path in jsonl_files:
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def embed_and_load(
    jsonl_files: list[str],
    db_dir: str = "chroma_db",
    model_name: str = "all-mpnet-base-v2",
    batch_size: int = 100,
):
    """Embed chunks and load into Chroma."""

    print(f"Loading embedding model: {model_name}")

    # Use Chroma's sentence-transformers wrapper for consistency
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )

    print(f"Connecting to Chroma database at {db_dir}")
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(
        name="handbooks",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Loading chunks from {len(jsonl_files)} file(s)")

    batch_ids = []
    batch_texts = []
    batch_embeddings = []
    batch_metadata = []

    total_chunks = 0
    chunk_counter = 0
    for chunk in load_chunks(jsonl_files):
        total_chunks += 1
        chunk_counter += 1

        # Prepare chunk data with unique sequential ID
        text = chunk["text"]
        chunk_id = f"chunk_{chunk_counter:06d}"

        batch_ids.append(chunk_id)
        batch_texts.append(text)

        # Include content_type in metadata
        metadata = chunk["metadata"].copy()
        metadata["content_type"] = chunk.get("content_type", "unknown")
        batch_metadata.append(metadata)

        # Batch insertion (embedding handled by Chroma's embedding function)
        if len(batch_ids) >= batch_size:
            print(f"Embedding batch ({total_chunks} chunks processed)")
            collection.add(
                ids=batch_ids,
                metadatas=batch_metadata,
                documents=batch_texts,
            )
            batch_ids.clear()
            batch_texts.clear()
            batch_metadata.clear()

    # Insert remaining batch
    if batch_ids:
        print(f"Embedding final batch ({total_chunks} chunks processed)")
        collection.add(
            ids=batch_ids,
            metadatas=batch_metadata,
            documents=batch_texts,
        )

    print(f"\nSuccessfully loaded {total_chunks} chunks into Chroma at {db_dir}")
    print(f"Collection 'handbooks' contains {collection.count()} chunks")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Embed chunks and load into Chroma.")
    ap.add_argument("chunks", nargs="+", help="JSONL chunk file(s)")
    ap.add_argument("--db", default="chroma_db", help="Chroma database directory")
    ap.add_argument(
        "--model",
        default="all-mpnet-base-v2",
        help="Sentence-transformers model name",
    )
    ap.add_argument("--batch-size", type=int, default=100, help="Embedding batch size")
    args = ap.parse_args(argv)

    # Expand glob patterns
    files = []
    for pattern in args.chunks:
        p = Path(pattern)
        if "*" in str(p):
            files.extend(sorted(p.parent.glob(p.name)))
        else:
            files.append(p)

    embed_and_load(
        [str(f) for f in files],
        db_dir=args.db,
        model_name=args.model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

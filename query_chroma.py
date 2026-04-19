"""
Query the Chroma vector database for handbook information.

Usage:
    source venv/bin/activate
    python query_chroma.py "What courses are in Level 200?" --top-k 5
"""

import argparse
import json

import chromadb


def query(
    query_text: str,
    db_dir: str = "chroma_db",
    top_k: int = 5,
):
    """Query the Chroma database and return top results."""
    import warnings
    warnings.filterwarnings('ignore')

    print(f"Connecting to database...")
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(name="handbooks")

    print(f"Database contains {collection.count()} chunks")
    print(f"Querying: '{query_text}'\n")

    print("Generating query embedding...")
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        print(f"Query error: {e}")
        return

    for i, (doc, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        1,
    ):
        similarity = 1 - distance  # Convert distance to similarity
        print(f"[{i}] Similarity: {similarity:.3f}")
        print(f"    Source: {metadata['source_file']}")
        print(f"    Type: {metadata['content_type']}")
        if metadata.get("programme"):
            print(f"    Programme: {metadata['programme']}")
        if metadata.get("level"):
            print(f"    Level: {metadata['level']}")
        if metadata.get("course_code"):
            print(f"    Course: {metadata['course_code']}")
        print(f"    Text: {doc[:200]}...\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Query Chroma handbook database.")
    ap.add_argument("query", help="Query text")
    ap.add_argument("--db", default="chroma_db", help="Chroma database directory")
    ap.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    args = ap.parse_args(argv)

    query(args.query, db_dir=args.db, top_k=args.top_k)


if __name__ == "__main__":
    main()

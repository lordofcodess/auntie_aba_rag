"""
RAG chat interface using Gemini API and Chroma retrieval.

Retrieves relevant handbook chunks and uses Gemini to generate answers.

Usage:
    source venv/bin/activate
    export GEMINI_API_KEY="your-api-key"
    python rag_chat.py --interactive

Or single query:
    python rag_chat.py "What are the Level 200 Computer Science courses?"
"""

import argparse
import os
import sys

import chromadb
from google import genai
from chromadb.utils import embedding_functions


class HandbookRAG:
    """RAG system combining Chroma retrieval with Gemini generation."""

    def __init__(self, db_dir: str = "chroma_db", model: str = "gemini-2.5-flash"):
        """Initialize RAG system with Chroma and Gemini."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.model_name = model
        self.client = genai.Client(api_key=api_key)

        # Initialize Chroma with embedding function
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )
        client = chromadb.PersistentClient(path=db_dir)
        self.collection = client.get_collection(
            name="handbooks", embedding_function=embedding_function
        )

        self.system_prompt = """You are an expert advisor for University of Ghana handbooks.
You have access to handbook information about academic programmes, courses, and regulations.

When answering questions:
1. Use ONLY the provided handbook context to answer
2. Be specific and cite the programme/level/department when relevant
3. If information is not in the handbooks, clearly state that
4. Format course listings clearly with course codes and titles when possible
5. Be helpful and conversational but accurate"""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve relevant chunks from Chroma with level-aware filtering."""
        # Extract level from query (e.g., "Level 200" → 200)
        level_filter = None
        for word in query.split():
            if word.isdigit() and 100 <= int(word) <= 400:
                level_filter = int(word)
                break

        # Retrieve significantly more results to account for semantic drift
        # Level information doesn't embed well numerically
        retrieve_k = max(top_k * 5, 50)

        results = self.collection.query(
            query_texts=[query],
            n_results=retrieve_k,
            include=["documents", "metadatas"]
        )

        chunks = []
        # Group results by level to better balance
        by_level = {}
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            meta_level = metadata.get("level")
            if meta_level not in by_level:
                by_level[meta_level] = []
            by_level[meta_level].append({"text": doc, "metadata": metadata})

        # Prioritize requested level, then add others
        if level_filter and level_filter in by_level:
            chunks.extend(by_level[level_filter][:top_k])
        else:
            # Fallback: return highest-scoring results for any level
            for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                chunks.append({"text": doc, "metadata": metadata})
                if len(chunks) >= top_k:
                    break

        return chunks[:top_k]

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        """Generate answer using Gemini with retrieved context."""
        # Build context string
        context_str = "HANDBOOK CONTEXT:\n" + "=" * 50 + "\n"
        for i, chunk in enumerate(context_chunks, 1):
            meta = chunk["metadata"]
            source = meta.get("source_file", "Unknown")
            content_type = meta.get("content_type", "Unknown")
            level = meta.get("level", "N/A")
            dept = meta.get("department", "N/A")

            context_str += f"\n[Chunk {i}] {content_type.upper()} | {source}\n"
            context_str += f"Level: {level} | Department: {dept}\n"
            context_str += f"Text: {chunk['text']}\n"
            context_str += "-" * 50 + "\n"

        # Build prompt
        prompt = f"""{self.system_prompt}

{context_str}

USER QUESTION: {query}

Please answer based on the handbook context provided above."""

        # Call Gemini
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text

    def chat(self, query: str, top_k: int = 5) -> dict:
        """Full RAG pipeline: retrieve + generate."""
        print(f"\n🔍 Retrieving relevant chunks...")
        chunks = self.retrieve(query, top_k=top_k)

        print(f"✓ Found {len(chunks)} relevant chunks")
        print(f"📝 Generating answer...\n")

        answer = self.generate(query, chunks)

        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "source_file": c["metadata"].get("source_file"),
                    "level": c["metadata"].get("level"),
                    "department": c["metadata"].get("department"),
                }
                for c in chunks
            ],
        }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="RAG chat interface for University of Ghana handbooks."
    )
    ap.add_argument(
        "query",
        nargs="?",
        help="Single query to ask (omit for interactive mode)",
    )
    ap.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve",
    )
    ap.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use",
    )
    ap.add_argument(
        "--db",
        default="chroma_db",
        help="Chroma database directory",
    )
    args = ap.parse_args(argv)

    try:
        rag = HandbookRAG(db_dir=args.db, model=args.model)
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        print("Please set GEMINI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    if args.interactive or (not args.query):
        # Interactive mode
        print("\n" + "=" * 60)
        print("📚 University of Ghana Handbook RAG Chat")
        print("=" * 60)
        print("Ask questions about handbooks. Type 'exit' or 'quit' to stop.\n")

        while True:
            try:
                query = input("You: ").strip()
                if not query:
                    continue
                if query.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break

                result = rag.chat(query, top_k=args.top_k)
                print(f"\nAssistant: {result['answer']}")
                print(f"\n📌 Sources: {len(result['sources'])} chunks retrieved")
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
    else:
        # Single query mode
        result = rag.chat(args.query, top_k=args.top_k)
        print(f"Assistant: {result['answer']}")
        print(f"\n📌 Sources: {len(result['sources'])} chunks retrieved")
        for src in result["sources"]:
            print(f"  - {src['source_file']} (Level {src['level']}, {src['department']})")


if __name__ == "__main__":
    main()

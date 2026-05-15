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

"""
RAG chat interface using Gemini API and Chroma retrieval.

Retrieves relevant handbook chunks and uses Gemini to generate answers.
Faculty/staff questions are routed to the separate 'faculty' Chroma collection.

Usage:
    source venv/bin/activate
    export GEMINI_API_KEY="your-api-key"
    python rag_chat.py --interactive

Or single query:
    python rag_chat.py "What are the Level 200 Computer Science courses?"
"""

import argparse
import glob
import json
import logging
import os
import re
import sys

import chromadb
from google import genai
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Faculty question detection  (NEW)
# ---------------------------------------------------------------------------

FACULTY_SIGNALS = [
    "prof", "professor", "dr.", "dr ", "lecturer", "faculty", "staff",
    "who is", "who are", "teaches", "department of", "email of",
    "contact", "research interest", "research interests", "academic staff",
]

FACULTY_LIST_SIGNALS = [
    "all lecturers", "all faculty", "all staff", "list of lecturers",
    "list of faculty", "lecturers in", "faculty in", "faculty at",
    "staff in", "staff at", "professors in", "who are the",
]

DEPT_KEYWORDS = {
    "computer science": "Department of Computer Science",
    "dcs":              "Department of Computer Science",
    "law":              "School of Law",
    "political science":"Department of Political Science",
    "business":         "University of Ghana Business School",
    "ugbs":             "University of Ghana Business School",
    "education":        "Department of Education Studies and Leadership",
    "esl":              "Department of Education Studies and Leadership",
}


def is_faculty_question(query: str) -> bool:
    q = query.lower()
    return any(s in q for s in FACULTY_SIGNALS)


def is_faculty_list_question(query: str) -> bool:
    q = query.lower()
    return any(s in q for s in FACULTY_LIST_SIGNALS)


def extract_dept_filter(query: str) -> str | None:
    q = query.lower()
    for kw, dept in DEPT_KEYWORDS.items():
        if kw in q:
            return dept
    return None


def extract_name_query(query: str) -> str | None:
    filler = (r"\b(who|is|are|tell|me|about|the|a|an|prof|professor|dr|mr|mrs|ms|"
              r"lecturer|staff|faculty|member|of|department|computer science|ug|"
              r"university|ghana|please|can|you|find|search|give|information|on|for)\b")
    name = re.sub(filler, "", query, flags=re.IGNORECASE)
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name if len(name) > 2 else None


# ---------------------------------------------------------------------------
# Faculty retriever  (NEW)
# ---------------------------------------------------------------------------

class FacultyRetriever:
    def __init__(self, chroma_dir: str = "chroma_db"):
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )
        client = chromadb.PersistentClient(path=chroma_dir)
        try:
            self.collection = client.get_collection(name="faculty", embedding_function=ef)
            logger.info("Faculty collection loaded: %d records", self.collection.count())
        except Exception:
            self.collection = None
            logger.warning("Faculty collection not found — run: python refresh.py")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if self.collection is None:
            return []

        name_query  = extract_name_query(query)
        dept_filter = extract_dept_filter(query)
        is_list     = is_faculty_list_question(query)
        n           = top_k * 4 if is_list else top_k * 3

        where = None
        if dept_filter and not is_list:
            where = {"department": {"$eq": dept_filter}}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n, self.collection.count() or 1),
                include=["documents", "metadatas", "distances"],
                where=where,
            )
        except Exception as e:
            logger.warning("Faculty retrieval failed: %s", e)
            return []

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({"text": doc, "metadata": meta,
                           "similarity": 1 - dist, "score": 1 - dist})

        # Score boosting
        for chunk in chunks:
            meta  = chunk["metadata"]
            cname = (meta.get("name") or "").lower()
            ctype = meta.get("content_type", "staff_profile")
            cdept = meta.get("department", "")

            if name_query and name_query.lower() in cname:
                chunk["score"] += 0.4
            elif name_query:
                overlap = set(name_query.lower().split()) & set(cname.split())
                if overlap:
                    chunk["score"] += 0.2 * (len(overlap) / len(name_query.split()))

            if ctype == "faculty_profile":        chunk["score"] += 0.1
            elif ctype == "admin_staff_profile":  chunk["score"] -= 0.05

            if dept_filter and dept_filter.lower() in cdept.lower():
                chunk["score"] += 0.15

        chunks.sort(key=lambda x: x["score"], reverse=True)
        return chunks if is_list else chunks[:top_k]


# ---------------------------------------------------------------------------
# Main RAG class  (your original, with faculty routing added)
# ---------------------------------------------------------------------------

class HandbookRAG:
    """RAG system combining Chroma retrieval with Gemini generation."""

    def __init__(self, db_dir: str = "chroma_db", model: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.model_name = model
        self.client = genai.Client(api_key=api_key)

        # Handbook collection (your original)
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )
        client = chromadb.PersistentClient(path=db_dir)
        self.collection = client.get_collection(
            name="handbooks", embedding_function=embedding_function
        )

        # Faculty retriever (NEW)
        self.faculty_retriever = FacultyRetriever(chroma_dir=db_dir)

        self.system_prompt = """You are Auntie Aba, a helpful assistant for the University of Ghana.

You have access to two types of indexed data:
1. Faculty/staff profiles from UG department websites.
2. Handbook, policy, and regulation documents from UG.

When answering questions:
1. Use ONLY the provided context to answer.
2. Be specific and cite the programme/level/department when relevant.
3. If the information is not in the context, clearly state that.
4. For faculty questions: include name, title, department, email, office, and research interests when available.
5. For handbook questions: format course listings clearly with course codes and titles when possible.
6. Do not refer to faculty context as "handbook context"."""

        self._build_bm25_index()

    def _build_bm25_index(self, chunks_dir: str = "chunks_out"):
        """Build BM25 index from JSONL chunk files."""
        self.bm25_corpus_texts = []
        self.bm25_corpus_meta = []
        self.bm25 = None

        try:
            for path in sorted(glob.glob(f"{chunks_dir}/*.jsonl")):
                with open(path) as f:
                    for line in f:
                        chunk = json.loads(line)
                        self.bm25_corpus_texts.append(chunk["text"])
                        self.bm25_corpus_meta.append(chunk.get("metadata", {}))

            if self.bm25_corpus_texts:
                tokenized = [text.lower().split() for text in self.bm25_corpus_texts]
                self.bm25 = BM25Okapi(tokenized)
                logger.info(f"BM25 index built with {len(self.bm25_corpus_texts)} documents")
        except Exception as e:
            logger.warning(f"Failed to build BM25 index: {e}")
            self.bm25 = None

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Hybrid retrieval: BM25 + semantic search with RRF."""
        bm25_results = {}
        if self.bm25:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            top_bm25_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 3]
            for rank, idx in enumerate(top_bm25_idx):
                text_key = self.bm25_corpus_texts[idx][:100]
                bm25_results[text_key] = {
                    "text": self.bm25_corpus_texts[idx],
                    "metadata": self.bm25_corpus_meta[idx],
                    "bm25_rank": rank,
                    "similarity": None,
                }

        retrieve_k = max(top_k * 5, 50)
        results = self.collection.query(
            query_texts=[query],
            n_results=retrieve_k,
            include=["documents", "metadatas", "distances"]
        )

        semantic_results = {}
        for rank, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            text_key = doc[:100]
            semantic_results[text_key] = {
                "text": doc,
                "metadata": meta,
                "semantic_rank": rank,
                "similarity": 1 - dist,
            }

        K = 60
        merged = {}
        for text_key, item in bm25_results.items():
            merged[text_key] = {**item, "rrf_score": 1 / (K + item["bm25_rank"])}
        for text_key, item in semantic_results.items():
            sem_rrf = 1 / (K + item["semantic_rank"])
            if text_key in merged:
                merged[text_key]["rrf_score"] += sem_rrf
                merged[text_key]["similarity"] = item["similarity"]
                merged[text_key]["semantic_rank"] = item["semantic_rank"]
            else:
                merged[text_key] = {**item, "rrf_score": sem_rrf}

        ranked = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)
        chunks = self._apply_filters(query, ranked, top_k)
        return chunks[:top_k]

    def _apply_filters(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        """Apply level, department, and handbook priority filters."""
        level_filter = None
        for word in query.split():
            if word.isdigit() and 100 <= int(word) <= 400:
                level_filter = int(word)
                break

        by_level_dept_source = {}
        for chunk in chunks:
            meta = chunk["metadata"]
            meta_level = meta.get("level")
            department = meta.get("department", "Unknown")
            source_file = meta.get("source_file", "Unknown")
            key = (meta_level, department, source_file)
            if key not in by_level_dept_source:
                by_level_dept_source[key] = []
            by_level_dept_source[key].append(chunk)

        handbook_priority = {
            "CBAS handbook 2017.md": 0,
            "CHS handbook 2017.md": 1,
            "Humanities Handbook 2017.md": 2,
        }

        deprioritize_depts = set()
        if "computer science" in query.lower():
            deprioritize_depts.add("DEPARTMENT OF INFORMATION TECHNOLOGY")

        sorted_keys = sorted(
            by_level_dept_source.keys(),
            key=lambda k: (
                k[1] in deprioritize_depts,
                k[0] != level_filter if level_filter else False,
                k[2] not in handbook_priority,
                handbook_priority.get(k[2], 999),
                k[2],
            ),
        )

        filtered = []
        for key in sorted_keys:
            filtered.extend(by_level_dept_source[key])
            if len(filtered) >= top_k:
                break

        if "computer science" in query.lower():
            cscd_chunks  = [c for c in filtered if "CSCD" in c["text"]]
            csit_chunks  = [c for c in filtered if "CSIT" in c["text"] and "CSCD" not in c["text"]]
            other_chunks = [c for c in filtered if "CSCD" not in c["text"] and "CSIT" not in c["text"]]
            filtered = cscd_chunks + other_chunks + csit_chunks

        return filtered

    def generate(self, query: str, context_chunks: list[dict], context_type: str = "handbook") -> str:
        """Generate answer using Gemini with retrieved context."""
        if context_type == "faculty":
            context_str = "FACULTY PROFILE CONTEXT:\n" + "=" * 50 + "\n"
            for i, chunk in enumerate(context_chunks, 1):
                context_str += f"\n[Profile {i}]\n{chunk['text']}\n"
                context_str += "-" * 50 + "\n"
        else:
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

        prompt = f"""{self.system_prompt}

{context_str}

USER QUESTION: {query}

Please answer based on the context provided above."""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text

    def chat(self, query: str, top_k: int = 5) -> dict:
        """Full RAG pipeline: retrieve + generate. Routes faculty questions separately."""

        # NEW: route faculty questions to faculty collection
        if is_faculty_question(query):
            logger.info(f"Faculty route: {query}")
            chunks = self.faculty_retriever.retrieve(query, top_k=top_k)
            answer = self.generate(query, chunks, context_type="faculty")
            return {
                "query": query,
                "answer": answer,
                "sources": [
                    {
                        "source_file": c["metadata"].get("profile_url", "faculty_index"),
                        "level": None,
                        "department": c["metadata"].get("department"),
                        "content_type": c["metadata"].get("content_type"),
                    }
                    for c in chunks
                ],
                "route": "faculty_index",
            }

        # Original handbook path — unchanged
        logger.info(f"Retrieving relevant chunks for query: {query}")
        chunks = self.retrieve(query, top_k=top_k)
        logger.info(f"Found {len(chunks)} relevant chunks")
        logger.info("Generating answer with Gemini")

        answer = self.generate(query, chunks, context_type="handbook")

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
            "route": "document_rag",
        }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="RAG chat interface for University of Ghana handbooks."
    )
    ap.add_argument("query", nargs="?", help="Single query to ask (omit for interactive mode)")
    ap.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    ap.add_argument("--top-k", type=int, default=10, help="Number of chunks to retrieve")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use")
    ap.add_argument("--db", default="chroma_db", help="Chroma database directory")
    args = ap.parse_args(argv)

    try:
        rag = HandbookRAG(db_dir=args.db, model=args.model)
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        print("Please set GEMINI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    if args.interactive or (not args.query):
        print("\n" + "=" * 60)
        print("📚 University of Ghana RAG Chat")
        print("=" * 60)
        print("Ask questions about handbooks or faculty. Type 'exit' to stop.\n")

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
                print(f"\n📌 Route: {result.get('route','?')} | Sources: {len(result['sources'])}")
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
    else:
        result = rag.chat(args.query, top_k=args.top_k)
        print(f"Assistant: {result['answer']}")
        print(f"\n📌 Route: {result.get('route','?')} | Sources: {len(result['sources'])}")
        for src in result["sources"]:
            print(f"  - {src['source_file']} (Level {src.get('level')}, {src.get('department')})")


if __name__ == "__main__":
    main()
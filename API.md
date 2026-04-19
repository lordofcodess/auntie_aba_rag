# Auntie Aba RAG API

FastAPI wrapper exposing the RAG system with three endpoints.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Set your Gemini API key:
```bash
export GEMINI_API_KEY="your-key-here"
```

## Running the Server

```bash
uvicorn api:app --reload
```

Server starts at `http://localhost:8000`. Interactive docs at `/docs`.

---

## Endpoints

### POST /chat

Full RAG pipeline: retrieve relevant chunks + generate answer with Gemini.

**Request:**
```json
{
  "query": "What are Level 300 Computer Science courses?",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "What are Level 300 Computer Science courses?",
  "answer": "Based on the CBAS handbook...",
  "sources": [
    {
      "source_file": "CBAS handbook 2017.md",
      "level": 300,
      "department": "DEPARTMENT OF COMPUTER SCIENCE",
      "content_type": "programme_table"
    },
    ...
  ]
}
```

---

### POST /search

Retrieve chunks without calling the LLM. Includes similarity scores.

**Request:**
```json
{
  "query": "gender equity definition",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "gender equity definition",
  "results": [
    {
      "text": "This is the official definition of 'Gender equity'...",
      "metadata": {
        "source_file": "Gender Policy_5.md",
        "content_type": "definition",
        "term": "Gender equity",
        "document_title": "GENDER POLICY",
        ...
      },
      "similarity": 0.92
    },
    ...
  ]
}
```

---

### POST /chunk

Upload a markdown document and chunk it with the appropriate chunker (handbook, policy, or generic).

**Request:**
```bash
curl -X POST http://localhost:8000/chunk \
  -F "file=@my_handbook.md"
```

**Response:**
```json
{
  "filename": "my_handbook.md",
  "chunker": "HandbookChunker",
  "total_chunks": 142,
  "chunks": [
    {
      "text": "...",
      "content_type": "programme_table",
      "breadcrumb": "Volume > College > Department",
      "metadata": {...}
    },
    ...
  ]
}
```

---

### GET /health

Liveness check.

**Response:**
```json
{
  "status": "ok",
  "db_loaded": true
}
```

---

## Examples

**Retrieve CS Level 300 courses:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Level 300 Computer Science", "top_k": 3}'
```

**Ask about admission requirements:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the admission requirements for Engineering?"}'
```

**Upload a new policy document:**
```bash
curl -X POST http://localhost:8000/chunk \
  -F "file=@new_policy.md"
```

---

## Performance Notes

- **First startup is slow** (~20s) — loads the sentence-transformer model once
- **Search queries are fast** (<200ms per query)
- **Chat queries call Gemini** (2-5s depending on network + model latency)
- **File uploads are immediate** — chunking is done synchronously on-request

---

## Architecture

```
API Request
    ↓
FastAPI lifespan loads RAG system once
    ↓
HandbookRAG.chat() or .retrieve()
    ↓
Chroma vector DB (persistent, ~5.7k chunks)
    ↓
Gemini (chat only)
    ↓
JSON response
```

See [README.md](README.md) for RAG system design.

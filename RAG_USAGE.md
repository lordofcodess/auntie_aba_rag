# Handbook RAG System - Usage Guide

Complete retrieval-augmented generation system for University of Ghana handbooks powered by Gemini API.

## Setup

### 1. Get Gemini API Key (Free)
- Visit: https://aistudio.google.com/app/apikey
- Create a new API key
- Set environment variable:
  ```bash
  export GEMINI_API_KEY="your-api-key-here"
  ```

### 2. Activate Virtual Environment
```bash
source venv/bin/activate
```

## Using the System

### Interactive Chat Mode
```bash
python rag_chat.py --interactive
```
Starts an interactive chat where you can ask multiple questions.

**Example:**
```
You: What are the admission requirements for Psychology programmes?
Assistant: [Retrieves relevant chunks and generates answer]

You: What Level 200 courses does Computer Science offer?
Assistant: [Context-aware response with specific courses]

You: exit
Goodbye!
```

### Single Query Mode
```bash
python rag_chat.py "What are the core courses for Level 300 engineering?"
```

### Advanced Options
```bash
# Retrieve more chunks for better context
python rag_chat.py "Your question" --top-k 10

# Use different Gemini model
python rag_chat.py "Your question" --model gemini-1.5-pro

# Use custom database
python rag_chat.py "Your question" --db /path/to/chroma_db
```

## System Components

### 1. Chunking (`chunk_all.py`)
- Splits handbooks into semantic chunks
- Preserves hierarchy: Volume → College → School → Department → Programme
- Extracts: course tables, descriptions, narratives
- Output: JSONL files with chunks + metadata

**Generated:**
- CBAS handbook 2017: 1,391 chunks
- CHS handbook 2017: 854 chunks
- Humanities handbook 2017: 2,068 chunks
- **Total: 4,313 chunks**

### 2. Embedding (`embed_and_load.py`)
- Embeds chunks using `all-mpnet-base-v2` (768 dimensions)
- Stores in Chroma persistent vector database
- Enables fast semantic search

### 3. Retrieval (`query_chroma.py`)
- Searches vector database by semantic similarity
- Returns top-k relevant chunks
- Preserves metadata for context

### 4. Generation (`rag_chat.py`)
- Retrieves relevant chunks for query
- Passes to Gemini with system prompt
- Generates accurate, sourced answers

## Architecture

```
User Query
    ↓
Query Embedding (all-mpnet-base-v2)
    ↓
Vector Search (Chroma)
    ↓
Top-K Chunk Retrieval
    ↓
Prompt Construction [Context + Query]
    ↓
Gemini API
    ↓
Generated Answer
```

## Performance

**Vector Database:**
- 4,313 chunks
- 54 MB persistent storage
- Fast cosine similarity search

**Gemini Models:**
- `gemini-2.5-flash`: **Recommended** - Fast, cheap, high quality
- `gemini-2.5-pro`: Higher quality, slower, more expensive
- `gemini-2.0-flash`: Older but still available

**Latency:**
- Retrieval: <100ms
- Gemini generation: 1-5 seconds (depends on model)

## Example Queries

```bash
# Course information
"What are the Level 200 courses in Computer Science?"
"List all core courses for Psychology Level 100"

# Programme information
"What are the admission requirements for Engineering?"
"Which programmes have a Level 400 project?"

# Regulations and structure
"What is the GPA requirement for probation?"
"How many credits are needed to graduate?"
"What are the elective options in Level 300?"
```

## Troubleshooting

**Issue:** `GEMINI_API_KEY not set`
- **Solution:** `export GEMINI_API_KEY="your-key"`

**Issue:** Chroma database not found
- **Solution:** Ensure you ran `python embed_and_load.py chunks_out/*.jsonl`

**Issue:** Poor answer quality
- **Solution:** Increase `--top-k` (e.g., `--top-k 10`)
- **Solution:** Try `--model gemini-1.5-pro` for better quality

**Issue:** Empty responses
- **Solution:** Check Gemini API quota/rate limits
- **Solution:** Verify API key is valid

## Cost

- **Gemini API:** Free tier available, ~$0.075/1M tokens for flash
- **Vector Database:** Local (free)
- **Embeddings:** One-time (free, runs locally)

## Future Enhancements

- [ ] Multi-turn conversation history
- [ ] Source highlighting and page references
- [ ] Confidence scores on answers
- [ ] Batch query processing
- [ ] Web interface (Streamlit/FastAPI)
- [ ] Support for other handbook formats

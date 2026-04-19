"""FastAPI wrapper for auntie_aba_rag RAG system.

Usage:
    uvicorn api:app --reload

Visit http://localhost:8000/docs for interactive docs.
"""

import logging
import pathlib
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chunk_all import pick_chunker
from contextualize import template_contextualize
from rag_chat import HandbookRAG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

rag: Optional[HandbookRAG] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load RAG system at startup, clean up at shutdown."""
    global rag
    try:
        rag = HandbookRAG()
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise
    yield
    logger.info("RAG system shutting down")


app = FastAPI(
    title="Auntie Aba RAG API",
    description="RAG API for University of Ghana handbooks and policies",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (set to specific URLs in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models
# ============================================================================


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class Source(BaseModel):
    source_file: Optional[str] = None
    level: Optional[int] = None
    department: Optional[str] = None
    content_type: Optional[str] = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]


class SearchResult(BaseModel):
    text: str
    metadata: dict
    similarity: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class ChunkResult(BaseModel):
    filename: str
    chunker: str
    total_chunks: int
    chunks: list[dict]


# ============================================================================
# Routes
# ============================================================================


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "db_loaded": rag is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Full RAG pipeline: retrieve chunks + generate answer with Gemini."""
    if rag is None:
        return {"detail": "RAG system not initialized"}, 500

    result = rag.chat(req.query, top_k=req.top_k)
    return result


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Retrieve chunks without calling the LLM."""
    if rag is None:
        return {"detail": "RAG system not initialized"}, 500

    chunks = rag.retrieve(req.query, top_k=req.top_k)
    return {"query": req.query, "results": chunks}


@app.post("/chunk", response_model=ChunkResult)
async def chunk_doc(file: UploadFile = File(...)):
    """Upload and chunk a markdown document."""
    if rag is None:
        return {"detail": "RAG system not initialized"}, 500

    # Save to temp file with original filename suffix (needed for routing)
    suffix = pathlib.Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=f"_{file.filename}", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = pathlib.Path(tmp.name)

    try:
        chunker = pick_chunker(tmp_path)
        chunks = chunker.chunk_file(tmp_path)
        chunks = [template_contextualize(c) for c in chunks]

        return {
            "filename": file.filename,
            "chunker": chunker.__class__.__name__,
            "total_chunks": len(chunks),
            "chunks": [c.to_dict() for c in chunks],
        }
    finally:
        tmp_path.unlink()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

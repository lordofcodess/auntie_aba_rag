"""
FastAPI wrapper around HandbookRAG.

Usage:
    source venv/bin/activate
    export GEMINI_API_KEY="your-api-key"
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /health        Liveness check
    POST /chat          Full RAG: retrieve + generate
    POST /retrieve      Retrieval only (debug / preview)
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from rag_chat import HandbookRAG, SELF_DESCRIPTION
from transcript import advise as analyze_transcript
from speech import transcribe_audio
from directions import rewrite_steps


ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}

ALLOWED_AUDIO_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mp3",
    "audio/mpeg",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/aac",
    "audio/ogg",
    "audio/flac",
    "audio/webm",
    "audio/aiff",
}

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB cap
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB cap for audio


rag: Optional[HandbookRAG] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    print("🚀 Loading HandbookRAG (this takes a few seconds)...")
    db_dir = os.getenv("CHROMA_DIR", "chroma_db")
    rag = HandbookRAG(db_dir=db_dir)
    print(f"✓ Ready to serve requests (chroma_dir={db_dir})")
    yield
    # No teardown needed


app = FastAPI(
    title="UG Handbook RAG API",
    description="Hybrid RAG over University of Ghana handbooks and policies.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the frontend to call us from a different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MAX_HISTORY_TURNS = 20


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=20000)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(10, ge=1, le=50, description="Number of chunks to retrieve")
    history: list[HistoryTurn] = Field(
        default_factory=list,
        description="Prior conversation turns for context. Retrieval still targets the latest turn only.",
    )


class Source(BaseModel):
    source_file: Optional[str] = None
    level: Optional[int] = None
    department: Optional[str] = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]
    probing: bool = False
    chitchat: bool = False


class RetrieveResponse(BaseModel):
    query: str
    chunks: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "ready": rag is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if HandbookRAG.is_self_question(req.query):
        return ChatResponse(
            query=req.query,
            answer=SELF_DESCRIPTION,
            sources=[],
            probing=False,
            chitchat=False,
        )

    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not yet initialized")
    try:
        history = [t.model_dump() for t in req.history[-MAX_HISTORY_TURNS:]]
        result = rag.chat(req.query, top_k=req.top_k, history=history)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: ChatRequest):
    """Return raw retrieved chunks without calling Gemini. Useful for debugging."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not yet initialized")
    try:
        chunks = rag.retrieve(req.query, top_k=req.top_k)
        return RetrieveResponse(query=req.query, chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


MAX_NOTES_LEN = 4000


@app.post("/voice/chat")
async def voice_chat(
    file: UploadFile = File(..., description="Audio recording (wav/mp3/webm/m4a/ogg/flac)"),
    top_k: int = Form(10, description="Number of chunks to retrieve"),
):
    """Transcribe an audio question via Gemini, then run the transcript through /chat."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not yet initialized")

    if not (1 <= top_k <= 50):
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 50")

    mime_type = (file.content_type or "").lower()
    if mime_type not in ALLOWED_AUDIO_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type '{mime_type}'. Allowed: {sorted(ALLOWED_AUDIO_MIME_TYPES)}",
        )

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio too large ({len(audio_bytes)} bytes). Max: {MAX_AUDIO_BYTES}",
        )

    # Normalize common aliases to what Gemini accepts
    alias_map = {
        "audio/x-wav": "audio/wav",
        "audio/mpeg": "audio/mp3",
        "audio/x-m4a": "audio/m4a",
    }
    gemini_mime = alias_map.get(mime_type, mime_type)

    try:
        transcript = transcribe_audio(
            client=rag.client,
            file_bytes=audio_bytes,
            mime_type=gemini_mime,
            model=rag.model_name,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")

    if not transcript or transcript.strip() == "[UNINTELLIGIBLE]":
        raise HTTPException(
            status_code=422,
            detail="Could not transcribe audio — recording may be silent or unintelligible",
        )

    try:
        result = rag.chat(transcript, top_k=top_k)
        result["transcript"] = transcript
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcript/analyze")
async def transcript_analyze(
    file: UploadFile = File(..., description="Transcript PDF or image"),
    notes: Optional[str] = Form(
        None,
        description="Optional free-text: focus area, goals, specific questions, etc.",
    ),
):
    """Upload a transcript (PDF or image) plus optional context notes, and get
    graduation + electives + class standing advice."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not yet initialized")

    mime_type = (file.content_type or "").lower()
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{mime_type}'. Allowed: {sorted(ALLOWED_MIME_TYPES)}",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes)} bytes). Max: {MAX_UPLOAD_BYTES}",
        )

    if notes is not None and len(notes) > MAX_NOTES_LEN:
        raise HTTPException(
            status_code=413,
            detail=f"Notes too long ({len(notes)} chars). Max: {MAX_NOTES_LEN}",
        )

    if mime_type == "image/jpg":
        mime_type = "image/jpeg"

    try:
        result = analyze_transcript(
            client=rag.client,
            rag=rag,
            file_bytes=file_bytes,
            mime_type=mime_type,
            notes=notes,
            model=rag.model_name,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RewriteStepsRequest(BaseModel):
    raw_steps: list[str] = Field(
        ..., min_length=1, max_length=40,
        description="Walking step instructions from Maps JS DirectionsService."
    )
    from_name: str = Field(..., min_length=1, max_length=200)
    to_name: str = Field(..., min_length=1, max_length=200)
    distance_label: str = Field(..., min_length=1, max_length=40, description='e.g. "1.0 km"')
    duration_minutes: int = Field(..., ge=0, le=600)


class RewriteStepsResponse(BaseModel):
    steps: list[str]


@app.post("/directions/rewrite-steps", response_model=RewriteStepsResponse)
def directions_rewrite_steps(req: RewriteStepsRequest):
    """Rewrite raw walking steps as natural conversational sentences.

    Frontend computes the route via Maps JS DirectionsService and posts the
    raw step instruction strings. Returns the same number of sentences in
    the same order.
    """
    try:
        steps = rewrite_steps(
            req.raw_steps,
            from_name=req.from_name,
            to_name=req.to_name,
            distance_label=req.distance_label,
            duration_minutes=req.duration_minutes,
        )
        return RewriteStepsResponse(steps=steps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

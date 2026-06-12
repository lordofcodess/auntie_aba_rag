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

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from auth import User, enforce_quota, get_current_user, quota_status
from rag_chat import HandbookRAG, SELF_DESCRIPTION
from transcript import advise as analyze_transcript
from cv import advise as analyze_cv
from speech import transcribe_audio
from directions import rewrite_steps
from document_context import build_document_history_content
from document_files import (
    DOCX_MIME_TYPE,
    DOCX_TEMPLATE_MIME_TYPE,
    infer_mime_from_filename,
    normalize_document_upload,
)


ALLOWED_MIME_TYPES = {
    "application/pdf",
    DOCX_MIME_TYPE,
    DOCX_TEMPLATE_MIME_TYPE,
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
    content: str = Field(..., max_length=60000)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: Optional[int] = Field(
        None, ge=1, le=50,
        description="Number of chunks to retrieve. If omitted, derived from mode (fast=5, thinking=15).",
    )
    mode: Literal["fast", "thinking"] = Field(
        "fast",
        description="Answer mode. 'fast' skips Gemini's reasoning step; 'thinking' enables it.",
    )
    history: list[HistoryTurn] = Field(
        default_factory=list,
        description="Prior conversation turns for context. Retrieval still targets the latest turn only.",
    )


class Source(BaseModel):
    source_file: Optional[str] = None
    level: Optional[int] = None
    department: Optional[str] = None


class Citation(BaseModel):
    uri: str
    title: str


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]
    probing: bool = False
    chitchat: bool = False
    mode: Optional[str] = None
    citations: list[Citation] = []
    via_web: bool = False


class RetrieveResponse(BaseModel):
    query: str
    chunks: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "ready": rag is not None}


class FeedbackRequest(BaseModel):
    rating: Optional[Literal["like", "dislike"]] = Field(
        None, description="Optional thumbs rating attached to the feedback."
    )
    message: str = Field(..., min_length=1, max_length=4000)
    context: Optional[str] = Field(
        None, max_length=10000,
        description="Optional surrounding context (e.g. last assistant message).",
    )
    email: Optional[str] = Field(None, max_length=200)


FEEDBACK_LOG_PATH = os.path.join(
    os.getenv("FEEDBACK_DIR", os.path.dirname(os.path.abspath(__file__))),
    "feedback.jsonl",
)


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Append a feedback entry to a local JSONL log."""
    import datetime as _dt
    import json as _json
    entry = {
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "rating": req.rating,
        "message": req.message,
        "context": req.context,
        "email": req.email,
    }
    try:
        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not record feedback: {e}")
    return {"ok": True}


# ─── Auth & quota helpers ─────────────────────────────────────────────────

def gate(
    request: Request,
    user: Optional[User] = Depends(get_current_user),
) -> Optional[User]:
    """Composed dependency: verify JWT (if any) then check anonymous quota."""
    return enforce_quota(request, user)


@app.get("/auth/quota")
def auth_quota(
    request: Request,
    user: Optional[User] = Depends(get_current_user),
):
    """Frontend can call this to display 'N free messages remaining'."""
    return quota_status(request, user)


@app.get("/auth/me")
def auth_me(user: Optional[User] = Depends(get_current_user)):
    if user is None:
        return {"authenticated": False}
    return {"authenticated": True, "id": user.id, "email": user.email}


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    user: Optional[User] = Depends(gate),
):
    if HandbookRAG.is_self_question(req.query):
        return ChatResponse(
            query=req.query,
            answer=SELF_DESCRIPTION,
            sources=[],
            probing=False,
            chitchat=False,
            mode=req.mode,
        )

    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not yet initialized")
    try:
        history = [t.model_dump() for t in req.history[-MAX_HISTORY_TURNS:]]
        result = rag.chat(req.query, top_k=req.top_k, mode=req.mode, history=history)
        result["mode"] = req.mode
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
    user: Optional[User] = Depends(gate),
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
    file: UploadFile = File(..., description="Transcript PDF, Word document, or image"),
    notes: Optional[str] = Form(
        None,
        description="Optional free-text: focus area, goals, specific questions, etc.",
    ),
    user: Optional[User] = Depends(gate),
):
    """Upload a transcript (PDF or image) plus optional context notes, and get
    graduation + electives + class standing advice."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not yet initialized")

    mime_type = infer_mime_from_filename(file.filename, (file.content_type or "").lower())
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
        file_bytes, mime_type = normalize_document_upload(file_bytes, mime_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        result = analyze_transcript(
            client=rag.client,
            rag=rag,
            file_bytes=file_bytes,
            mime_type=mime_type,
            notes=notes,
            model=rag.model_name,
        )
        result["doc_type"] = "transcript"
        result["assistant_history_content"] = build_document_history_content(
            advice=result.get("advice") or "",
            doc_type="transcript",
            extracted=result.get("extracted") or {},
            notes=result.get("notes"),
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _read_and_validate_upload(
    file: UploadFile,
    notes: Optional[str],
) -> tuple[bytes, str]:
    """Shared validation for document uploads. Returns (bytes, normalized_mime)."""
    mime_type = infer_mime_from_filename(file.filename, (file.content_type or "").lower())
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{mime_type}'. Allowed: {sorted(ALLOWED_MIME_TYPES)}",
        )
    if notes is not None and len(notes) > MAX_NOTES_LEN:
        raise HTTPException(
            status_code=413,
            detail=f"Notes too long ({len(notes)} chars). Max: {MAX_NOTES_LEN}",
        )
    if mime_type == "image/jpg":
        mime_type = "image/jpeg"
    return mime_type


async def _read_upload_bytes(file: UploadFile) -> bytes:
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes)} bytes). Max: {MAX_UPLOAD_BYTES}",
        )
    return file_bytes


def _normalize_upload_bytes(file_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    try:
        return normalize_document_upload(file_bytes, mime_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


CLASSIFY_PROMPT = """You are a document classifier. Look at the attached file and decide which ONE of these labels best describes it:

- transcript : an academic transcript (lists courses, grades, GPA, semester results)
- cv : a CV / résumé (work experience, education, skills, projects)
- other : anything else (essay, certificate, ID card, letter, syllabus, etc.)

Return ONLY the label (one word: transcript, cv, or other). No explanation."""


def _classify_document(client, file_bytes: bytes, mime_type: str, model: str) -> str:
    """One quick multimodal call to label the document. Falls back to 'other'."""
    from google.genai import types as gt
    try:
        part = gt.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        resp = client.models.generate_content(model=model, contents=[part, CLASSIFY_PROMPT])
        label = (resp.text or "").strip().lower().split()[0] if resp.text else "other"
        label = label.strip(".,:;")
        if label not in ("transcript", "cv", "other"):
            return "other"
        return label
    except Exception:
        return "other"


@app.post("/cv/analyze")
async def cv_analyze(
    file: UploadFile = File(..., description="CV / résumé PDF, Word document, or image"),
    notes: Optional[str] = Form(
        None,
        description="Optional: target role, focus area, specific question.",
    ),
    user: Optional[User] = Depends(gate),
):
    """Upload a CV (PDF or image) plus optional notes; get strengths, weaknesses,
    career-fit, UG-programme matches, and prioritized next steps."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not yet initialized")
    mime_type = _read_and_validate_upload(file, notes)
    file_bytes = await _read_upload_bytes(file)
    file_bytes, mime_type = _normalize_upload_bytes(file_bytes, mime_type)
    try:
        result = analyze_cv(
            client=rag.client,
            rag=rag,
            file_bytes=file_bytes,
            mime_type=mime_type,
            notes=notes,
            model=rag.model_name,
        )
        result["doc_type"] = "cv"
        result["assistant_history_content"] = build_document_history_content(
            advice=result.get("advice") or "",
            doc_type="cv",
            extracted=result.get("extracted") or {},
            notes=result.get("notes"),
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/document/analyze")
async def document_analyze(
    file: UploadFile = File(..., description="Any supported document (PDF, Word document, or image)"),
    notes: Optional[str] = Form(None, description="Optional context notes."),
    user: Optional[User] = Depends(gate),
):
    """Auto-detect the document type (transcript / cv / other) and route to the
    matching analyser. Frontend uses this when the user shouldn't have to pick.

    Response includes `doc_type` so the UI knows which kind came back. For
    `other`, returns a friendly note that we only handle transcripts and CVs."""
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not yet initialized")
    mime_type = _read_and_validate_upload(file, notes)
    file_bytes = await _read_upload_bytes(file)
    file_bytes, mime_type = _normalize_upload_bytes(file_bytes, mime_type)

    label = _classify_document(rag.client, file_bytes, mime_type, rag.model_name)
    try:
        if label == "transcript":
            result = analyze_transcript(
                client=rag.client, rag=rag, file_bytes=file_bytes,
                mime_type=mime_type, notes=notes, model=rag.model_name,
            )
            result["doc_type"] = "transcript"
            result["assistant_history_content"] = build_document_history_content(
                advice=result.get("advice") or "",
                doc_type="transcript",
                extracted=result.get("extracted") or {},
                notes=result.get("notes"),
            )
            return result
        if label == "cv":
            result = analyze_cv(
                client=rag.client, rag=rag, file_bytes=file_bytes,
                mime_type=mime_type, notes=notes, model=rag.model_name,
            )
            result["doc_type"] = "cv"
            result["assistant_history_content"] = build_document_history_content(
                advice=result.get("advice") or "",
                doc_type="cv",
                extracted=result.get("extracted") or {},
                notes=result.get("notes"),
            )
            return result
        # 'other' — we don't have a dedicated analyser yet
        return {
            "doc_type": "other",
            "extracted": {},
            "notes": notes,
            "advice": (
                "I couldn't tell whether this was a transcript or a CV — it looks "
                "like something else (e.g. an essay, certificate, or ID). I only "
                "analyse transcripts and CVs right now. If this IS a transcript "
                "or CV, upload it via the dedicated transcript or CV option to "
                "force the right pipeline."
            ),
            "handbook_chunks_used": 0,
        }
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

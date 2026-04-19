"""
Contextual retrieval: for each chunk, generate a short blurb describing
where it sits in the source document, and prepend it to the chunk text
before embedding.

Two implementations:

  1. template_contextualize() — no LLM needed. Generates a deterministic
     one-sentence blurb from the chunk's metadata (breadcrumb, content type,
     course code, etc.). Fast, free, good enough for structured docs.

  2. llm_contextualize() — calls Claude with the chunk and its parent
     section, gets back a 1-2 sentence natural-language blurb. Slower and
     costs tokens, but gives richer context for unstructured docs.

Both return a new Chunk with the blurb prepended to .text (and also stored
separately in .metadata['context_blurb'] so you can re-generate without
re-embedding).

Usage:
    from chunker import HandbookChunker
    from contextualize import template_contextualize

    chunks = HandbookChunker().chunk_file("handbook.md")
    chunks = [template_contextualize(c) for c in chunks]
    # chunks now have richer .text for embedding

For the LLM version you'll need an Anthropic API key:
    from anthropic import Anthropic
    client = Anthropic()  # reads ANTHROPIC_API_KEY env var
    chunks = [llm_contextualize(c, parent_text=p, client=client) for c, p in ...]
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

# We import Chunk lazily to avoid a hard dependency if users only want
# template-based contextualization.
try:
    from chunker import Chunk
except ImportError:  # pragma: no cover
    Chunk = Any  # type: ignore


# ---------------------------------------------------------------------------
# Template-based contextualization (no LLM)
# ---------------------------------------------------------------------------

def template_contextualize(chunk: "Chunk") -> "Chunk":
    """Generate a deterministic context blurb from chunk metadata and
    prepend it to the chunk text. Returns a new Chunk."""
    blurb = _template_blurb(chunk)
    if not blurb:
        return chunk
    new_text = f"{blurb}\n\n{chunk.text}"
    new_meta = {**chunk.metadata, "context_blurb": blurb}
    return replace(chunk, text=new_text, metadata=new_meta)


def _template_blurb(chunk: "Chunk") -> str:
    m = chunk.metadata
    ct = chunk.content_type

    if ct == "programme_table":
        parts = []
        if m.get("programme"):
            parts.append(f"the {m['programme']} programme")
        elif m.get("department"):
            parts.append(f"the {_titleize(m['department'])}")
        elif m.get("school"):
            parts.append(f"the {_titleize(m['school'])}")
        if m.get("level"):
            parts.append(f"Level {m['level']}")
        if m.get("semester"):
            parts.append(f"Semester {m['semester']}")
        qualifier = " ".join(parts) if parts else "a University of Ghana programme"
        return (
            f"This is a programme-structure table listing the courses for "
            f"{qualifier}. Each row gives course code, title, core/elective "
            f"status, and credit weight."
        )

    if ct == "course_description":
        code = m.get("course_code", "")
        title = m.get("course_title", "")
        locator = code + (f" ({title})" if title else "")
        where = ""
        if m.get("department"):
            where = f" taught by the {_titleize(m['department'])}"
        elif m.get("school"):
            where = f" offered at the {_titleize(m['school'])}"
        return (
            f"This is the official course description for {locator}{where}. "
            f"It describes learning outcomes, topics covered, and is the "
            f"definitive reference for what the course contains."
        )

    if ct == "narrative":
        section = m.get("section") or "this programme"
        scope = _best_scope(m)
        return (
            f"This is narrative / prose content from {section}"
            + (f" in {scope}" if scope else "")
            + ". It typically covers programme overview, admission "
            f"requirements, regulations, or faculty listings."
        )

    # Section handling — check for policy vs generic
    if ct == "section" or ct == "section_intro":
        # Policy document sections (from PolicyChunker)
        if m.get("section_number") is not None or m.get("document_type", "").replace("_", " ") != "document":
            section_num = m.get("section_number")
            section_title = m.get("section_title")
            doc_title = m.get("document_title", "document")
            doc_type = m.get("document_type", "document").replace("_", " ")

            section_locator = ""
            if section_num and section_title:
                section_locator = f"Section {section_num} ({section_title})"
            elif section_num:
                section_locator = f"Section {section_num}"
            elif section_title:
                section_locator = section_title

            if section_locator:
                return (
                    f"This is {section_locator} from {doc_title.upper()} — a {doc_type} "
                    f"of the University of Ghana. It sets out rules, procedures, or "
                    f"requirements binding on the University community."
                )
            return (
                f"This is from {doc_title.upper()} — a {doc_type} of the University of Ghana. "
                f"It sets out rules, procedures, or requirements binding on the University community."
            )
        # Generic chunker sections
        else:
            path = " > ".join(m.get("heading_path", []))
            if path:
                return f"This section is located at: {path}."
            return ""

    if ct == "definition":
        term = m.get("term", "a term")
        doc_title = m.get("document_title", "document")
        return (
            f"This is the official definition of '{term}' as used in {doc_title.upper()}. "
            f"It is the authoritative meaning to apply when interpreting this document."
        )

    if ct == "definitions_table":
        doc_title = m.get("document_title", "document")
        term_count = m.get("term_count", "several")
        return (
            f"This is the complete definitions section of {doc_title.upper()}, "
            f"listing {term_count} defined terms. Use it to look up how specific "
            f"terms are used in this document."
        )

    return ""


def _titleize(name: str) -> str:
    """Turn 'SCHOOL OF VETERINARY MEDICINE' into 'School of Veterinary Medicine'."""
    small = {"of", "and", "the", "for", "in", "a", "to"}
    words = name.lower().split()
    out = []
    for i, w in enumerate(words):
        if i > 0 and w in small:
            out.append(w)
        else:
            out.append(w.capitalize())
    return " ".join(out)


def _best_scope(m: dict) -> str:
    for key in ("programme", "department", "school", "college"):
        if m.get(key):
            return _titleize(m[key])
    return ""


# ---------------------------------------------------------------------------
# LLM-based contextualization
# ---------------------------------------------------------------------------

class _AnthropicLike(Protocol):
    """Duck type for the anthropic.Anthropic client."""
    def messages(self) -> Any: ...  # simplified; real client has .messages.create()


CONTEXT_PROMPT = """\
Here is a document excerpt with surrounding context:

<parent_context>
{parent_text}
</parent_context>

<chunk>
{chunk_text}
</chunk>

Write ONE to TWO short sentences describing what this chunk is about and \
where it sits in the document. The goal is to help a search system understand \
the chunk without reading the rest of the document. Do not summarize the chunk \
content itself — just describe its role and location. Respond with only the \
sentences, no preamble.\
"""


def llm_contextualize(
    chunk: "Chunk",
    parent_text: str,
    client: Any,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 150,
) -> "Chunk":
    """
    Use an LLM to generate a 1-2 sentence context blurb and prepend it.

    parent_text should be the surrounding section (e.g. the full programme
    overview, or the first 2000 chars of the document) so the model has
    something to anchor against. Pass the anthropic client as `client`.

    Uses Haiku by default because this is a high-volume / low-creativity
    task — Haiku is fast and cheap and the job is easy. Swap to Sonnet for
    higher quality if you have budget.
    """
    prompt = CONTEXT_PROMPT.format(
        parent_text=parent_text[:4000],
        chunk_text=chunk.text[:4000],
    )
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    blurb = resp.content[0].text.strip()
    if not blurb:
        return chunk
    new_text = f"{blurb}\n\n{chunk.text}"
    new_meta = {**chunk.metadata, "context_blurb": blurb, "context_source": "llm"}
    return replace(chunk, text=new_text, metadata=new_meta)


# ---------------------------------------------------------------------------
# Parent-section lookup helper
# ---------------------------------------------------------------------------

def build_parent_index(chunks: list["Chunk"]) -> dict[str, str]:
    """
    Build a map of {breadcrumb_prefix -> concatenated narrative text} so
    each chunk can retrieve the right parent context for llm_contextualize().
    Uses narrative chunks as the canonical "parent" for each section.
    """
    out: dict[str, str] = {}
    for c in chunks:
        if c.content_type in ("narrative", "section"):
            # Drop deepest breadcrumb element — the parent is one level up
            parts = c.breadcrumb.split(" > ")
            parent_bc = " > ".join(parts[:-1]) if len(parts) > 1 else c.breadcrumb
            out.setdefault(parent_bc, "")
            # Only keep the first narrative we see per parent (typically the
            # programme intro, which is exactly what we want)
            if not out[parent_bc]:
                out[parent_bc] = c.text
    return out


def find_parent(chunk: "Chunk", parent_index: dict[str, str]) -> str:
    """Walk up the breadcrumb looking for a matching parent in the index."""
    parts = chunk.breadcrumb.split(" > ")
    while parts:
        key = " > ".join(parts)
        if key in parent_index:
            return parent_index[key]
        parts.pop()
    return ""

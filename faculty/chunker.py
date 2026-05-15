"""
faculty/chunker.py — Convert normalized faculty records into Chroma-ready chunks.

Each record becomes one chunk with a rich text body and clean metadata.

Run:
    python -m faculty.chunker
"""

import json
import logging
import re
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

NORMALIZED_PATH = Path("data/processed_faculty/faculty_normalized.json")
CHUNKS_PATH     = Path("data/processed_faculty/faculty_chunks.jsonl")


# ---------------------------------------------------------------------------
# Text builder
# ---------------------------------------------------------------------------

def _build_chunk_text(record: dict) -> str:
    """
    Build a rich natural-language text block for embedding.
    Putting all fields as readable prose gives better semantic search results
    than just concatenating field values.
    """
    lines = []

    name  = record.get("name") or ""
    rank  = record.get("academic_rank") or ""
    title = record.get("title") or ""
    dept  = record.get("department") or ""
    coll  = record.get("college") or ""

    # Header line — most important for name-based retrieval
    header_parts = [p for p in [rank or title, name] if p]
    lines.append(", ".join(header_parts) if header_parts else name)

    if dept:  lines.append(f"Department: {dept}")
    if coll:  lines.append(f"College: {coll}")

    if record.get("email"):  lines.append(f"Email: {record['email']}")
    if record.get("phone"):  lines.append(f"Phone: {record['phone']}")
    if record.get("office"): lines.append(f"Office: {record['office']}")

    if record.get("biography_summary"):
        lines.append(f"Biography: {record['biography_summary']}")

    if record.get("research_interests"):
        lines.append("Research Interests: " + "; ".join(record["research_interests"]))

    if record.get("courses_taught"):
        lines.append("Courses Taught: " + "; ".join(record["courses_taught"]))

    if record.get("google_scholar_url"):
        lines.append(f"Google Scholar: {record['google_scholar_url']}")

    if record.get("orcid_url"):
        lines.append(f"ORCID: {record['orcid_url']}")

    if record.get("profile_url"):
        lines.append(f"Profile: {record['profile_url']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def _safe_meta(value) -> str | None:
    """Chroma metadata values must be str, int, float, or bool — not None or list."""
    if value is None:          return None
    if isinstance(value, str): return value.strip() or None
    return str(value)


def _build_metadata(record: dict) -> dict:
    """
    Build Chroma-safe metadata. Lists are joined to strings.
    None values are excluded (Chroma rejects None metadata values).
    """
    raw = {
        "name":               _safe_meta(record.get("name")),
        "title":              _safe_meta(record.get("title")),
        "academic_rank":      _safe_meta(record.get("academic_rank")),
        "department":         _safe_meta(record.get("department")),
        "college":            _safe_meta(record.get("college")),
        "email":              _safe_meta(record.get("email")),
        "phone":              _safe_meta(record.get("phone")),
        "office":             _safe_meta(record.get("office")),
        "profile_url":        _safe_meta(record.get("profile_url")),
        "google_scholar_url": _safe_meta(record.get("google_scholar_url")),
        "orcid_url":          _safe_meta(record.get("orcid_url")),
        "content_type":       _safe_meta(record.get("content_type")) or "staff_profile",
        # Lists → joined strings for Chroma compatibility
        "research_interests": "; ".join(record.get("research_interests") or []) or None,
        "courses_taught":     "; ".join(record.get("courses_taught") or [])     or None,
    }
    # Strip out None values — Chroma will crash on them
    return {k: v for k, v in raw.items() if v is not None}


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

def chunk_records(records: list[dict]) -> list[dict]:
    chunks = []
    for record in records:
        text = _build_chunk_text(record)
        if not text.strip():
            logger.debug("Skipping empty chunk for: %s", record.get("name"))
            continue

        meta = _build_metadata(record)

        chunk = {
            "id":       str(uuid.uuid4()),
            "text":     text,
            "metadata": meta,
        }
        chunks.append(chunk)

    logger.info("Created %d chunks from %d records", len(chunks), len(records))
    return chunks


def run():
    if not NORMALIZED_PATH.exists():
        raise FileNotFoundError(
            f"Normalized faculty file not found: {NORMALIZED_PATH}\n"
            "Run normalizer first: python -m faculty.normalizer"
        )

    records = json.loads(NORMALIZED_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d normalized records", len(records))

    chunks = chunk_records(records)

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info("Saved %d chunks to %s", len(chunks), CHUNKS_PATH)
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    chunks = run()
    print(f"\nDone. {len(chunks)} chunks saved to {CHUNKS_PATH}")
    print("\nSample chunk:")
    if chunks:
        print(chunks[0]["text"])
        print("\nMetadata:", chunks[0]["metadata"])

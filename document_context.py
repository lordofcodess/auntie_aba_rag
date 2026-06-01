"""Helpers for carrying uploaded-document context through chat history.

The UI can render the visible advice while persisting `assistant_history_content`
as the assistant turn. On later /chat calls, rag_chat extracts the hidden JSON
block and gives it to Gemini separately from normal conversation history.
"""

from __future__ import annotations

import json
import re
from typing import Any


START_MARKER = "<!-- nana-aba-document-context:v1"
END_MARKER = "nana-aba-document-context:end -->"

_DOCUMENT_CONTEXT_RE = re.compile(
    r"<!--\s*nana-aba-document-context:v1\s*(.*?)\s*nana-aba-document-context:end\s*-->",
    re.DOTALL | re.IGNORECASE,
)


def build_document_history_content(
    *,
    advice: str,
    doc_type: str,
    extracted: dict[str, Any],
    notes: str | None,
) -> str:
    """Return assistant content with a hidden structured document context block."""
    payload = {
        "doc_type": doc_type,
        "notes": notes,
        "extracted": extracted,
    }
    payload_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    visible = (advice or "").rstrip()
    return f"{visible}\n\n{START_MARKER}\n{payload_json}\n{END_MARKER}"


def strip_document_context(content: str) -> str:
    """Remove hidden document blocks from text meant to be shown as dialogue."""
    return _DOCUMENT_CONTEXT_RE.sub("", content or "").strip()


def extract_document_contexts(
    history: list[dict[str, Any]],
    *,
    max_contexts: int = 2,
) -> list[dict[str, Any]]:
    """Extract recent document context payloads from chat history."""
    contexts: list[dict[str, Any]] = []
    for turn in history:
        content = turn.get("content") or ""
        if not isinstance(content, str):
            continue
        for match in _DOCUMENT_CONTEXT_RE.finditer(content):
            raw = match.group(1).strip()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if not isinstance(payload.get("extracted"), dict):
                continue
            doc_type = payload.get("doc_type")
            if doc_type not in ("transcript", "cv"):
                continue
            contexts.append(
                {
                    "doc_type": doc_type,
                    "notes": payload.get("notes"),
                    "extracted": payload["extracted"],
                }
            )
    return contexts[-max_contexts:]


def format_document_contexts(
    contexts: list[dict[str, Any]],
    *,
    max_chars: int = 30000,
) -> str:
    """Render document contexts for a model prompt with a hard size cap."""
    if not contexts:
        return ""
    blocks = []
    for i, payload in enumerate(contexts, start=1):
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        blocks.append(f"DOCUMENT {i}:\n{body}")
    rendered = "\n\n".join(blocks).strip()
    if len(rendered) > max_chars:
        return rendered[:max_chars].rstrip() + "\n...[truncated]"
    return rendered

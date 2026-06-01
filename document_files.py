"""Upload file normalization for document analysis endpoints."""

from __future__ import annotations

import os
import io
import zipfile
import xml.etree.ElementTree as ET


DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
DOCX_TEMPLATE_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.template"
TEXT_MIME_TYPE = "text/plain"

WORD_MIME_TYPES = {
    DOCX_MIME_TYPE,
    DOCX_TEMPLATE_MIME_TYPE,
}


def infer_mime_from_filename(filename: str | None, fallback: str) -> str:
    """Fill in missing browser MIME types from the uploaded filename."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext == ".docx":
        return DOCX_MIME_TYPE
    if ext == ".dotx":
        return DOCX_TEMPLATE_MIME_TYPE
    return fallback


def extract_docx_text(file_bytes: bytes) -> str:
    """Extract visible text from a modern Word .docx/.dotx file."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            xml_bytes = zf.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as e:
        raise ValueError("Uploaded Word file is not a valid .docx/.dotx document") from e

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        raise ValueError("Could not read text from the uploaded Word document") from e

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.iterfind(".//w:p", ns):
        parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{{{ns['w']}}}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{{{ns['w']}}}tab":
                parts.append("\t")
            elif node.tag == f"{{{ns['w']}}}br":
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)

    extracted = "\n".join(paragraphs).strip()
    if not extracted:
        raise ValueError("No readable text found in the uploaded Word document")
    return extracted


def normalize_document_upload(file_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    """Convert supported office documents to text for Gemini."""
    if mime_type in WORD_MIME_TYPES:
        text = extract_docx_text(file_bytes)
        return text.encode("utf-8"), TEXT_MIME_TYPE
    return file_bytes, mime_type

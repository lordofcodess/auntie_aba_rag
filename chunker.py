"""
Chunkers for University of Ghana handbooks + generic markdown docs.

Two chunkers:
  - HandbookChunker: structure-aware, knows about programme tables, course
    descriptions, college/school/department/programme/level/semester hierarchy.
  - GenericHeadingChunker: walks markdown headings, chunks sections, splits
    long ones on paragraph boundaries. Fallback for anything else.

Both emit Chunk objects with .text, .content_type, .breadcrumb, .metadata.
Serialize to JSONL with chunks_to_jsonl().
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    content_type: str
    breadcrumb: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chunks_to_jsonl(chunks: list[Chunk], out_path: str | Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Shared cleanup
# ---------------------------------------------------------------------------

_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_PAGENUM_RE = re.compile(r"^\s*\d{1,4}\s*$")
_HTML_TAG_STRIP = re.compile(r"</?u>|<br\s*/?>", re.IGNORECASE)

# Programme type detection
_PROGRAMME_TYPE_RE = re.compile(
    r"\((?:Single\s+Major|Major-Minor|Combined\s+Major|Double\s+Major|Joint\s+Honours)\)|"
    r"(?:Single\s+Major|Major-Minor|Combined\s+Major|Double\s+Major|Joint\s+Honours)",
    re.IGNORECASE
)


def clean_line(line: str) -> str:
    line = _IMAGE_RE.sub("", line)
    line = _HTML_TAG_STRIP.sub(" ", line)
    return line.rstrip()


def is_page_number_line(line: str) -> bool:
    return bool(_PAGENUM_RE.match(line))


def extract_programme_type(text: str) -> str | None:
    """Extract programme type (Single Major, Major-Minor, Combined, etc.) from text."""
    # Check for explicit programme type markers
    match = _PROGRAMME_TYPE_RE.search(text)
    if match:
        prog_type = match.group(0).strip("()")
        return prog_type.title()

    # Check for combined programmes indicated by "X AND Y" pattern in headings
    # e.g., "ACCOUNTING AND ECONOMICS", "BANKING & FINANCE AND ECONOMICS"
    lines = text.split('\n')
    for line in lines[:5]:  # Check first few lines (likely headings)
        # Look for pattern like "WORD AND WORD" or "WORD & WORD AND WORD"
        if re.search(r'\b\w+\s+(?:and|&)\s+\w+', line, re.IGNORECASE):
            # Exclude patterns that are just "X AND Y Students" or "X AND Y STUDENTS"
            if not re.search(r'(?:student|structure|level|semester|course)', line, re.IGNORECASE):
                # This looks like a combined programme name
                return "Combined"

    return None


def preprocess_lines(raw_text: str) -> list[str]:
    """Strip image refs + standalone page numbers, normalize whitespace."""
    out = []
    for line in raw_text.splitlines():
        if is_page_number_line(line):
            continue
        out.append(clean_line(line))
    return out


# ---------------------------------------------------------------------------
# Heading parsing
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
# Strip markdown emphasis markers and underline tags from heading text
_HEADING_CLEAN_RE = re.compile(r"[*_`]|</?u>", re.IGNORECASE)


def parse_heading(line: str) -> tuple[int, str] | None:
    m = _HEADING_RE.match(line)
    if not m:
        return None
    level = len(m.group(1))
    text = _HEADING_CLEAN_RE.sub("", m.group(2)).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".:")
    return level, text


# Lines that aren't markdown headings but function as section markers in
# these handbooks — bold-only single-line labels like "**First Semester**"
# and italic-bold variants like "***Level 200***".
_PSEUDO_HEADING_RE = re.compile(r"^\s*\*{2,3}([^*\n]+?)\*{2,3}\s*$")


def parse_pseudo_heading(line: str) -> str | None:
    m = _PSEUDO_HEADING_RE.match(line)
    if not m:
        return None
    text = m.group(1).strip().rstrip(".:")
    text = re.sub(r"\s+", " ", text)
    return text or None


# ---------------------------------------------------------------------------
# Semantic classification of headings
# ---------------------------------------------------------------------------

# Course code at start of heading: "UGRC 141:", "UGRC141:", "BCMB 303", ranges "UGRC 220-239"
_COURSE_CODE_RE = re.compile(
    r"^([A-Z]{3,5})\s*(\d{3})(?:\s*[-–]\s*\d{3})?\s*[:\-]?\s*(.*)$"
)
_LEVEL_RE = re.compile(r"\bLEVEL\s*(\d{3})\b", re.IGNORECASE)
_SEMESTER_NAMED_RE = re.compile(r"\b(FIRST|SECOND)\s*SEMESTER\b", re.IGNORECASE)
_SEMESTER_NUMBERED_RE = re.compile(r"\bSEMESTER\s*([12]|[IVX]+)\b", re.IGNORECASE)


def _extract_semester(text: str) -> int | None:
    m = _SEMESTER_NAMED_RE.search(text)
    if m:
        return 1 if m.group(1).upper() == "FIRST" else 2
    m = _SEMESTER_NUMBERED_RE.search(text)
    if m:
        sem_str = m.group(1).upper()
        if sem_str in ("I", "1"):
            return 1
        elif sem_str in ("II", "2"):
            return 2
    return None
# Anchored at start — a programme heading leads with the degree marker
# (e.g. "BSc Administration", "BA French", "Bachelor of Arts"), rather than
# just mentioning "Bachelor" somewhere in the middle.
_PROGRAMME_RE = re.compile(
    r"^\s*(BSc\.?|B\.Sc\.?|BA\.?|B\.A\.?|BMus|BFA|MB\s*ChB|BDS|Bachelor\s+of|Diploma\s+in)\b",
    re.IGNORECASE,
)


# Words that look like a course-code prefix (3-5 uppercase letters) but
# aren't — usually heading words like "LEVEL 100", "YEAR 2", "PART I".
_NON_COURSE_PREFIXES = {
    "LEVEL", "YEAR", "UNIT", "PART", "TOTAL", "GROUP", "STAGE",
    "BLOCK", "TERM", "NOTE", "TABLE", "CORE",
}


def classify_heading(text: str) -> dict[str, Any]:
    """
    Return a dict of semantic tags detected in the heading.
    Keys that may be set: kind, course_code, course_title, level, semester,
    programme_name.

    Order matters: hierarchy markers (college/school/.../level/semester) are
    checked before course-code detection so a heading like "LEVEL 100 COURSES"
    doesn't get misread as course code "LEVEL 100".
    """
    out: dict[str, Any] = {"kind": "section", "raw": text}
    upper = text.upper()

    if "COLLEGE OF" in upper:
        out["kind"] = "college"
        return out

    if "SCHOOL OF" in upper or re.search(r"\bBUSINESS SCHOOL\b", upper):
        out["kind"] = "school"
        return out

    if "DEPARTMENT OF" in upper:
        out["kind"] = "department"
        return out

    # Programme detection: BSc / BA / Bachelor...
    if _PROGRAMME_RE.search(text):
        out["kind"] = "programme"
        out["programme_name"] = text
        lm = _LEVEL_RE.search(text)
        if lm:
            out["level"] = int(lm.group(1))
        return out

    # Level headings like "LEVEL 100 COURSES" or "B. LEVEL 200" or
    # "Level 100 courses – Semester 1"
    lm = _LEVEL_RE.search(text)
    if lm:
        out["kind"] = "level"
        out["level"] = int(lm.group(1))
        sem = _extract_semester(text)
        if sem is not None:
            out["semester"] = sem
        return out

    # Semester heading on its own
    sem = _extract_semester(text)
    if sem is not None:
        out["kind"] = "semester"
        out["semester"] = sem
        return out

    # Course code — must be LAST because this is the most permissive match
    m = _COURSE_CODE_RE.match(text)
    if m and m.group(1).upper() not in _NON_COURSE_PREFIXES:
        out["kind"] = "course"
        out["course_code"] = f"{m.group(1)} {m.group(2)}"
        title = m.group(3).strip(" :-")
        if title:
            out["course_title"] = title
        return out

    if "PROGRAMME STRUCTURE" in upper or "COURSE STRUCTURE" in upper:
        out["kind"] = "programme_structure"
        return out

    return out


# ---------------------------------------------------------------------------
# Table linearization
# ---------------------------------------------------------------------------

def linearize_table(html: str) -> tuple[str, dict[str, Any]]:
    """
    Convert an HTML <table> to pipe-separated text rows, and pull out any
    level/semester hints that live inside the table's spanning header cells
    (e.g. `<th colspan="4">SECOND SEMESTER</th>`).
    Returns (linearized_text, hints_dict).
    """
    soup = BeautifulSoup(html, "html.parser")
    lines: list[str] = []
    hints: dict[str, Any] = {}
    for row in soup.find_all("tr"):
        cells = [
            re.sub(r"\s+", " ", c.get_text(separator=" ", strip=True))
            for c in row.find_all(["th", "td"])
        ]
        cells = [c for c in cells if c]
        if cells:
            # Look for level/semester hints in cells (often in colspan headers)
            for c in cells:
                if "level" not in hints:
                    lm = _LEVEL_RE.search(c)
                    if lm:
                        hints["level"] = int(lm.group(1))
                if "semester" not in hints:
                    sem = _extract_semester(c)
                    if sem is not None:
                        hints["semester"] = sem
            lines.append(" | ".join(cells))
    return "\n".join(lines), hints


# ---------------------------------------------------------------------------
# Breadcrumb state
# ---------------------------------------------------------------------------

class BreadcrumbState:
    """
    Tracks the current semantic hierarchy as we walk headings.
    When a heading of kind K appears, all levels below K are reset.
    """

    HIERARCHY = ["college", "school", "department", "programme", "level", "semester"]

    def __init__(self, volume: str | None = None):
        self.volume = volume
        self._state: dict[str, Any] = {}
        # Track a free-form "section" for non-hierarchy headings
        self.current_section: str | None = None

    def update(self, classified: dict[str, Any]) -> None:
        kind = classified["kind"]
        if kind == "college":
            self._reset_from("college")
            self._state["college"] = classified["raw"]
        elif kind == "school":
            self._reset_from("school")
            self._state["school"] = classified["raw"]
        elif kind == "department":
            self._reset_from("department")
            self._state["department"] = classified["raw"]
        elif kind == "programme":
            self._reset_from("programme")
            self._state["programme"] = classified.get("programme_name", classified["raw"])
            if "level" in classified:
                self._state["level"] = classified["level"]
        elif kind == "level":
            self._reset_from("level")
            self._state["level"] = classified["level"]
            if "semester" in classified:
                self._state["semester"] = classified["semester"]
        elif kind == "semester":
            self._reset_from("semester")
            self._state["semester"] = classified["semester"]
        elif kind == "programme_structure":
            # Doesn't clear anything but marks context
            self.current_section = "Programme Structure"
        elif kind == "course":
            # A course-code heading resets the "section" tag but doesn't
            # touch the programme hierarchy
            self.current_section = None
        else:
            # Generic section heading, doesn't alter hierarchy
            self.current_section = classified["raw"]

    def _reset_from(self, kind: str) -> None:
        idx = self.HIERARCHY.index(kind)
        for k in self.HIERARCHY[idx:]:
            self._state.pop(k, None)

    def snapshot(self) -> dict[str, Any]:
        out = dict(self._state)
        if self.volume:
            out["volume"] = self.volume
        if self.current_section:
            out["section"] = self.current_section
        return out

    def breadcrumb_str(self) -> str:
        parts: list[str] = []
        if self.volume:
            parts.append(self.volume)
        for k in self.HIERARCHY:
            v = self._state.get(k)
            if v is None:
                continue
            if k == "level":
                parts.append(f"Level {v}")
            elif k == "semester":
                parts.append(f"Semester {v}")
            else:
                parts.append(str(v))
        if self.current_section and self.current_section not in parts:
            parts.append(self.current_section)
        return " > ".join(parts)


# ---------------------------------------------------------------------------
# Handbook-specific chunker
# ---------------------------------------------------------------------------

# Inferred from filename — tweak/add as you onboard more handbooks.
VOLUME_HINTS = [
    ("CBAS", "Volume 3: Sciences"),
    ("CHS", "Volume 4: Health Sciences"),
    ("Humanities", "Volume 2: Humanities"),
]


def infer_volume(file_path: str | Path) -> str | None:
    name = Path(file_path).name
    for key, label in VOLUME_HINTS:
        if key.lower() in name.lower():
            return label
    return None


class HandbookChunker:
    """Structure-aware chunker for UG handbooks."""

    # Cap narrative prose chunks at this many chars before splitting
    MAX_NARRATIVE_CHARS = 2000
    # Drop narrative chunks shorter than this (usually noise)
    MIN_NARRATIVE_CHARS = 80

    def chunk_file(self, path: str | Path) -> list[Chunk]:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        volume = infer_volume(path)
        return list(self._chunk_text(text, source_file=path.name, volume=volume))

    # ------------------------------------------------------------------ walk
    def _chunk_text(
        self, text: str, source_file: str, volume: str | None
    ) -> Iterator[Chunk]:
        lines = preprocess_lines(text)
        state = BreadcrumbState(volume=volume)

        i = 0
        n = len(lines)
        # Buffer for narrative prose under the current heading
        narrative_buf: list[str] = []
        # If currently inside a course description, collect until next heading
        in_course: dict[str, Any] | None = None
        course_buf: list[str] = []
        # Track line numbers for source_lines metadata
        buf_start_line = 0

        def flush_narrative(end_line: int) -> Iterator[Chunk]:
            nonlocal narrative_buf, buf_start_line
            if not narrative_buf:
                return
            body = "\n".join(narrative_buf).strip()
            narrative_buf = []
            if len(body) < self.MIN_NARRATIVE_CHARS:
                return
            for sub in self._split_narrative(body):
                yield self._make_chunk(
                    text=sub,
                    content_type="narrative",
                    state=state,
                    source_file=source_file,
                    source_lines=[buf_start_line, end_line],
                )

        def flush_course(end_line: int) -> Iterator[Chunk]:
            nonlocal in_course, course_buf
            if in_course is None:
                yield from ()
                return
            body = "\n".join(course_buf).strip()
            if body:
                meta = {
                    "course_code": in_course["course_code"],
                }
                if "course_title" in in_course:
                    meta["course_title"] = in_course["course_title"]
                yield self._make_chunk(
                    text=body,
                    content_type="course_description",
                    state=state,
                    source_file=source_file,
                    source_lines=[in_course["start_line"], end_line],
                    extra_metadata=meta,
                )
            in_course = None
            course_buf = []

        while i < n:
            line = lines[i]

            # Blank line — just accumulate
            if not line.strip():
                if in_course is not None:
                    course_buf.append(line)
                else:
                    narrative_buf.append(line)
                i += 1
                continue

            # Table block
            if line.lstrip().startswith("<table"):
                # Flush any pending narrative or course text
                yield from flush_narrative(i)
                yield from flush_course(i)
                # Collect full table
                start = i
                table_lines = [line]
                i += 1
                while i < n and "</table>" not in lines[i]:
                    table_lines.append(lines[i])
                    i += 1
                if i < n:
                    table_lines.append(lines[i])  # closing tag
                    i += 1
                table_html = "\n".join(table_lines)
                linearized, hints = linearize_table(table_html)
                if linearized:
                    # Use table-internal level/semester hints to fill gaps,
                    # but don't override what the breadcrumb already set.
                    extra = {}
                    snap = state.snapshot()
                    if "level" in hints and not snap.get("level"):
                        extra["level"] = hints["level"]
                    if "semester" in hints and not snap.get("semester"):
                        extra["semester"] = hints["semester"]
                    yield self._make_chunk(
                        text=linearized,
                        content_type="programme_table",
                        state=state,
                        source_file=source_file,
                        source_lines=[start + 1, i],
                        extra_metadata=extra or None,
                    )
                buf_start_line = i + 1
                continue

            # Heading line
            h = parse_heading(line)
            if h is not None:
                _, htext = h
                # Any pending buffers close here
                yield from flush_narrative(i)
                yield from flush_course(i)

                classified = classify_heading(htext)
                state.update(classified)

                if classified["kind"] == "course":
                    in_course = {
                        "course_code": classified["course_code"],
                        "start_line": i + 1,
                    }
                    if "course_title" in classified:
                        in_course["course_title"] = classified["course_title"]
                    course_buf = []
                    # Put the heading text itself at the top of the chunk so the
                    # title travels with the description
                    course_buf.append(htext)
                else:
                    buf_start_line = i + 1
                i += 1
                continue

            # Pseudo-heading (bold-only line that carries semester/level info
            # but isn't a real markdown heading — common in these handbooks)
            pseudo = parse_pseudo_heading(line)
            if pseudo is not None:
                classified = classify_heading(pseudo)
                # Only let pseudo-headings update state for hierarchy-affecting
                # kinds — random bold labels shouldn't register as sections.
                if classified["kind"] in ("level", "semester"):
                    # Flush any pending narrative so chunks below get the new
                    # semester/level metadata correctly.
                    if in_course is None:
                        yield from flush_narrative(i)
                    state.update(classified)
                    if in_course is None:
                        buf_start_line = i + 1
                    i += 1
                    continue
                # Fall through to normal content handling for non-hierarchy
                # pseudo-headings (like "**Core**", "**Electives**")

            # Regular content line
            if in_course is not None:
                course_buf.append(line)
            else:
                if not narrative_buf:
                    buf_start_line = i + 1
                narrative_buf.append(line)
            i += 1

        # End of file flushes
        yield from flush_narrative(n)
        yield from flush_course(n)

    # ------------------------------------------------------------------ helpers
    def _make_chunk(
        self,
        *,
        text: str,
        content_type: str,
        state: BreadcrumbState,
        source_file: str,
        source_lines: list[int],
        extra_metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        meta = {
            **state.snapshot(),
            "source_file": source_file,
            "source_lines": source_lines,
        }
        if extra_metadata:
            meta.update(extra_metadata)

        # Extract programme type from text
        prog_type = extract_programme_type(text)
        if prog_type:
            meta["programme_type"] = prog_type

        breadcrumb = state.breadcrumb_str()
        # Prepend breadcrumb to the text so embeddings capture location
        prepended = f"[{breadcrumb}]\n\n{text}" if breadcrumb else text
        return Chunk(
            text=prepended,
            content_type=content_type,
            breadcrumb=breadcrumb,
            metadata=meta,
        )

    def _split_narrative(self, body: str) -> list[str]:
        """If a narrative block exceeds MAX_NARRATIVE_CHARS, split on paragraphs."""
        if len(body) <= self.MAX_NARRATIVE_CHARS:
            return [body]
        paras = re.split(r"\n\s*\n", body)
        out: list[str] = []
        cur: list[str] = []
        cur_len = 0
        for p in paras:
            p = p.strip()
            if not p:
                continue
            if cur_len + len(p) > self.MAX_NARRATIVE_CHARS and cur:
                out.append("\n\n".join(cur))
                cur = [p]
                cur_len = len(p)
            else:
                cur.append(p)
                cur_len += len(p) + 2
        if cur:
            out.append("\n\n".join(cur))
        return out


# ---------------------------------------------------------------------------
# Generic heading-aware chunker (fallback)
# ---------------------------------------------------------------------------

class GenericHeadingChunker:
    """
    Walk markdown headings, emit one chunk per section. Split long sections
    on paragraph boundaries. Maintains breadcrumb from heading levels.
    """

    MAX_CHARS = 2000
    MIN_CHARS = 80

    def chunk_file(self, path: str | Path) -> list[Chunk]:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        return list(self._chunk_text(text, source_file=path.name))

    def _chunk_text(self, text: str, source_file: str) -> Iterator[Chunk]:
        lines = preprocess_lines(text)
        stack: list[tuple[int, str]] = []  # (level, text)
        buf: list[str] = []
        buf_start = 0

        # Extract document title from first #/##/### heading (4+ chars), or use filename
        doc_title = Path(source_file).stem
        for ln in lines[:30]:
            m = re.match(r"^#{1,3}\s+(.*?)\s*$", ln)
            if m:
                cand = re.sub(r"[*_`]|</?u>", "", m.group(1)).strip()
                if len(cand) >= 4:
                    doc_title = cand
                    break

        def breadcrumb() -> str:
            parts = [doc_title] + [h for _, h in stack]
            return " > ".join(parts)

        def flush(end_line: int) -> Iterator[Chunk]:
            nonlocal buf, buf_start
            body = "\n".join(buf).strip()
            buf = []
            if len(body) < self.MIN_CHARS:
                return
            pieces = [body] if len(body) <= self.MAX_CHARS else _split_on_paragraphs(
                body, self.MAX_CHARS
            )
            for piece in pieces:
                bc = breadcrumb()
                prepended = f"[{bc}]\n\n{piece}" if bc else piece
                yield Chunk(
                    text=prepended,
                    content_type="section",
                    breadcrumb=bc,
                    metadata={
                        "source_file": source_file,
                        "source_lines": [buf_start, end_line],
                        "heading_path": [h for _, h in stack],
                        "document_title": doc_title,
                        "document_type": "document",
                    },
                )

        for i, line in enumerate(lines):
            h = parse_heading(line)
            if h is not None:
                yield from flush(i)
                level, htext = h
                # Pop any equal-or-deeper headings, then push
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, htext))
                buf_start = i + 1
                continue
            if not buf:
                buf_start = i + 1
            buf.append(line)

        yield from flush(len(lines))


def _split_on_paragraphs(body: str, max_chars: int) -> list[str]:
    paras = re.split(r"\n\s*\n", body)
    out, cur, cur_len = [], [], 0
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if cur_len + len(p) > max_chars and cur:
            out.append("\n\n".join(cur))
            cur, cur_len = [p], len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 2
    if cur:
        out.append("\n\n".join(cur))
    return out

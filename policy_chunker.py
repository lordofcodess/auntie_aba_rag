"""
Chunker for University of Ghana policy / regulation / rules / guidelines /
codes of conduct / charters.

Design (calibrated to actual user queries like "what are the requirements
for becoming a professor", "who is on the Promotions Committee", "what
happens if a student fails to pay fees"):

  * Medium granularity — chunk at 2nd-level sections (e.g. 3.19) so
    multi-clause answers stay together, unless the section is very long
    in which case we split on clause boundaries.
  * Enumeration preservation — never split in the middle of an i/ii/iii
    or (a)(b)(c) sequence.
  * Heading inheritance — prepend full section path to chunk text so
    clauses mentioning "Vice-Chancellor as Chair" retain the context of
    "Promotions Committee Composition" that might only live in a parent
    heading.
  * Document metadata extraction — pull title, document type, publication
    number, volume, year, approving authority from the cover page; attach
    to every chunk for filterable retrieval.
  * Definitions tables — one chunk per term, plus one parent chunk for
    the whole table.
  * Cleanup — strip repeated footers ("UG Risk Management Policy"),
    page numbers, roman-numeral page markers, images, duplicate title
    blocks (common on cover pages).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from bs4 import BeautifulSoup

from chunker import Chunk, preprocess_lines


# ---------------------------------------------------------------------------
# Document metadata extraction
# ---------------------------------------------------------------------------

# Policy cover pages commonly carry:
#   NO. 912   FRIDAY, MAY 24, 2019   VOL. 56 NO. 17
#   POLICY NO. 937 | Volume 58 | Number 4
#   Publication Number 975 | Volume 60 | Number 3
_PUBNUM_RE = re.compile(
    r"(?:POLICY\s+)?(?:NO\.?|NUMBER|PUBLICATION\s+NUMBER)\s*[:\-]?\s*(\d{2,4})",
    re.IGNORECASE,
)
_VOLUME_RE = re.compile(r"(?:VOL\.?|VOLUME)\s*[:\-]?\s*(\d{1,3})", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_AUTHORITY_RE = re.compile(
    r"(?:Published\s+(?:Under\s+the\s+)?Authority\s+of\s+(?:the\s+)?)"
    r"([A-Z][A-Za-z\s]+?)(?:[\.\n]|$)",
    re.IGNORECASE,
)

# Heuristic: document type from filename + first heading
_TYPE_KEYWORDS = [
    ("regulations", "regulations"),
    ("guidelines", "guidelines"),
    ("rules", "rules"),
    ("code_of_conduct", "code_of_conduct"),
    ("code of conduct", "code_of_conduct"),
    ("charter", "charter"),
    ("handbook", "handbook"),
    ("policy", "policy"),
]


def infer_doc_type(filename: str, first_heading: str) -> str:
    hay = (filename + " " + first_heading).lower()
    for needle, label in _TYPE_KEYWORDS:
        if needle in hay:
            return label
    return "document"


def extract_doc_metadata(lines: list[str], filename: str) -> dict[str, Any]:
    """Scan the first ~40 non-blank lines of a doc for cover-page info."""
    window = []
    for ln in lines:
        if ln.strip():
            window.append(ln)
        if len(window) >= 40:
            break
    blob = "\n".join(window)

    # Title = first meaningful # heading that isn't a generic boilerplate.
    # These templates recur across UG policy docs as gazette/cover-page
    # headers and should never be chosen as the document title.
    BOILERPLATE_TITLES = {
        "UNIVERSITY OF GHANA",
        "SPECIAL REPORTER",
        "UNIVERSITY OF GHANA SPECIAL REPORTER",
        "UG SPECIAL REPORTER",
        "PUBLISHED BY AUTHORITY",
        "TABLE OF CONTENTS",
        "CONTENTS",
        "TABLE OF CONTENT",
        "FOREWORD",
        "PREAMBLE",
    }
    title = None
    for ln in window:
        m = re.match(r"^#{1,3}\s+(.*?)\s*$", ln)
        if not m:
            continue
        cand = re.sub(r"[*_`]|</?u>", "", m.group(1)).strip()
        cand = re.sub(r"\s+", " ", cand)
        up = cand.upper().rstrip(".:,")
        if up in BOILERPLATE_TITLES or len(cand) < 4:
            continue
        # Skip titles that start with UG gazette template phrases — the
        # real policy title comes shortly after.
        if up.startswith(("UNIVERSITY OF GHANA SPECIAL",
                          "UG SPECIAL REPORTER")):
            continue
        title = cand
        break

    meta: dict[str, Any] = {
        "document_title": title or Path(filename).stem,
        "document_type": infer_doc_type(filename, title or ""),
        "source_file": filename,
    }

    m = _PUBNUM_RE.search(blob)
    if m:
        meta["publication_number"] = m.group(1)
    m = _VOLUME_RE.search(blob)
    if m:
        meta["volume"] = m.group(1)
    # Year: look for a standalone 4-digit year, prefer the most recent one
    # to avoid grabbing "2010 (Act 806)" style references.
    years = [int(y.group(0)) for y in re.finditer(r"\b(?:19|20)\d{2}\b", blob)]
    if years:
        meta["year"] = max(years)
    m = _AUTHORITY_RE.search(blob)
    if m:
        auth = m.group(1).strip().rstrip(",.")
        if 3 < len(auth) < 80:
            meta["approving_authority"] = auth

    return meta


# ---------------------------------------------------------------------------
# Section numbering
# ---------------------------------------------------------------------------

# Matches section numbers at start of heading text:
#   "1 PURPOSE", "1.0 Purpose", "2.1 Application", "3.19.4 Members"
#   "A. Undergraduate", "B. Postgraduate"
_SECTION_NUM_RE = re.compile(
    r"^\s*(?:"
    r"(\d+(?:\.\d+)*)(?:\.?)"    # 1 or 1.0 or 3.19.4
    r"|"
    r"([A-Z])\.(?!\d)"           # A. Undergraduate (single letter + dot)
    r")\s+(.+?)\s*$"
)


def parse_section_header(heading_text: str) -> dict[str, Any]:
    """
    Return dict with 'number', 'depth', 'title', 'raw' if it's a numbered/
    lettered section, else {'title': heading_text, 'depth': None}.
    """
    m = _SECTION_NUM_RE.match(heading_text)
    if not m:
        return {"title": heading_text.strip(), "depth": None, "raw": heading_text}
    num = m.group(1) or m.group(2)
    title = m.group(3).strip()
    if m.group(1):  # numeric
        depth = len([p for p in num.split(".") if p != "0"]) if num != "0" else 1
        # "1.0" is the same as "1" semantically — depth 1
        parts = num.split(".")
        if len(parts) > 1 and parts[-1] == "0":
            num = ".".join(parts[:-1])
            depth = len(num.split("."))
        else:
            depth = len(parts)
    else:  # lettered
        depth = 1
    return {"number": num, "depth": depth, "title": title, "raw": heading_text}


_HEADING_LINE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_HEADING_INLINE_CLEAN = re.compile(r"[*_`]|</?u>", re.IGNORECASE)


def parse_heading_line(line: str) -> tuple[int, str] | None:
    m = _HEADING_LINE.match(line)
    if not m:
        return None
    hlevel = len(m.group(1))
    text = _HEADING_INLINE_CLEAN.sub("", m.group(2)).strip()
    text = re.sub(r"\s+", " ", text)
    return hlevel, text


# ---------------------------------------------------------------------------
# Enumeration detection
# ---------------------------------------------------------------------------

# Lines that start an enumerated clause:
#   "i. ...", "ii. ...", "iii) ...", "(a) ...", "(i) ...", "1. ...", "2) ..."
_ENUM_MARKERS = [
    re.compile(r"^\s*\(?[ivxlcdm]+[\.\)]\s", re.IGNORECASE),   # i. ii. iii)
    re.compile(r"^\s*\([a-z]\)\s"),                             # (a) (b)
    re.compile(r"^\s*[a-z]\)\s"),                               # a) b)
    re.compile(r"^\s*\d+[\.\)]\s"),                             # 1. 2)
]


def is_enum_start(line: str) -> bool:
    return any(p.match(line) for p in _ENUM_MARKERS)


# ---------------------------------------------------------------------------
# Table handling
# ---------------------------------------------------------------------------

def is_definitions_table(table_html: str) -> bool:
    """Heuristic: 2-column table with headers that look like term/definition."""
    soup = BeautifulSoup(table_html, "html.parser")
    first_row = soup.find("tr")
    if not first_row:
        return False
    headers = [
        h.get_text(" ", strip=True).lower() for h in first_row.find_all(["th", "td"])
    ]
    if len(headers) != 2:
        return False
    left = headers[0]
    right = headers[1]
    term_words = {"term", "word", "word/term", "acronym", "abbreviation"}
    def_words = {"definition", "meaning", "description"}
    return (
        any(w in left for w in term_words)
        and any(w in right for w in def_words)
    )


def linearize_table(table_html: str) -> str:
    soup = BeautifulSoup(table_html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [
            re.sub(r"\s+", " ", c.get_text(" ", strip=True))
            for c in tr.find_all(["th", "td"])
        ]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_definitions(table_html: str) -> list[tuple[str, str]]:
    """Return list of (term, definition) tuples from a definitions table."""
    soup = BeautifulSoup(table_html, "html.parser")
    out = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) != 2:
            continue
        term = re.sub(r"\s+", " ", cells[0].get_text(" ", strip=True))
        definition = re.sub(r"\s+", " ", cells[1].get_text(" ", strip=True))
        # Strip markdown emphasis markers from term
        term = re.sub(r"[*_`]|</?u>", "", term).strip()
        # Skip header row
        if term.lower() in {"word/term", "term", "word", "acronym"}:
            continue
        if term and definition:
            out.append((term, definition))
    return out


# ---------------------------------------------------------------------------
# Repeated footer / boilerplate detection
# ---------------------------------------------------------------------------

def detect_boilerplate_lines(lines: list[str]) -> set[str]:
    """
    Find lines that repeat many times throughout the document — these are
    almost always page footers like "University of Ghana Risk Management
    Policy" that should be stripped.
    """
    from collections import Counter
    trimmed = [re.sub(r"\s+", " ", ln).strip(" *_`") for ln in lines if ln.strip()]
    counts = Counter(trimmed)
    boiler: set[str] = set()
    threshold = max(3, len(lines) // 500)  # scale with doc size
    for text, n in counts.items():
        if n < threshold:
            continue
        # Only treat as boilerplate if it looks like a footer (short, not prose)
        if len(text) > 120:
            continue
        if text.count(" ") > 15:  # too wordy for a footer
            continue
        # Never strip HTML tags — they're needed for table detection
        if re.search(r"</?[a-zA-Z]+[^>]*>", text):
            continue
        boiler.add(text)
    return boiler


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

@dataclass
class _Section:
    """In-flight section accumulator."""
    number: str | None
    depth: int | None
    title: str
    heading_raw: str
    parent_path: list[tuple[str | None, str]]  # [(number, title), ...]
    body_lines: list[str] = field(default_factory=list)
    start_line: int = 0
    has_children: bool = False


class PolicyChunker:
    # Aim for this size on the chunked body, but allow overflow to preserve
    # enumerated clauses.
    TARGET_CHARS = 2500
    MAX_CHARS = 4500  # hard ceiling; we'll split above this
    MIN_CHARS = 80

    def chunk_file(self, path: str | Path) -> list[Chunk]:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        lines = preprocess_lines(text)
        boilerplate = detect_boilerplate_lines(lines)
        lines = [ln for ln in lines if ln.strip() not in boilerplate]
        doc_meta = extract_doc_metadata(lines, path.name)
        return list(self._walk(lines, doc_meta))

    # ------------------------------------------------------------------ walk
    def _walk(
        self, lines: list[str], doc_meta: dict[str, Any]
    ) -> Iterator[Chunk]:
        # Section stack holds (number, title, raw_heading, depth)
        # When we encounter a new heading, pop sections at same-or-deeper depth.
        stack: list[_Section] = []
        i = 0
        n = len(lines)

        def flush_stack_above(new_depth: int) -> Iterator[Chunk]:
            """Close any sections on the stack at depth >= new_depth."""
            while stack and stack[-1].depth is not None and new_depth is not None \
                    and stack[-1].depth >= new_depth:
                yield from self._emit_section(stack.pop(), doc_meta)
            # Also close untitled (depth=None) sections if we're starting
            # a proper numbered section
            while stack and stack[-1].depth is None and new_depth is not None:
                yield from self._emit_section(stack.pop(), doc_meta)

        while i < n:
            line = lines[i]

            # Table handling
            if line.lstrip().startswith("<table"):
                start = i
                buf = [line]
                i += 1
                while i < n and "</table>" not in lines[i]:
                    buf.append(lines[i])
                    i += 1
                if i < n:
                    buf.append(lines[i])
                    i += 1
                table_html = "\n".join(buf)
                # Definitions table → per-term chunks
                if is_definitions_table(table_html):
                    yield from self._emit_definitions(
                        table_html, stack, doc_meta,
                        source_lines=[start + 1, i],
                    )
                else:
                    # Regular table becomes part of current section body,
                    # as linearized text
                    linear = linearize_table(table_html)
                    if stack:
                        stack[-1].body_lines.append(linear)
                    # else: orphan table, skip
                continue

            # Heading line
            h = parse_heading_line(line)
            if h is not None:
                hlevel, htext = h
                parsed = parse_section_header(htext)
                # Skip boilerplate headings
                up = parsed["title"].upper().rstrip(".:,")
                if up in {"UNIVERSITY OF GHANA", "SPECIAL REPORTER",
                          "PUBLISHED BY AUTHORITY", "TABLE OF CONTENTS",
                          "CONTENTS", "TABLE OF CONTENT"}:
                    i += 1
                    continue

                if parsed["depth"] is not None:
                    # Numbered/lettered section
                    new_depth = parsed["depth"]
                    # Flush sections at this depth or deeper
                    yield from flush_stack_above(new_depth)
                    # Mark parent as having children (so we only emit parent
                    # if it has its own body text before the child started)
                    if stack:
                        stack[-1].has_children = True
                        yield from self._emit_section_prelude(stack[-1], doc_meta)
                    parent_path = [
                        (s.number, s.title) for s in stack
                    ]
                    stack.append(_Section(
                        number=parsed["number"],
                        depth=new_depth,
                        title=parsed["title"],
                        heading_raw=htext,
                        parent_path=parent_path,
                        start_line=i + 1,
                    ))
                else:
                    # Unnumbered heading — treat as a shallow section only if
                    # there's no numbered section active, else as body text.
                    # Skip boilerplate cover-page headings and doc title repeats.
                    t_up = parsed["title"].upper().rstrip(".:,")
                    if (
                        t_up == doc_meta["document_title"].upper()
                        or t_up.startswith(("UNIVERSITY OF GHANA SPECIAL",
                                            "UG SPECIAL REPORTER"))
                        or t_up in {
                            "UNIVERSITY OF GHANA",
                            "SPECIAL REPORTER",
                            "PUBLISHED BY AUTHORITY",
                            "FOREWORD",
                        }
                    ):
                        i += 1
                        continue
                    if not any(s.depth is not None for s in stack):
                        yield from flush_stack_above(1)
                        parent_path = [(s.number, s.title) for s in stack]
                        stack.append(_Section(
                            number=None,
                            depth=None,
                            title=parsed["title"],
                            heading_raw=htext,
                            parent_path=parent_path,
                            start_line=i + 1,
                        ))
                    else:
                        # Inside a numbered section — keep as body content
                        stack[-1].body_lines.append(f"**{parsed['title']}**")
                i += 1
                continue

            # Regular content line
            if stack:
                stack[-1].body_lines.append(line)
            # Else: pre-section preamble, dropped for now (usually cover-page
            # fluff we already extracted metadata from).
            i += 1

        # Flush remaining stack
        while stack:
            yield from self._emit_section(stack.pop(), doc_meta)

    # -------------------------------------------------------- emit helpers
    def _emit_section_prelude(
        self, section: _Section, doc_meta: dict
    ) -> Iterator[Chunk]:
        """
        If a section has its own body text BEFORE its first subsection,
        emit that as a separate chunk. Only called once per parent.
        Marked via section.has_children side channel.
        """
        if section.body_lines and getattr(section, "_prelude_emitted", False) is False:
            body = "\n".join(section.body_lines).strip()
            if len(body) >= self.MIN_CHARS:
                yield from self._emit_body_as_chunks(
                    section, body, doc_meta, content_type="section_intro",
                )
            section.body_lines = []
            section._prelude_emitted = True  # type: ignore[attr-defined]

    def _emit_section(
        self, section: _Section, doc_meta: dict
    ) -> Iterator[Chunk]:
        body = "\n".join(section.body_lines).strip()
        if len(body) < self.MIN_CHARS and not section.has_children:
            # Section had neither substantive body nor sub-content. Skip.
            return
        if body:
            yield from self._emit_body_as_chunks(
                section, body, doc_meta, content_type="section",
            )

    def _emit_body_as_chunks(
        self,
        section: _Section,
        body: str,
        doc_meta: dict,
        content_type: str,
    ) -> Iterator[Chunk]:
        pieces = self._split_respecting_enums(body)
        for idx, piece in enumerate(pieces):
            bc, crumbs = self._build_breadcrumb(section, doc_meta)
            inherit = self._build_heading_inheritance(section)
            prefix_parts = []
            if bc:
                prefix_parts.append(f"[{bc}]")
            if inherit:
                prefix_parts.append(inherit)
            prefix = "\n\n".join(prefix_parts)
            text = f"{prefix}\n\n{piece}" if prefix else piece

            meta = dict(doc_meta)
            if section.number:
                meta["section_number"] = section.number
            if section.title:
                meta["section_title"] = section.title
            # Full section path for filtering ("what's in section 3.19?")
            if crumbs:
                meta["section_path"] = crumbs
            meta["content_type"] = content_type
            if len(pieces) > 1:
                meta["chunk_index"] = idx
                meta["chunk_total"] = len(pieces)

            yield Chunk(
                text=text,
                content_type=content_type,
                breadcrumb=bc,
                metadata=meta,
            )

    def _build_breadcrumb(
        self, section: _Section, doc_meta: dict
    ) -> tuple[str, list[str]]:
        parts = [doc_meta["document_title"]]
        crumbs: list[str] = [doc_meta["document_title"]]
        for num, title in section.parent_path:
            label = f"§{num} {title}" if num else title
            parts.append(label)
            crumbs.append(label)
        if section.number or section.title:
            label = (
                f"§{section.number} {section.title}"
                if section.number else section.title
            )
            parts.append(label)
            crumbs.append(label)
        return " > ".join(parts), crumbs

    def _build_heading_inheritance(self, section: _Section) -> str:
        """
        Build a prose context line that repeats parent headings in the chunk
        text so embeddings capture semantic context. E.g. a clause buried
        in §3.19.4 will get the text of §3.19's title baked in.
        """
        parents = [title for _, title in section.parent_path if title]
        if section.title:
            parents.append(section.title)
        if not parents:
            return ""
        if len(parents) == 1:
            return f"(From section: {parents[0]})"
        return f"(Under: {' → '.join(parents)})"

    # ----------------------------------------------- enum-aware splitting
    def _split_respecting_enums(self, body: str) -> list[str]:
        if len(body) <= self.MAX_CHARS:
            return [body]

        # Group lines into "atomic blocks". An atomic block is:
        #   - a clause + all its enumerated sub-items, OR
        #   - a standalone paragraph.
        lines = body.split("\n")
        blocks: list[str] = []
        cur: list[str] = []
        in_enum = False

        def flush():
            if cur:
                blocks.append("\n".join(cur).strip())
                cur.clear()

        for ln in lines:
            stripped = ln.strip()
            if not stripped:
                if not in_enum:
                    flush()
                    cur.append("")
                else:
                    cur.append(ln)
                continue
            if is_enum_start(ln):
                in_enum = True
                cur.append(ln)
            else:
                if in_enum and ln.startswith((" ", "\t")):
                    # Continuation of enum item
                    cur.append(ln)
                else:
                    # Non-enum line breaks the enum
                    if in_enum:
                        in_enum = False
                    cur.append(ln)
        flush()
        blocks = [b for b in blocks if b]

        # Pack blocks greedily up to TARGET_CHARS, but never split a block.
        out: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for b in blocks:
            if buf and buf_len + len(b) > self.TARGET_CHARS:
                out.append("\n\n".join(buf))
                buf = [b]
                buf_len = len(b)
            else:
                buf.append(b)
                buf_len += len(b) + 2
        if buf:
            out.append("\n\n".join(buf))
        return out

    # ------------------------------------------------ definitions emission
    def _emit_definitions(
        self,
        table_html: str,
        stack: list[_Section],
        doc_meta: dict,
        source_lines: list[int],
    ) -> Iterator[Chunk]:
        pairs = extract_definitions(table_html)
        if not pairs:
            return
        # Parent chunk containing the full table (useful for "list all
        # defined terms" queries)
        current = stack[-1] if stack else _Section(
            number=None, depth=None, title="Definitions",
            heading_raw="Definitions", parent_path=[],
        )
        bc, crumbs = self._build_breadcrumb(current, doc_meta)
        inherit = self._build_heading_inheritance(current)
        full_text = "\n".join(f"{term}: {defn}" for term, defn in pairs)
        parent_meta = dict(doc_meta)
        if current.number:
            parent_meta["section_number"] = current.number
        if current.title:
            parent_meta["section_title"] = current.title
        if crumbs:
            parent_meta["section_path"] = crumbs
        parent_meta["content_type"] = "definitions_table"
        parent_meta["definition_count"] = len(pairs)
        parent_meta["source_lines"] = source_lines
        prefix = "\n\n".join(p for p in [f"[{bc}]" if bc else "", inherit] if p)
        yield Chunk(
            text=f"{prefix}\n\nDefinitions in this section:\n{full_text}"
                 if prefix else f"Definitions in this section:\n{full_text}",
            content_type="definitions_table",
            breadcrumb=bc,
            metadata=parent_meta,
        )
        # Per-term chunks
        for term, defn in pairs:
            meta = dict(doc_meta)
            if current.number:
                meta["section_number"] = current.number
            if current.title:
                meta["section_title"] = current.title
            if crumbs:
                meta["section_path"] = crumbs
            meta["content_type"] = "definition"
            meta["term"] = term
            meta["source_lines"] = source_lines
            prefix = "\n\n".join(p for p in [f"[{bc}]" if bc else "", inherit] if p)
            text = (
                f"{prefix}\n\nDefinition of \"{term}\" in "
                f"{doc_meta['document_title']}:\n{defn}"
                if prefix
                else f"Definition of \"{term}\":\n{defn}"
            )
            yield Chunk(
                text=text,
                content_type="definition",
                breadcrumb=bc,
                metadata=meta,
            )

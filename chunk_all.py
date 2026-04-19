"""
Driver: chunk all handbooks, apply template contextualization, write JSONL.

Routes documents:
  - Files whose names match a known handbook → HandbookChunker
  - Everything else → GenericHeadingChunker
  - LLM contextualization is opt-in (needs API key + wiring)

Usage:
    python chunk_all.py <input_file_or_dir> ... --out out_dir/

Or programmatically:
    from chunk_all import process_files
    stats = process_files(["handbook.md"], out_dir="chunks/")
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from chunker import (
    Chunk,
    HandbookChunker,
    GenericHeadingChunker,
    chunks_to_jsonl,
)
from contextualize import template_contextualize

try:
    from policy_chunker import PolicyChunker
except ImportError:
    PolicyChunker = None


# File-name patterns that indicate a UG handbook
HANDBOOK_MARKERS = ("handbook", "cbas", "chs", "humanities")

# File-name patterns that indicate a policy/regulations/rules document
POLICY_MARKERS = (
    "policy", "regulations", "rules", "guidelines", "charter",
    "code_of_conduct", "code of conduct",  # both underscore and space variants
    "handbook_for_heads", "handbook for heads",
    "masters_regulations", "masters regulations",
    "junior_members", "junior members",
    "appeals_board", "appeals board",
    "disciplinary", "disciplinary boards",
    "financial_regulations", "financial regulations",
    "audit_charter", "audit charter",
)


def pick_chunker(path: Path):
    """Route documents to the appropriate chunker.

    Order matters: handbook markers are checked first (they're specific),
    then policy markers, then fall back to generic.
    """
    name = path.name.lower()

    # Check handbook markers first (most specific)
    if any(m in name for m in HANDBOOK_MARKERS):
        return HandbookChunker()

    # Check policy markers
    if any(m in name for m in POLICY_MARKERS):
        if PolicyChunker is None:
            raise ImportError("PolicyChunker not available; install policy_chunker.py")
        return PolicyChunker()

    # Fall back to generic chunker for everything else
    return GenericHeadingChunker()


def process_files(
    paths: list[str | Path],
    out_dir: str | Path,
    contextualize: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for p in paths:
        p = Path(p)
        chunker = pick_chunker(p)
        kind = chunker.__class__.__name__
        chunks = chunker.chunk_file(p)
        if contextualize:
            chunks = [template_contextualize(c) for c in chunks]
        out_file = out_dir / (p.stem + ".chunks.jsonl")
        chunks_to_jsonl(chunks, out_file)

        types = Counter(c.content_type for c in chunks)
        tables = [c for c in chunks if c.content_type == "programme_table"]
        coverage = {}
        if tables:
            coverage = {
                k: sum(1 for c in tables if c.metadata.get(k))
                for k in ("school", "department", "programme", "level", "semester")
            }
        all_stats[p.name] = {
            "chunker": kind,
            "total_chunks": len(chunks),
            "by_type": dict(types),
            "table_metadata_coverage": coverage,
            "table_count": len(tables),
            "out_file": str(out_file),
        }

    return all_stats


def format_stats(stats: dict) -> str:
    lines = []
    for fname, s in stats.items():
        lines.append(f"── {fname} ──")
        lines.append(f"   chunker      : {s['chunker']}")
        lines.append(f"   total chunks : {s['total_chunks']}")
        lines.append(f"   by type      : {s['by_type']}")
        if s["table_count"]:
            cov = s["table_metadata_coverage"]
            tc = s["table_count"]
            cov_str = "  ".join(
                f"{k}={v}/{tc} ({100*v/tc:.0f}%)" for k, v in cov.items()
            )
            lines.append(f"   table cov.   : {cov_str}")
        lines.append(f"   written to   : {s['out_file']}")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Chunk handbooks + generic markdown.")
    ap.add_argument("inputs", nargs="+", help="Markdown file(s) to chunk")
    ap.add_argument("--out", default="chunks_out", help="Output directory")
    ap.add_argument(
        "--no-context",
        action="store_true",
        help="Skip template contextualization",
    )
    args = ap.parse_args(argv)

    paths: list[Path] = []
    for i in args.inputs:
        p = Path(i)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.md")))
        else:
            paths.append(p)

    stats = process_files(paths, out_dir=args.out, contextualize=not args.no_context)
    print(format_stats(stats))


if __name__ == "__main__":
    main()

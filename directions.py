"""Directions support for UG Legon — natural-language step rewriter.

The frontend computes routes via the Google Maps JS DirectionsService (which
gives us authoritative steps and a polyline). It posts the raw step strings
here and we use Gemini to rewrite them as natural conversational sentences,
weaving in recognizable UG landmarks where possible.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Optional

from google import genai


_directions_client: Optional[genai.Client] = None


def _get_client() -> Optional[genai.Client]:
    """Lazily build a genai.Client for step rewriting.

    Prefers GEMINI_API_KEY (the main shared key); falls back to
    MAPS_GEMINI_API_KEY only if no main key is set. Cached after first build.
    """
    global _directions_client
    if _directions_client is not None:
        return _directions_client
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("MAPS_GEMINI_API_KEY")
    if not api_key:
        return None
    _directions_client = genai.Client(api_key=api_key)
    return _directions_client


# Recognizable UG Legon landmarks the rewriter is allowed to mention. Anything
# outside this list must NOT appear in rewritten steps.
UG_LANDMARKS = sorted({
    "Akuafo Hall",
    "African Union Hall",
    "Alexander Kwapong Hall",
    "Balme Library",
    "Bush Canteen",
    "Central Cafeteria",
    "Commonwealth Hall",
    "Department of Computer Science",
    "Elizabeth Sey Hall",
    "Great Hall",
    "Hilla Limann Hall",
    "Jean Nelson Aka Hall",
    "Jones-Quartey Building",
    "Legon Hall",
    "Mensah Sarbah Hall",
    "Night Market",
    "University of Ghana Business School",
    "University of Ghana School of Law",
    "Volta Hall",
})


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of model output (handles stray fences)."""
    if not text:
        return None
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def rewrite_steps(
    raw_steps: list[str],
    *,
    from_name: str,
    to_name: str,
    distance_label: str,
    duration_minutes: int,
    model: str = "gemini-2.5-flash",
) -> list[str]:
    """Rewrite raw walking step strings as natural conversational sentences.

    Returns the rewritten list, or the input list unchanged on any failure.
    Length is preserved (one rewrite per input step).
    """
    if not raw_steps:
        return []

    client = _get_client()
    if client is None:
        return raw_steps

    landmarks_block = "\n".join(f"  - {name}" for name in UG_LANDMARKS)
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(raw_steps))

    prompt = (
        "Rewrite these walking directions as natural, friendly, conversational "
        "sentences for a University of Ghana, Legon student who navigates by "
        "landmarks (halls, departments, the bookshop, the Night Market) more "
        "than by road names.\n\n"
        "Output STRICT JSON only — no markdown, no code fences, no commentary "
        "— in this shape:\n"
        '{"steps": ["sentence 1", "sentence 2", ...]}\n\n'
        "Hard rules:\n"
        "- Output exactly one sentence per input step, same order, same count.\n"
        "- Keep ALL road names and distances exactly as given. Do NOT change, "
        "approximate, drop, or rename them. Do NOT change a number.\n"
        "- Never invent landmarks, halls, or buildings that aren't in the "
        "allowed list below.\n\n"
        "Landmark guidance:\n"
        "- Where you are CONFIDENT a step passes near or arrives at one of the "
        "allowed UG landmarks below, weave it into the sentence as a "
        "recognition cue. Examples: 'Turn right onto J.K.M. Hodasi Rd and walk "
        "for 638 m — you'll pass Akuafo Hall on your right.'\n"
        "- The final step should mention the destination by name.\n"
        "- If you're unsure whether a landmark is along a step, do NOT add it. "
        "Plain road + distance is fine.\n\n"
        "Allowed UG landmarks (use only these names; never invent others):\n"
        f"{landmarks_block}\n\n"
        "Style:\n"
        "- Sound like a friend giving directions in plain English.\n"
        "- Vary phrasing across sentences ('Head down…', 'Continue along…', "
        "'Take a quick right onto…', 'Walk straight past…').\n"
        "- Light connectors at the start of later steps are fine ('Then…', "
        "'Next…', 'After that…'); don't overdo it.\n\n"
        f"Origin: {from_name}. Destination: {to_name}. Total walk: "
        f"{distance_label}, ~{duration_minutes} min.\n\n"
        f"Raw steps:\n{numbered}"
    )

    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        parsed = _extract_json(resp.text or "")
        if not parsed:
            return raw_steps
        steps = parsed.get("steps")
        if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
            return raw_steps
        if len(steps) != len(raw_steps):
            return raw_steps
        cleaned = [s.strip() for s in steps if s.strip()]
        return cleaned if cleaned else raw_steps
    except Exception as e:
        print(f"⚠️  rewrite_steps failed: {type(e).__name__}: {e}", file=sys.stderr)
        return raw_steps

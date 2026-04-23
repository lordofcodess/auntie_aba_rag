"""
Transcript analysis: extract courses from PDF/image, advise on graduation + GPA.

Two Gemini calls:
  1. Extract — multimodal: read the transcript, return structured JSON
  2. Advise — text: combine extracted data with handbook context, return advice
"""

import json
import re
from typing import Optional

from google import genai
from google.genai import types as genai_types


EXTRACTION_PROMPT = """You are an expert at reading University of Ghana academic transcripts.

Extract the following from the attached transcript and return ONLY a JSON object (no prose, no markdown fences):

{
  "student_name": "full name or null",
  "student_id": "ID/registration number or null",
  "programme": "e.g. 'BSc Computer Science' or null",
  "department": "e.g. 'DEPARTMENT OF COMPUTER SCIENCE' or null",
  "current_level": "100, 200, 300, or 400 — the level the student is currently at or most recently completed",
  "cumulative_gpa": "CGPA as a float, or null",
  "total_credits_earned": "integer or null",
  "courses": [
    {
      "code": "e.g. CSCD 301",
      "title": "course title if visible",
      "credits": "integer",
      "grade": "e.g. A, B+, C, F",
      "level": "100/200/300/400 inferred from course code",
      "semester": "1 or 2 if determinable, else null"
    }
  ]
}

Rules:
- If a field is not visible, use null — do not guess.
- Include every course you can see, even those in progress or failed.
- Infer `level` from the middle digit of the course code (e.g. CSCD 301 → 300-level).
- Return ONLY the JSON, nothing else."""


ADVICE_PROMPT_TEMPLATE = """You are an academic advisor for University of Ghana students.

A student has submitted their transcript. Below is (1) the extracted transcript data,
(2) relevant handbook context about their programme, graduation requirements, and
University of Ghana regulations on classification, and (3) optional notes from the
student providing extra context or specific questions.

Produce advice covering ALL THREE of the following areas:

1. **What's Left to Graduate** — list outstanding required/core courses and required credit counts.
   Use the handbook context to know what is required. If the student has taken a course,
   mark it done. If not, list it as outstanding.

2. **Elective Recommendations** — based on what the student has taken and what's left,
   recommend 2–4 specific electives at their current or next level, with brief reasons
   (e.g. aligns with a focus area, prerequisite met, complements existing strengths).
   If the student's notes mention a focus area (e.g. ML, cybersecurity, grad school),
   weight your recommendations toward that focus.

3. **Class Standing / GPA Outlook** — given the cumulative GPA and UG's classification
   bands (First Class, Second Class Upper, Second Class Lower, Third Class, Pass),
   state the current standing and what target GPA is needed to move up or stay in range.
   If CGPA is missing, say so explicitly.

After the three sections, if the student's notes contain a specific question, answer it
in a final **Student's Question** section. If there are no notes or no question, skip
that section.

Format your response with those headings as Markdown sections. Be specific —
cite course codes and credit counts. If the handbook context is insufficient for a
section, say so directly rather than guessing.

=== TRANSCRIPT DATA ===
{transcript_json}

=== HANDBOOK CONTEXT ===
{handbook_context}

=== STUDENT NOTES / QUESTION ===
{student_notes}

=== INSTRUCTION ===
Please analyze this transcript and give me advice covering all three areas above,
taking the student's notes into account where relevant."""


def extract_transcript(client: genai.Client, file_bytes: bytes, mime_type: str, model: str) -> dict:
    """Call Gemini with the attached file, parse JSON response."""
    uploaded = genai_types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
    response = client.models.generate_content(
        model=model,
        contents=[uploaded, EXTRACTION_PROMPT],
    )
    text = response.text.strip()

    # Strip possible markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini did not return valid JSON: {e}\nRaw output:\n{text[:500]}")


def build_handbook_queries(extracted: dict, notes: Optional[str] = None) -> list[str]:
    """Produce a small set of targeted queries against the RAG for handbook context."""
    queries = []
    programme = extracted.get("programme")
    dept = extracted.get("department")
    level = extracted.get("current_level")

    subject = programme or dept or "the student's programme"

    queries.append(f"Graduation requirements for {subject}")
    if level:
        queries.append(f"Level {level} core courses for {subject}")
        try:
            lv = int(level)
            next_level = min(lv + 100, 400)
            queries.append(f"Level {next_level} electives for {subject}")
        except (ValueError, TypeError):
            pass
    queries.append("Classification bands First Class Second Class Third Class Pass GPA requirements")
    queries.append("Bachelor degree credits required for graduation University of Ghana")

    # Pull an extra query driven by the student's free-text notes
    if notes and notes.strip():
        queries.append(notes.strip())

    return queries


def gather_handbook_context(
    rag,
    extracted: dict,
    notes: Optional[str] = None,
    per_query_k: int = 4,
) -> str:
    """Run several retrieval queries and concatenate their results."""
    queries = build_handbook_queries(extracted, notes=notes)
    seen_texts = set()
    blocks = []
    for q in queries:
        chunks = rag.retrieve(q, top_k=per_query_k)
        for c in chunks:
            text = c["text"]
            if text in seen_texts:
                continue
            seen_texts.add(text)
            meta = c["metadata"]
            source = meta.get("source_file", "Unknown")
            blocks.append(f"[FROM: {source}]\n{text}")
    return "\n\n---\n\n".join(blocks[:30])


def advise(
    client: genai.Client,
    rag,
    file_bytes: bytes,
    mime_type: str,
    notes: Optional[str] = None,
    model: str = "gemini-2.5-flash",
) -> dict:
    """End-to-end: extract → retrieve handbook context → generate advice."""
    extracted = extract_transcript(client, file_bytes, mime_type, model)
    context = gather_handbook_context(rag, extracted, notes=notes)

    student_notes = (notes or "").strip() or "(no additional notes provided)"

    prompt = ADVICE_PROMPT_TEMPLATE.format(
        transcript_json=json.dumps(extracted, indent=2),
        handbook_context=context or "(no relevant handbook context retrieved)",
        student_notes=student_notes,
    )

    response = client.models.generate_content(model=model, contents=prompt)

    return {
        "extracted": extracted,
        "notes": notes,
        "advice": response.text,
        "handbook_chunks_used": len(context.split("---")) if context else 0,
    }

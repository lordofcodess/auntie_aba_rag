"""CV / résumé analysis: extract structured info, advise on career fit + UG opportunities.

Two Gemini calls:
  1. Extract — multimodal: read the CV/PDF/image, return structured JSON
  2. Advise — text: combine extracted data with UG handbook context, return advice

Output mirrors transcript.advise() shape (extracted, notes, advice, handbook_chunks_used)
so the frontend can render both via the same upload pipeline.
"""

import datetime
import json
import re
from typing import Optional

from google import genai
from google.genai import types as genai_types


EXTRACTION_PROMPT = """You are an expert at reading CVs / résumés.

Extract the following from the attached CV and return ONLY a JSON object (no prose, no markdown fences):

{
  "full_name": "string or null",
  "headline": "short tagline if visible (e.g. 'Final-year CS student seeking SWE internship') or null",
  "contact": {
    "email": "string or null",
    "phone": "string or null",
    "location": "city, country or null",
    "links": ["url", "..."]
  },
  "education": [
    {
      "institution": "school/university name",
      "qualification": "e.g. 'BSc Computer Science', 'WASSCE'",
      "field": "subject area or null",
      "start": "year or null",
      "end": "year or 'present' or null",
      "gpa_or_grade": "string or null"
    }
  ],
  "experience": [
    {
      "title": "job title",
      "organization": "employer/organization",
      "location": "city or null",
      "start": "month/year or year",
      "end": "month/year, year, or 'present'",
      "summary": "1-2 sentence bullet summary"
    }
  ],
  "skills": {
    "technical": ["..."],
    "tools_and_languages": ["..."],
    "soft": ["..."]
  },
  "projects": [
    {"name": "project name", "summary": "short description", "tech": ["..."]}
  ],
  "publications_or_research": ["..."],
  "certifications": ["..."],
  "languages": ["..."],
  "awards": ["..."],
  "inferred_field": "e.g. 'software engineering', 'public health', 'agribusiness' — your best guess at the candidate's primary field based on the whole CV",
  "experience_level": "one of: 'student', 'entry', 'mid', 'senior' — best guess from years and titles",
  "audience_hint": "one of: 'student' | 'staff' | null — set to 'staff' if the CV shows academic-staff signals (Lecturer/Senior Lecturer/Associate Professor/Professor titles, peer-reviewed publications, supervision of students, teaching positions at a university, research grants). Set to 'student' if it's a student or early-career CV without those signals. null if genuinely ambiguous.",
  "current_academic_rank": "for staff CVs, the most recent academic title (e.g. 'Lecturer', 'Senior Lecturer', 'Associate Professor', 'Professor'). null if not a staff CV or not visible.",
  "publication_count_visible": "integer count of peer-reviewed publications/papers visible on the CV, or null if none / unclear"
}

Rules:
- If a field is not visible, use null (or [] for list fields) — do not guess.
- For dates, preserve what's on the CV (e.g. 'Jun 2024', '2023-2025'). Don't normalize.
- `inferred_field` and `experience_level` are your best inference; explain nothing.
- Return ONLY the JSON, nothing else."""


ADVICE_PROMPT_TEMPLATE = """You are an advisor for University of Ghana students AND academic staff.

TODAY'S DATE: {today}

Use this date when judging whether entries on the CV are past, current, or
future. Do NOT assume your training cutoff is "now" — anything dated on or
before {today} has already happened. Only flag a date as future-dated if it
is genuinely after {today}.

A user has submitted a CV. Below is (1) the structured CV data (note the
`audience_hint` field — "staff" means a UG academic-staff CV, "student" means
a student/early-career CV), (2) relevant UG handbook context, and (3) optional
notes from the user.

Adapt your advice to the audience:

- For **student / early-career CVs**, focus sections 4–5 on jobs and further
  study at UG (internships, entry-level roles, MPhil/PhD admission fit,
  scholarship opportunities).

- For **UG academic-staff CVs** (lecturers, senior lecturers, associate or
  full professors), focus sections 4–5 on academic career progression:
  promotion readiness, research portfolio positioning, grant/fellowship
  opportunities, sabbatical and leave eligibility, supervision and service
  expectations. Reference UG's promotion criteria from the handbook context.

Produce advice covering ALL of the following sections, each as a clear
Markdown heading. Be specific — reference real roles, programmes, ranks,
publication counts, or course codes. If the CV is silent on something you'd
normally mention, say so directly rather than guessing.

1. **CV Snapshot** — one sentence summary (field, audience, standout strengths).

2. **CV Strengths** — 2–4 bullets, concrete to what's on the CV.

3. **CV Weaknesses & Quick Fixes** — 3–5 bullets, each with ONE actionable fix.
   For staff CVs, weigh things like publication venue mix, citation visibility,
   teaching evidence, grant attribution, and supervision counts. For student
   CVs, weigh quantified bullets, projects section, relevant coursework.

4. **Career / Role Fit** (audience-dependent):
   - Student: 2–4 specific role categories the candidate is competitive for
     (e.g. "junior software engineer", "research assistant in public health"),
     plus a brief gap analysis.
   - Staff: promotion readiness for the next rank (based on the handbook
     promotion criteria in context), key strengths vs. typical gaps, and any
     visible deficits that would slow promotion. Name the next rank explicitly.

5. **UG Programme / Opportunity Fit** (audience-dependent):
   - Student: 1–3 UG programmes, departments, or schools that match for further
     study, research, or roles. Use actual names from the handbook context —
     do not invent any.
   - Staff: 1–3 specific UG opportunities relevant now — sabbatical eligibility,
     grant/fellowship fit, relevant research centres or institutes for
     collaboration, leadership/service positions worth targeting. Cite the
     handbook context where applicable.

6. **Targeted Next Steps** — 3 prioritized actions ranked by impact, written
   for the right audience. For staff: things like "submit a publication to
   <venue>", "apply for the X fellowship by <month>", "formalize supervision
   of an MPhil student to strengthen the promotion file." For students: things
   like "add a Projects section", "apply to MPhil <programme>", "earn one
   industry-recognized cert by <month>."

If the user's notes contain a specific question, answer it in a final
**Your Question** section. If not, skip that section.

=== CV DATA ===
{cv_json}

=== UG HANDBOOK CONTEXT ===
{handbook_context}

=== USER NOTES / QUESTION ===
{user_notes}

=== INSTRUCTION ===
Please analyse this CV and give me advice covering all six sections above,
taking the user's notes and the audience (student vs. staff) into account.
Format as Markdown."""


# Used when the user provided notes alongside the upload. Their notes ARE the
# primary intent — we answer them directly instead of running the full
# six-section default analysis.
TARGETED_PROMPT_TEMPLATE = """You are an advisor for University of Ghana students AND academic staff.

TODAY'S DATE: {today}

Use this date when judging whether entries on the CV are past, current, or
future. Do NOT assume your training cutoff is "now" — anything dated on or
before {today} has already happened.

The user uploaded a CV AND provided notes. Their notes ARE the primary intent.

Decision protocol:
- If the notes are a direct question (e.g. "What's my biggest weakness?",
  "Am I ready for a SWE internship?", "What programmes match my background?"),
  answer it directly using the CV data + any relevant UG handbook context.
  Do NOT run the full six-section CV review.
- If the notes are context or preferences (e.g. "I'm targeting grad school",
  "I want a tech internship", "I'm applying for promotion to Associate
  Professor"), focus your answer on what they're targeting. Cover only the
  sections that are clearly relevant to that goal. Be concise.

Adapt to the audience (`audience_hint` in the CV data: "staff" = UG academic
staff CV, "student" = student/early-career CV). For staff, frame around
academic career progression (promotion, grants, sabbaticals). For students,
frame around jobs, internships, further study.

=== CV DATA ===
{cv_json}

=== UG HANDBOOK CONTEXT ===
{handbook_context}

=== USER NOTES / QUESTION ===
{user_notes}

=== INSTRUCTION ===
Answer the user's actual question or focus on their stated intent. Be specific
— cite real CV details (role titles, employer names, education entries,
projects, publication counts). Keep it focused — do not pad with the default
six-section report.

If you genuinely think a full CV review would benefit the user beyond what
they asked, add a single italic line at the very end:
*Want a full CV review? Just say "review my CV".*

Format as Markdown. Skip preamble — answer directly."""


def extract_cv(client: genai.Client, file_bytes: bytes, mime_type: str, model: str) -> dict:
    """Call Gemini with the attached file, parse JSON response."""
    uploaded = genai_types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
    response = client.models.generate_content(
        model=model,
        contents=[uploaded, EXTRACTION_PROMPT],
    )
    text = (response.text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini did not return valid JSON for CV: {e}\nRaw output:\n{text[:500]}")


def _is_staff(extracted: dict) -> bool:
    """Decide whether the CV is from UG academic staff.

    Trusts the extractor's `audience_hint` first, then falls back to checking
    for staff-shaped signals (academic rank, publications, lecturer titles).
    """
    hint = (extracted.get("audience_hint") or "").lower().strip()
    if hint == "staff":
        return True
    if hint == "student":
        return False
    # Fallback heuristics
    if extracted.get("current_academic_rank"):
        return True
    pub_count = extracted.get("publication_count_visible")
    if isinstance(pub_count, int) and pub_count >= 3:
        return True
    titles = " ".join(
        (e.get("title") or "").lower() for e in (extracted.get("experience") or [])
    )
    staff_words = ("lecturer", "professor", "research fellow", "head of department", "dean")
    return any(w in titles for w in staff_words)


def build_handbook_queries(extracted: dict, notes: Optional[str] = None) -> list[str]:
    """Targeted queries to surface UG context that matches the CV (audience-aware)."""
    queries: list[str] = []
    field = extracted.get("inferred_field")
    level = extracted.get("experience_level")
    staff = _is_staff(extracted)

    if staff:
        # Faculty / academic staff context: promotion, research, leave, supervision.
        queries.append("Criteria for promotion to Senior Lecturer Associate Professor Professor University of Ghana")
        queries.append("Conditions of service academic staff teaching research publication requirements")
        queries.append("Sabbatical leave and study leave policy academic staff")
        queries.append("Research policy grants funding University of Ghana")
        queries.append("Supervision of graduate students MPhil PhD regulations")
        rank = extracted.get("current_academic_rank")
        if rank:
            queries.append(f"Promotion criteria from {rank} to the next rank")
        if field:
            queries.append(f"{field} department research focus areas University of Ghana")
    else:
        # Student / early-career: programmes, admissions, electives.
        if field:
            queries.append(f"UG programmes related to {field}")
            queries.append(f"{field} department University of Ghana")
        for edu in (extracted.get("education") or [])[:2]:
            q = (edu.get("qualification") or "").strip()
            f = (edu.get("field") or "").strip()
            if q and f:
                queries.append(f"{q} {f} University of Ghana")
            elif q:
                queries.append(f"{q} University of Ghana")
        if level in ("entry", "mid", "senior"):
            queries.append("Postgraduate masters MPhil PhD programmes admission requirements")
        else:
            queries.append("Undergraduate admission requirements programme structure")

    if notes and notes.strip():
        queries.append(notes.strip())

    # Dedup while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped


def gather_handbook_context(
    rag,
    extracted: dict,
    notes: Optional[str] = None,
    per_query_k: int = 4,
) -> tuple[str, int]:
    """Run a few retrieval queries; return concatenated text + chunk count."""
    queries = build_handbook_queries(extracted, notes=notes)
    seen_texts: set[str] = set()
    blocks: list[str] = []
    for q in queries:
        try:
            chunks = rag.retrieve(q, top_k=per_query_k)
        except Exception:
            continue
        for c in chunks:
            text = c["text"]
            if text in seen_texts:
                continue
            seen_texts.add(text)
            meta = c["metadata"]
            source = meta.get("source_file", "Unknown")
            blocks.append(f"[FROM: {source}]\n{text}")
    capped = blocks[:30]
    return "\n\n---\n\n".join(capped), len(capped)


def advise(
    client: genai.Client,
    rag,
    file_bytes: bytes,
    mime_type: str,
    notes: Optional[str] = None,
    model: str = "gemini-2.5-flash",
) -> dict:
    """End-to-end: extract → retrieve handbook context → generate advice.

    If `notes` is non-empty, the user's question/intent drives the response
    (TARGETED template). If no notes are provided, the default six-section
    CV review runs.
    """
    extracted = extract_cv(client, file_bytes, mime_type, model)
    context, chunks_used = gather_handbook_context(rag, extracted, notes=notes)

    notes_stripped = (notes or "").strip()
    has_notes = bool(notes_stripped)
    user_notes = notes_stripped or "(no additional notes provided)"

    template = TARGETED_PROMPT_TEMPLATE if has_notes else ADVICE_PROMPT_TEMPLATE
    prompt = template.format(
        today=datetime.date.today().isoformat(),
        cv_json=json.dumps(extracted, indent=2),
        handbook_context=context or "(no relevant handbook context retrieved)",
        user_notes=user_notes,
    )

    response = client.models.generate_content(model=model, contents=prompt)

    return {
        "extracted": extracted,
        "notes": notes,
        "advice": response.text or "",
        "handbook_chunks_used": chunks_used,
    }

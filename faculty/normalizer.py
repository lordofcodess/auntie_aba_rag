"""
faculty/normalizer.py — Clean and standardize raw scraped faculty records.

Run:
    python -m faculty.normalizer
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

RAW_PATH  = Path("data/processed_faculty/faculty_raw.json")
OUT_PATH  = Path("data/processed_faculty/faculty_normalized.json")

# ---------------------------------------------------------------------------
# Department name standardization
# ---------------------------------------------------------------------------

DEPT_ALIASES: dict[str, str] = {
    # Computer Science
    "dept. of computer science":        "Department of Computer Science",
    "department of computer science":   "Department of Computer Science",
    "computer science":                 "Department of Computer Science",
    "dcs":                              "Department of Computer Science",
    # Law
    "school of law":                    "School of Law",
    "faculty of law":                   "School of Law",
    "law":                              "School of Law",
    # Business
    "ugbs":                             "University of Ghana Business School",
    "ug business school":               "University of Ghana Business School",
    "business school":                  "University of Ghana Business School",
    # Political Science
    "dept. of political science":       "Department of Political Science",
    "political science":                "Department of Political Science",
    # Education
    "esl":                              "Department of Education Studies and Leadership",
    "education studies and leadership": "Department of Education Studies and Leadership",
    # PAD
    "pad":                              "Department of Public Administration and Health Services Management",
    # COHOLD
    "cohold":                           "College of Health Sciences",
}

COLLEGE_ALIASES: dict[str, str] = {
    "cbas":                             "College of Basic and Applied Sciences",
    "college of basic and applied sciences": "College of Basic and Applied Sciences",
    "college of humanities":            "College of Humanities",
    "humanities":                       "College of Humanities",
    "college of health sciences":       "College of Health Sciences",
    "health sciences":                  "College of Health Sciences",
}

RANK_TOKENS = [
    "Professor",
    "Associate Professor",
    "Assistant Professor",
    "Senior Lecturer",
    "Lecturer",
    "Senior Research Fellow",
    "Research Fellow",
    "Visiting Scholar",
    "Emeritus Professor",
]

ADMIN_TITLE_SIGNALS = [
    "registrar", "administrative assistant", "assistant registrar",
    "public relations", "documentation officer", "manager", "secretary",
    "coordinator", "finance officer", "hr officer", "accountant",
]

BAD_NAMES = [
    "overview", "contact us", "message from", "university leadership",
    "main navigation", "e-news", "biography", "welcome", "home",
    "administration", "leadership",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_str(value) -> str | None:
    if not value: return None
    return re.sub(r"\s+", " ", str(value)).strip() or None


def _normalize_email(email: str | None) -> str | None:
    if not email: return None
    email = email.strip().lower()
    # Remove common obfuscation
    email = email.replace(" [at] ", "@").replace(" at ", "@").replace("[dot]", ".")
    if re.match(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", email):
        return email
    return None


def _normalize_phone(phone: str | None) -> str | None:
    if not phone: return None
    # Keep only digits, spaces, +, -, ()
    cleaned = re.sub(r"[^\d\s\+\-\(\)]", "", phone).strip()
    return cleaned if len(re.sub(r"\D", "", cleaned)) >= 7 else None


def _normalize_dept(dept: str | None) -> str | None:
    if not dept: return None
    key = dept.strip().lower()
    return DEPT_ALIASES.get(key, _clean_str(dept))


def _normalize_college(college: str | None) -> str | None:
    if not college: return None
    key = college.strip().lower()
    return COLLEGE_ALIASES.get(key, _clean_str(college))


def _extract_rank(title: str | None, existing_rank: str | None) -> str | None:
    if existing_rank: return _clean_str(existing_rank)
    if not title: return None
    for rank in RANK_TOKENS:
        if rank.lower() in title.lower():
            return rank
    return None


def _clean_list(items) -> list[str]:
    if not items: return []
    return [re.sub(r"\s+", " ", str(x)).strip() for x in items if str(x).strip()]


def _is_bad_name(name: str | None) -> bool:
    if not name: return True
    nl = name.lower().strip()
    if any(nl.startswith(b) for b in BAD_NAMES): return True
    if len(name.split()) < 2: return True   # single-word names are usually not people
    return False


def _is_admin_staff(record: dict) -> bool:
    title = (record.get("title") or "").lower()
    return any(sig in title for sig in ADMIN_TITLE_SIGNALS)


def _classify_content_type(record: dict) -> str:
    """
    Classify each record so the chunker and retriever can prioritize correctly.
    - faculty_profile:    academic staff (lecturers, professors, researchers)
    - admin_staff_profile: administrative / support staff
    - staff_profile:      general staff where role is unclear
    """
    rank  = record.get("academic_rank") or ""
    title = record.get("title") or ""
    combined = (rank + " " + title).lower()

    if any(token.lower() in combined for token in RANK_TOKENS):
        return "faculty_profile"
    if _is_admin_staff(record):
        return "admin_staff_profile"
    return "staff_profile"


# ---------------------------------------------------------------------------
# Main normalizer
# ---------------------------------------------------------------------------

def normalize(records: list[dict]) -> list[dict]:
    normalized = []

    for raw in records:
        name = _clean_str(raw.get("name"))

        # Drop bad names early
        if _is_bad_name(name):
            logger.debug("Dropping bad name: %s", name)
            continue

        # Drop records from event/news URLs
        url = raw.get("profile_url") or ""
        if any(p in url.lower() for p in ["/events/", "/news/", "/index.php/events/"]):
            logger.debug("Dropping event/news URL: %s", url)
            continue

        title       = _clean_str(raw.get("title"))
        rank        = _extract_rank(title, raw.get("academic_rank"))
        dept        = _normalize_dept(raw.get("department"))
        college     = _normalize_college(raw.get("college"))
        email       = _normalize_email(raw.get("email"))
        phone       = _normalize_phone(raw.get("phone"))
        office      = _clean_str(raw.get("office"))
        bio         = _clean_str(raw.get("biography_summary"))
        research    = _clean_list(raw.get("research_interests", []))
        courses     = _clean_list(raw.get("courses_taught", []))
        pubs        = _clean_list(raw.get("publication_links", []))
        scholar     = _clean_str(raw.get("google_scholar_url"))
        orcid       = _clean_str(raw.get("orcid_url"))
        website     = _clean_str(raw.get("personal_website"))
        profile_url = _clean_str(raw.get("profile_url"))
        image_url   = _clean_str(raw.get("image_url"))

        record = {
            "name":               name,
            "title":              title,
            "academic_rank":      rank,
            "department":         dept,
            "college":            college,
            "email":              email,
            "phone":              phone,
            "office":             office,
            "biography_summary":  bio,
            "research_interests": research,
            "courses_taught":     courses,
            "publication_links":  pubs,
            "google_scholar_url": scholar,
            "orcid_url":          orcid,
            "personal_website":   website,
            "profile_url":        profile_url,
            "image_url":          image_url,
            "content_type":       _classify_content_type({
                                      "title": title, "academic_rank": rank
                                  }),
        }

        # Must have at least a name + one identifying field
        if not any([email, title, dept, college, rank]):
            logger.debug("Dropping record with no identifying fields: %s", name)
            continue

        normalized.append(record)

    logger.info("Normalized %d -> %d records", len(records), len(normalized))
    return normalized


def run():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw faculty file not found: {RAW_PATH}\nRun scraper first.")

    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d raw records from %s", len(raw), RAW_PATH)

    normalized = normalize(raw)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved %d normalized records to %s", len(normalized), OUT_PATH)
    return normalized


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    records = run()
    print(f"\nDone. {len(records)} normalized records saved to {OUT_PATH}")

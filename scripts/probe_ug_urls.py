"""Probe UG URLs via Gemini URL Context to see which return real content.

Usage:
    GEMINI_API_KEY=... python scripts/probe_ug_urls.py
"""

import os
import sys
import time

from google import genai
from google.genai import types

URLS_TO_PROBE = [
    # User-supplied targeted faculty/staff URLs
    "https://www.ug.edu.gh/about-ug/leadership",
    "https://cbas.ug.edu.gh/staff?tab=management",
    "https://dcs.ug.edu.gh/faculty?group=6664788d33d7c32069301986",
    "https://ugbs.ug.edu.gh/faculty",
    "https://www.ug.edu.gh/esl/people/faculty",
    "https://law.ug.edu.gh/faculty",
    "https://pscience.ug.edu.gh/faculty-staff",
    "https://cohold.ug.edu.gh/staff",
    "https://www.ug.edu.gh/pad/staff",
    # User-supplied discovery seeds
    "https://www.ug.edu.gh/staff-faculty-services",
    "https://www.ug.edu.gh/academics/colleges-schools",
    # Subdomain candidates for currently JS-rendered www.ug.edu.gh/<dept> entries
    "https://chemistry.ug.edu.gh",
    "https://chem.ug.edu.gh",
    "https://math.ug.edu.gh",
    "https://mathematics.ug.edu.gh",
    "https://physics.ug.edu.gh",
    "https://stat.ug.edu.gh",
    "https://statistics.ug.edu.gh",
    "https://earthscience.ug.edu.gh",
    "https://earth.ug.edu.gh",
    "https://dmse.ug.edu.gh",
    # Other commonly-needed (already in list — confirms still working)
    "https://dcs.ug.edu.gh/",
]

PROMPT = (
    "Read the page at {url} using the URL Context tool. Then output exactly ONE line in this strict format:\n\n"
    "STATUS=<RICH|THIN|EMPTY> | CHARS=<integer> | SNIPPET=<first ~80 chars of meaningful body text>\n\n"
    "Rules:\n"
    "- RICH = page returned substantive content (multiple paragraphs, lists, names, real info).\n"
    "- THIN = page returned only a title, a navbar, or a sentence or two — looks like a JS shell.\n"
    "- EMPTY = page returned nothing usable / fetch failed.\n"
    "Output ONLY that one line. No prose, no explanation."
)


def probe(client: genai.Client, url: str) -> str:
    cfg = types.GenerateContentConfig(
        tools=[types.Tool(url_context=types.UrlContext())],
    )
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=PROMPT.format(url=url),
            config=cfg,
        )
        return (resp.text or "").strip().splitlines()[0] if resp.text else "EMPTY=true | no response"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def main() -> int:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        return 1
    client = genai.Client(api_key=key)
    print(f"Probing {len(URLS_TO_PROBE)} URLs via Gemini URL Context...\n")
    for url in URLS_TO_PROBE:
        result = probe(client, url)
        print(f"{url}\n  → {result}\n")
        time.sleep(1.0)  # gentle pacing
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
faculty/scraper.py — Offline UG faculty scraper.

Scrapes all departments in SEED_URLS, saves raw HTML and structured JSON.

Run:
    python -m faculty.scraper
"""

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed URLs — all UG departments
# ---------------------------------------------------------------------------

DISCOVERY_SEED_URLS = [
    "https://www.ug.edu.gh/staff-faculty-services",
    "https://www.ug.edu.gh/academics/colleges-schools",
    "https://old1.ug.edu.gh/departments",
]

TARGETED_SEED_URLS = [
    "https://www.ug.edu.gh/about-ug/leadership",
    "https://cbas.ug.edu.gh/staff?tab=management",
    "https://dcs.ug.edu.gh/faculty?group=6664788d33d7c32069301986",
    "https://ugbs.ug.edu.gh/faculty",
    "https://www.ug.edu.gh/esl/people/faculty",
    "https://law.ug.edu.gh/faculty",
    "https://pscience.ug.edu.gh/faculty-staff",
    "https://cohold.ug.edu.gh/staff",
    "https://www.ug.edu.gh/pad/staff",
]

SEED_URLS = list(dict.fromkeys(DISCOVERY_SEED_URLS + TARGETED_SEED_URLS))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR       = Path("data/raw_faculty")
PROCESSED_DIR = Path("data/processed_faculty")

REQUEST_DELAY          = 1.0
MAX_LISTING_PAGES      = 120
PLAYWRIGHT_TIMEOUT_MS  = 15_000
PLAYWRIGHT_SETTLE_MS   = 2_000

JS_RENDERED_DOMAINS = {"dcs.ug.edu.gh", "cbas.ug.edu.gh", "ugbs.ug.edu.gh"}

ALLOWED_QUERY_PARAMS = {"tab", "group", "page", "p", "pg", "offset", "category", "department"}

PROFILE_PATH_PATTERNS = [
    "/staff/", "/faculty/", "/profile/", "/people/", "/person/",
    "/academic-staff/", "/node/", "/user/", "/team/", "/lecturer/", "/researcher/",
]

NEGATIVE_URL_PATTERNS = [
    "/events/", "/event/", "/news/", "/article/", "/category/",
    "/index.php/events/", "/blog/", "/project/", "/course/", "/programme/",
    "/overview", "/contact", "/leadership", "/about/overview",
    "/about/biography", "/about/hod-message", "/about/administrative-professionals",
]

HEADERS = {"User-Agent": "UGFacultyBot/2.0 (research project)"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FacultyRecord:
    name:               str | None  = None
    title:              str | None  = None
    academic_rank:      str | None  = None
    department:         str | None  = None
    college:            str | None  = None
    email:              str | None  = None
    phone:              str | None  = None
    office:             str | None  = None
    biography_summary:  str | None  = None
    research_interests: list[str]   = field(default_factory=list)
    courses_taught:     list[str]   = field(default_factory=list)
    publication_links:  list[str]   = field(default_factory=list)
    google_scholar_url: str | None  = None
    orcid_url:          str | None  = None
    personal_website:   str | None  = None
    profile_url:        str | None  = None
    image_url:          str | None  = None
    source_page:        str | None  = None


# ---------------------------------------------------------------------------
# Session + Playwright
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    retry = Retry(
        total=4, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s


class PlaywrightFetcher:
    def __init__(self):
        self._pw = self._browser = None

    def start(self):
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Run: pip install playwright && playwright install chromium")
        self._pw      = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        logger.info("Playwright browser started")

    def stop(self):
        if self._browser: self._browser.close()
        if self._pw:      self._pw.stop()
        logger.info("Playwright browser stopped")

    def fetch(self, url: str) -> str:
        page = self._browser.new_page(user_agent=HEADERS["User-Agent"])
        try:
            page.goto(url, wait_until="networkidle", timeout=PLAYWRIGHT_TIMEOUT_MS)
            page.wait_for_timeout(PLAYWRIGHT_SETTLE_MS)
            html = page.content()
        except PlaywrightTimeoutError:
            logger.warning("Playwright timeout on %s — using partial content", url)
            html = page.content()
        finally:
            page.close()
        time.sleep(REQUEST_DELAY)
        return html


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class UGFacultyScraper:
    def __init__(self, seed_urls: Iterable[str] | None = None):
        self.seed_urls = list(dict.fromkeys(seed_urls or SEED_URLS))
        self.session   = _make_session()
        self._pw: PlaywrightFetcher | None = None

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def _needs_js(self, url: str) -> bool:
        return urlparse(url).netloc.lower() in JS_RENDERED_DOMAINS

    def _start_pw(self):
        if self._pw is None:
            self._pw = PlaywrightFetcher()
            self._pw.start()

    def fetch(self, url: str) -> str:
        if self._needs_js(url):
            self._start_pw()
            return self._pw.fetch(url)
        r = self.session.get(url, timeout=30)
        r.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return r.text

    def save_raw(self, url: str, html: str):
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                      (urlparse(url).netloc + urlparse(url).path).strip("/")) or "index"
        (RAW_DIR / f"{name}.html").write_text(html, encoding="utf-8")

    def soup(self, html: str) -> BeautifulSoup:
        try:    return BeautifulSoup(html, "lxml")
        except: return BeautifulSoup(html, "html.parser")

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def _same_site(self, url: str) -> bool:
        return urlparse(url).netloc.lower().endswith("ug.edu.gh")

    def _allowed_query(self, url: str) -> bool:
        q = urlparse(url).query
        return not q or set(parse_qs(q).keys()).issubset(ALLOWED_QUERY_PARAMS)

    def _looks_like_listing(self, url: str, text: str) -> bool:
        u, t = url.lower(), text.lower()
        return any(p in u for p in ["/faculty", "/staff", "/people", "/academic-staff",
                                     "/leadership", "/departments", "/schools", "/colleges"]) \
            or any(w in t for w in ["faculty", "staff", "people", "leadership",
                                     "department", "college", "school"])

    def _looks_like_profile(self, url: str, text: str) -> bool:
        u, t = url.lower(), text.lower()
        if any(p in u for p in NEGATIVE_URL_PATTERNS): return False
        if not self._allowed_query(url):               return False
        if any(p in u for p in PROFILE_PATH_PATTERNS): return True
        if any(w in t for w in ["prof", "dr", "lecturer", "staff profile",
                                  "mr.", "mrs.", "ms.", "phd"]): return True
        return False

    # ------------------------------------------------------------------
    # Discovery + collection
    # ------------------------------------------------------------------

    def discover_listing_pages(self, start_url: str) -> list[str]:
        html = self.fetch(start_url)
        self.save_raw(start_url, html)
        s = self.soup(html)
        pages = set()
        for a in s.select("a[href]"):
            href = a.get("href", "").strip()
            if not href: continue
            full = urljoin(start_url, href)
            if self._same_site(full) and self._looks_like_listing(full, a.get_text(" ", strip=True)):
                pages.add(full)
        return sorted(pages)[:MAX_LISTING_PAGES]

    def collect_profile_links(self, listing_url: str) -> list[str]:
        # DCS: special MongoDB ID pattern
        if "dcs.ug.edu.gh" in listing_url:
            return self._collect_dcs_links(listing_url)

        visited, links, queue = set(), set(), [listing_url]
        while queue:
            url = queue.pop(0)
            if url in visited: continue
            visited.add(url)
            try:
                html = self.fetch(url)
                self.save_raw(url, html)
            except Exception as e:
                logger.warning("Failed listing page %s: %s", url, e)
                continue
            s = self.soup(html)
            for a in s.select("a[href]"):
                href = a.get("href", "").strip()
                if not href: continue
                full = urljoin(url, href)
                if self._same_site(full) and self._looks_like_profile(full, a.get_text(" ", strip=True)):
                    links.add(full)
            for nxt in self._next_pages(s, url):
                if nxt not in visited: queue.append(nxt)
        return sorted(links)

    def _collect_dcs_links(self, listing_url: str) -> list[str]:
        links = set()
        for url in [listing_url, "https://dcs.ug.edu.gh/faculty"]:
            try:
                html = self.fetch(url)
                self.save_raw(url, html)
                s = self.soup(html)
                for a in s.select("a[href]"):
                    href = a.get("href", "")
                    if re.match(r"^/faculty/[a-f0-9]{24}$", href):
                        links.add("https://dcs.ug.edu.gh" + href)
            except Exception as e:
                logger.warning("DCS fetch failed %s: %s", url, e)
        logger.info("DCS: found %d profile links", len(links))
        return sorted(links)

    def _next_pages(self, s: BeautifulSoup, current: str) -> list[str]:
        for sel in ["a.next", "a[rel='next']", "li.next a", "li.pager__item--next a",
                    "a[aria-label='Next']", "a[title='Go to next page']"]:
            node = s.select_one(sel)
            if node and node.get("href"):
                full = urljoin(current, node["href"])
                if self._same_site(full): return [full]
        for a in s.select("a[href]"):
            if a.get_text(" ", strip=True).lower() in {"next", "next »", "»", "›", ">"}:
                full = urljoin(current, a["href"])
                if self._same_site(full) and full != current: return [full]
        return []

    # ------------------------------------------------------------------
    # Profile parsing
    # ------------------------------------------------------------------

    def scrape_profile(self, url: str) -> FacultyRecord:
        html = self.fetch(url)
        self.save_raw(url, html)
        s = self.soup(html)
        if "dcs.ug.edu.gh" in url:
            return self._parse_dcs(s, url)
        return self._parse_generic(s, url)

    def _pick(self, s: BeautifulSoup, selectors: list[str]) -> str | None:
        for sel in selectors:
            node = s.select_one(sel)
            if node:
                t = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
                if t: return t
        return None

    def _parse_dcs(self, s: BeautifulSoup, url: str) -> FacultyRecord:
        name  = self._pick(s, [".name"])
        title = self._pick(s, [".role"])
        bio   = self._pick(s, [".biography", ".summary"])

        email = phone = office = None
        for g in s.select(".info-group"):
            lbl = g.select_one(".label")
            val = g.select_one(".info")
            if not lbl or not val: continue
            l = lbl.get_text(" ", strip=True).lower()
            v = re.sub(r"\s+", " ", val.get_text(" ", strip=True)).strip()
            if "email"  in l: email  = email  or v
            if "phone"  in l or "tel" in l: phone  = phone  or v
            if "office" in l or "room" in l: office = office or v

        if not email:
            m = s.select_one("a[href^='mailto:']")
            if m: email = m["href"].replace("mailto:", "").strip()

        research, courses = [], []
        for item in s.select(".AccordionItem"):
            h = item.select_one("h3, .AccordionTrigger")
            c = item.select_one(".AccordionContent, .AccordionContentWrapper")
            if not h or not c: continue
            heading = h.get_text(" ", strip=True).lower()
            if "research" in heading:
                research = [re.sub(r"\s+", " ", li.get_text(" ")).strip()
                            for li in c.select("li")] or \
                           [p.strip() for p in re.split(r";|\||\n", c.get_text(" ")) if p.strip()]
            elif "course" in heading or "teaching" in heading:
                courses = [re.sub(r"\s+", " ", li.get_text(" ")).strip()
                           for li in c.select("li") if li.get_text(strip=True)]

        img_tag = s.select_one(".img-wrapper img, .left-section img, .profile img")
        image   = urljoin(url, img_tag["src"]) if img_tag and img_tag.get("src") \
                  and not img_tag["src"].startswith("data:") else None

        scholar = next((urljoin(url, a["href"]) for a in s.select("a[href]")
                        if "scholar.google" in a.get("href", "")), None)
        orcid   = next((urljoin(url, a["href"]) for a in s.select("a[href]")
                        if "orcid.org" in a.get("href", "")), None)

        return FacultyRecord(
            name=name, title=title, academic_rank=title,
            department="Department of Computer Science",
            college="College of Basic and Applied Sciences",
            email=email, phone=phone, office=office,
            biography_summary=bio,
            research_interests=[r for r in research if r],
            courses_taught=[c for c in courses if c],
            google_scholar_url=scholar, orcid_url=orcid,
            profile_url=url, source_page=url, image_url=image,
        )

    def _parse_generic(self, s: BeautifulSoup, url: str) -> FacultyRecord:
        name  = self._pick(s, ["h1", ".node-title", ".page-title", ".field--name-title",
                                ".person-name", ".faculty-name", ".staff-name", ".entry-title"])
        title = self._pick(s, [".field--name-field-rank", ".field--name-field-position",
                                ".views-field-field-rank", ".profile-title", ".position", ".job-title"])
        dept  = self._pick(s, [".department", ".staff-department",
                                ".field--name-field-department", ".views-field-field-department"])
        college = self._pick(s, [".college", ".school",
                                  ".field--name-field-college", ".views-field-field-college"])
        bio   = self._pick(s, [".biography", ".bio", ".profile-bio",
                                ".field--name-field-biography", ".field--name-body"])

        mailto = s.select_one("a[href^='mailto:']")
        email  = mailto["href"].replace("mailto:", "").strip() if mailto else None

        phone  = self._pick(s, [".phone", ".contact-phone", ".field--name-field-phone-number"])
        office = self._pick(s, [".office", ".location", ".field--name-field-office"])

        # Research interests — split on semicolons/pipes only (not commas)
        research = []
        for sel in [".research-interests", ".field--name-field-research-interests", ".interests"]:
            node = s.select_one(sel)
            if not node: continue
            items = [re.sub(r"\s+", " ", li.get_text(" ")).strip() for li in node.select("li")]
            research = [x for x in items if x] or \
                       [p.strip() for p in re.split(r";|\|", node.get_text(" ")) if p.strip()]
            if research: break

        scholar = next((urljoin(url, a["href"]) for a in s.select("a[href]")
                        if "scholar.google" in a.get("href", "")), None)
        orcid   = next((urljoin(url, a["href"]) for a in s.select("a[href]")
                        if "orcid.org" in a.get("href", "")), None)

        # Image — profile containers only
        image = None
        for sel in [".profile-photo img", ".staff-photo img", ".faculty-photo img",
                    ".field--name-user-picture img", ".views-field-field-image img",
                    ".profile-picture img", ".headshot img"]:
            img = s.select_one(sel)
            if img and img.get("src") and not img["src"].startswith("data:"):
                src = urljoin(url, img["src"])
                if re.search(r"\.(jpg|jpeg|png|webp|gif)(\?|$)", src, re.I) \
                        or any(p in src for p in ["/files/", "/images/", "/sites/"]):
                    image = src
                    break

        record = FacultyRecord(
            name=name, title=title, department=dept, college=college,
            email=email, phone=phone, office=office,
            biography_summary=bio[:1200] if bio else None,
            research_interests=research,
            google_scholar_url=scholar, orcid_url=orcid,
            profile_url=url, source_page=url, image_url=image,
        )
        # Infer dept/college from URL if not found on page
        record.department = record.department or self._infer_dept(url)
        record.college    = record.college    or self._infer_college(url)
        record.academic_rank = self._infer_rank(title or "")
        return record

    def _infer_rank(self, title: str) -> str | None:
        for rank in ["Professor", "Associate Professor", "Assistant Professor",
                     "Senior Lecturer", "Lecturer", "Research Fellow", "Senior Research Fellow"]:
            if rank.lower() in title.lower(): return rank
        return None

    def _infer_dept(self, url: str) -> str | None:
        u = url.lower()
        for key, val in {
            "dcs.ug.edu.gh":     "Department of Computer Science",
            "pscience.ug.edu.gh":"Department of Political Science",
            "law.ug.edu.gh":     "School of Law",
            "ugbs.ug.edu.gh":    "University of Ghana Business School",
            "/esl/":             "Department of Education Studies and Leadership",
        }.items():
            if key in u: return val
        return None

    def _infer_college(self, url: str) -> str | None:
        u = url.lower()
        for key, val in {
            "dcs.ug.edu.gh":     "College of Basic and Applied Sciences",
            "cbas.ug.edu.gh":    "College of Basic and Applied Sciences",
            "law.ug.edu.gh":     "College of Humanities",
            "ugbs.ug.edu.gh":    "College of Humanities",
            "pscience.ug.edu.gh":"College of Humanities",
        }.items():
            if key in u: return val
        return None

    def _is_valid(self, r: FacultyRecord) -> bool:
        if not r.name: return False
        nl = r.name.lower()
        if any(nl.startswith(x) for x in ["inaugural", "seminar", "event", "news",
                                            "announcement", "overview", "welcome"]): return False
        if r.profile_url and any(p in r.profile_url.lower()
                                  for p in ["/events/", "/news/"]): return False
        if not any([r.email, r.title, r.department, r.college, r.academic_rank]): return False
        return True

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> list[FacultyRecord]:
        listing_pages: set[str] = set()

        for seed in self.seed_urls:
            try:
                if seed in DISCOVERY_SEED_URLS:
                    pages = self.discover_listing_pages(seed)
                    listing_pages.update(pages)
                    logger.info("Discovered %d pages from %s", len(pages), seed)
                else:
                    listing_pages.add(seed)
            except Exception as e:
                logger.warning("Seed failed %s: %s", seed, e)

        listing_pages.update(TARGETED_SEED_URLS)

        all_profile_urls: set[str] = set()
        for listing in sorted(listing_pages):
            try:
                links = self.collect_profile_links(listing)
                logger.info("Found %d profile links from %s", len(links), listing)
                all_profile_urls.update(links)
            except Exception as e:
                logger.warning("Listing failed %s: %s", listing, e)

        records, skipped = [], []
        for url in sorted(all_profile_urls):
            try:
                r = self.scrape_profile(url)
                if self._is_valid(r):
                    records.append(r)
                    logger.info("Scraped: %s", r.name)
                else:
                    skipped.append(url)
                    logger.debug("Skipped: %s", url)
            except Exception as e:
                logger.warning("Profile failed %s: %s", url, e)

        # Save
        out = PROCESSED_DIR / "faculty_raw.json"
        out.write_text(json.dumps([asdict(r) for r in records],
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved %d records to %s", len(records), out)

        skipped_out = PROCESSED_DIR / "faculty_skipped.json"
        skipped_out.write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")

        if self._pw: self._pw.stop()
        return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    scraper = UGFacultyScraper()
    records = scraper.run()
    print(f"\nDone. {len(records)} faculty records saved to {PROCESSED_DIR}/faculty_raw.json")

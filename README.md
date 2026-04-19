# RAG chunker for University of Ghana documents

Structure-aware chunkers for University of Ghana undergraduate handbooks and
policy documents, with a generic fallback for other markdown docs and a
contextualization layer to boost retrieval quality.

## Files

| File | What it does |
|---|---|
| `chunker.py` | `HandbookChunker` (structure-aware) + `GenericHeadingChunker` (fallback) |
| `policy_chunker.py` | `PolicyChunker` (policy/regulations/rules documents) |
| `contextualize.py` | `template_contextualize()` (free, deterministic) + `llm_contextualize()` (Claude-powered) |
| `chunk_all.py` | CLI driver — routes docs to the right chunker, writes JSONL |
| `*.chunks.jsonl` | One chunk per line, ready to embed and load into a vector DB |

## How it works

### Document routing

`chunk_all.py` routes documents to the appropriate chunker based on filename:

1. **Handbook documents** (filenames matching `handbook|cbas|chs|humanities`) → `HandbookChunker`
2. **Policy documents** (filenames matching `policy|regulations|rules|guidelines|charter|code_of_conduct|...`) → `PolicyChunker`
3. **Everything else** → `GenericHeadingChunker`

Order matters: handbook markers are checked first (they're the most specific).

### HandbookChunker

Walks the markdown line-by-line, maintaining a breadcrumb stack
(`Volume > College > School > Department > Programme > Level > Semester`).
Recognizes four content types:

1. **`programme_table`** — HTML `<table>` blocks listing courses. Linearized
   to `course_code | title | C/E | credits` rows so embeddings work.
2. **`course_description`** — paragraphs under course-code headings like
   `UGRC 141:` or `BCMB 303`. One chunk per course.
3. **`narrative`** — programme overviews, admission requirements, faculty
   lists. Chunked per section; long sections split on paragraph breaks.
4. (Generic chunker emits `section` chunks instead.)

Each chunk gets:
- `text` — the content with breadcrumb prepended (optionally a context blurb too)
- `breadcrumb` — the hierarchical location string
- `content_type` — one of the four above
- `metadata` — `{volume, college, school, department, programme, level, semester,
  course_code, course_title, section, source_file, source_lines}`

### Robustness features

- Strips `![...](...)` image refs and standalone page numbers
- Detects **pseudo-headings** — bold-only lines like `**First Semester**`
  that aren't markdown headings but function as section markers
- Extracts **level/semester from `colspan` header cells** inside tables
  (e.g. `<th colspan="4">SECOND SEMESTER</th>`)
- Blacklists false-positive course codes like `LEVEL 100` that match the
  `AAAA 999` pattern

### PolicyChunker

Chunks policy, regulations, rules, guidelines, and charters at the section level,
preserving enumerated clauses and maintaining a breadcrumb of section hierarchy.
Recognizes content types:

1. **`section`** — a numbered/lettered section (e.g. 3.19, A, B.2.1)
2. **`section_intro`** — introductory text before subsections
3. **`definition`** — a single term from a definitions table
4. **`definitions_table`** — the complete definitions section

Extracts document metadata from the cover page:
- `document_title` — policy/regulation name
- `document_type` — "policy", "regulations", "rules", "guidelines", etc.
- `publication_number`, `volume`, `year`, `approving_authority`

Each chunk includes:
- `section_number` — the section locator (e.g. "3.19.4")
- `section_title` — the section heading
- `parent_path` — the breadcrumb of parent sections

Cleanup features:
- Strips repeated footers ("UG Risk Management Policy")
- Removes page numbers and roman-numeral page markers
- Detects and separates definitions tables (one chunk per term)
- Preserves enumeration sequences (i, ii, iii) (a), (b), (c) together

### GenericHeadingChunker

Fallback chunker for markdown documents that don't match handbook or policy
patterns. Walks markdown headings, maintains a breadcrumb, and chunks by section.

Each chunk gets:
- `document_title` — first heading 4+ chars long, or filename
- `document_type` — always "document" (keeps it simple)
- `heading_path` — array of headings from root to this section

### Metadata coverage on your three handbooks

| File | Chunks | Tables with level | Tables with school |
|---|---|---|---|
| CBAS | 1,391 | 97% | 100% |
| CHS | 854 | 82% | 97% |
| Humanities | 2,068 | 93% | 99% |

### Contextualization

Two options, both prepend a short blurb to each chunk before embedding:

**Template (free, runs now):**
```python
from chunk_all import process_files

# Automatically routes documents and applies contextualization
stats = process_files(["handbook.md", "policy.md"], out_dir="chunks/")
```

Produces blurbs like:

For handbook programme tables:
> *This is a programme-structure table listing the courses for the Department
> of Family and Consumer Sciences Level 200 Semester 1. Each row gives course
> code, title, core/elective status, and credit weight.*

For policy sections:
> *This is Section 3.19 (Graduate Studies Committee) from HANDBOOK FOR
> MASTER'S DEGREE PROGRAMMES — a regulations document of the University
> of Ghana. It sets out rules, procedures, or requirements binding on
> the University community.*

For policy definitions:
> *This is the official definition of "Gender equity" as used in GENDER
> POLICY. It is the authoritative meaning to apply when interpreting
> this policy.*

**LLM (higher quality, costs tokens):**
```python
from anthropic import Anthropic
from contextualize import llm_contextualize, build_parent_index, find_parent

client = Anthropic()
parents = build_parent_index(chunks)
chunks = [
    llm_contextualize(c, parent_text=find_parent(c, parents), client=client)
    for c in chunks
]
```

Uses Haiku by default (fast + cheap for high-volume). Swap to Sonnet/Opus
if you want richer contextualization.

## CLI usage

```bash
python chunk_all.py \
  CBAS_handbook_2017.md \
  CHS_handbook_2017.md \
  Humanities_Handbook_2017.md \
  --out chunks_out/
```

## Loading into a vector DB

Each line of the JSONL is a single chunk. Typical pipeline:

```python
import json

with open("CBAS_handbook_2017.chunks.jsonl") as f:
    for line in f:
        c = json.loads(line)
        embedding = embed(c["text"])
        vector_db.upsert(
            id=f"{c['metadata']['source_file']}:{c['metadata']['source_lines'][0]}",
            vector=embedding,
            metadata=c["metadata"],
            text=c["text"],
        )
```

## Retrieval tips

1. **Hybrid search** — dense embeddings miss exact course codes (`UGRC 150`).
   Run BM25 in parallel and merge.
2. **Metadata filters** — for queries like *"Level 400 electives for
   Psychology"*, use `metadata.level == 400` + `metadata.department like
   "%PSYCHOLOGY%"` + `metadata.content_type == "programme_table"` as a
   pre-filter, then rank by vector similarity.
3. **Small-to-big** — retrieve on small chunks (one table, one course), but
   at generation time pull adjacent narrative chunks via `source_lines`
   proximity to give the model the full programme context.

## Quality verification for policy documents

After chunking a policy corpus, verify these three signals:

1. **Title extraction**: Check that no chunks have `document_title` = "UNIVERSITY OF GHANA"
   or other boilerplate. If any do, the title extractor's boilerplate filter needs tuning
   for that document.

   ```bash
   jq -r '.metadata.document_title' policy.chunks.jsonl | sort | uniq -c | sort -rn
   ```

2. **Enumeration preservation**: Find a section you know contains a list (e.g. 5-7 items).
   Confirm all items are in a single chunk.

   ```bash
   jq '.metadata | select(.section_number=="3.19")' policy.chunks.jsonl
   ```

3. **Definitions tables**: Check for per-term definitions chunks. A definitions section
   with 30 terms should produce 30+ `definition` chunks plus one `definitions_table` chunk.

   ```bash
   jq 'select(.content_type == "definition" or .content_type == "definitions_table")' policy.chunks.jsonl | wc -l
   ```

## Adding new document types

When a new document family comes in (e.g. exam regulations, research papers), add a rule
to `chunk_all.pick_chunker()`. For truly novel structure, write a new chunker class —
all chunkers share the `Chunk` dataclass so the downstream pipeline doesn't need to change.

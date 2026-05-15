"""
refresh.py — Manual refresh command for faculty data.

Runs the full pipeline in sequence:
    scrape → normalize → chunk → load into Chroma

Usage:
    python refresh.py                    # full refresh
    python refresh.py --skip-scrape      # normalize/chunk/load only (reuse existing raw data)
    python refresh.py --step scrape      # run one step only
    python refresh.py --step normalize
    python refresh.py --step chunk
    python refresh.py --step load
"""

import argparse
import logging
import sys
import time

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("refresh.log", encoding="utf-8"),
    ],
)


def step_scrape():
    logger.info("=" * 50)
    logger.info("STEP 1/4 — Scraping faculty pages")
    logger.info("=" * 50)
    from faculty.scraper import UGFacultyScraper
    scraper = UGFacultyScraper()
    records = scraper.run()
    logger.info("Scrape complete: %d records", len(records))
    return len(records)


def step_normalize():
    logger.info("=" * 50)
    logger.info("STEP 2/4 — Normalizing records")
    logger.info("=" * 50)
    from faculty.normalizer import run as normalize_run
    records = normalize_run()
    logger.info("Normalize complete: %d records", len(records))
    return len(records)


def step_chunk():
    logger.info("=" * 50)
    logger.info("STEP 3/4 — Chunking records")
    logger.info("=" * 50)
    from faculty.chunker import run as chunk_run
    chunks = chunk_run()
    logger.info("Chunk complete: %d chunks", len(chunks))
    return len(chunks)


def step_load():
    logger.info("=" * 50)
    logger.info("STEP 4/4 — Loading into Chroma")
    logger.info("=" * 50)
    from faculty.loader import run as loader_run
    count = loader_run()
    logger.info("Load complete: %d records in Chroma", count)
    return count


STEPS = {
    "scrape":    step_scrape,
    "normalize": step_normalize,
    "chunk":     step_chunk,
    "load":      step_load,
}


def main():
    ap = argparse.ArgumentParser(description="Auntie Aba faculty data refresh")
    ap.add_argument("--skip-scrape", action="store_true",
                    help="Skip scraping, use existing raw data")
    ap.add_argument("--step", choices=list(STEPS.keys()),
                    help="Run only one step of the pipeline")
    args = ap.parse_args()

    start = time.time()

    if args.step:
        # Single step mode
        fn = STEPS[args.step]
        fn()
    else:
        # Full pipeline
        if not args.skip_scrape:
            step_scrape()
        else:
            logger.info("Skipping scrape — using existing raw data")

        step_normalize()
        step_chunk()
        step_load()

    elapsed = time.time() - start
    logger.info("=" * 50)
    logger.info("Refresh complete in %.1fs", elapsed)
    logger.info("=" * 50)
    print(f"\n✅ Done in {elapsed:.1f}s — faculty data is up to date in Chroma.")


if __name__ == "__main__":
    main()

"""
Public Fake News Dataset Downloader
=====================================
Downloads the Fernandez & Devaraj (2019) Philippine Fake News Corpus and
merges it into dataset.csv. Articles are tagged with source_dataset="fernandez2019"
so they are always distinguishable from self-scraped articles in the CSV,
even though training combines both.

Reference:
  Fernandez & Devaraj (2019). "Computing the Linguistic-Based Cues of Fake News
  in the Philippines Towards its Detection." ACM WIMS 2019.
  GitHub: github.com/aaroncarlfernandez/Philippine-Fake-News-Corpus

Usage:
  python download_public.py              # download Fernandez, merge into dataset.csv
  python download_public.py --dry-run    # show stats without writing
  python download_public.py --list       # list available datasets
"""

import argparse
import csv
import io
import logging
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_FILE = "dataset.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


@dataclass
class Article:
    title:          str
    text:           str
    label:          str   # "fake" | "real"
    source:         str
    url:            str
    date:           str
    source_dataset: str = "fernandez2019"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_csv_text(url: str) -> str | None:
    log.info(f"  Downloading {url} ...")
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        log.info(f"  OK — {len(r.content):,} bytes")
        return r.text
    except Exception as e:
        log.warning(f"  FAILED: {e}")
        return None


def load_existing_urls(path: str) -> set[str]:
    seen: set[str] = set()
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add(row.get("url", ""))
    except FileNotFoundError:
        pass
    return seen


def load_existing_titles(path: str) -> set[str]:
    """Fallback dedup by normalized title when URLs are missing."""
    seen: set[str] = set()
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = row.get("title", "").strip().lower()
                if t:
                    seen.add(t)
    except FileNotFoundError:
        pass
    return seen


def append_csv(articles: list[Article], path: str):
    """Append articles to CSV. Writes header only if file does not yet exist."""
    fieldnames = ["title", "text", "label", "source", "url", "date", "source_dataset"]
    file_exists = Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for art in articles:
            writer.writerow({
                "title":          art.title,
                "text":           art.text,
                "label":          art.label,
                "source":         art.source,
                "url":            art.url,
                "date":           art.date,
                "source_dataset": art.source_dataset,
            })


def clean(s: str) -> str:
    return " ".join(str(s).split()).strip()


# ---------------------------------------------------------------------------
# Fernandez & Devaraj (2019) — Philippine Fake News Corpus
# 22,458 articles: 14,802 Credible / 7,656 Not Credible
# Credible sources: Philippine Daily Inquirer, Manila Bulletin, Manila Times
# Not Credible: Adobo Chronicles, GR Pundit, Get Real Philippines, Thinking Pinoy
# Labeling authority: Philippine Senate, CMFR, CBCP
# ---------------------------------------------------------------------------

FERNANDEZ_ZIP_URL = (
    "https://raw.githubusercontent.com/aaroncarlfernandez/Philippine-Fake-News-Corpus"
    "/master/Philippine%20Fake%20News%20Corpus.zip"
)
FERNANDEZ_ZIP_INNER = "Philippine Fake News Corpus.csv"


def download_fernandez() -> list[Article]:
    """
    Downloads the Fernandez & Devaraj (2019) Philippine Fake News Corpus ZIP (~23 MB),
    extracts the single CSV inside, and returns labeled Article objects.

    CSV columns: Headline, Content, Authors, Date, URL, Brand, Label
    Label values: 'Credible' (real) / 'Not Credible' (fake)
    """
    log.info("=== Fernandez & Devaraj (2019) — Philippine Fake News Corpus ===")
    log.info(f"  Downloading ZIP from GitHub (~23 MB) ...")

    try:
        r = requests.get(FERNANDEZ_ZIP_URL, headers=HEADERS, timeout=180)
        r.raise_for_status()
        log.info(f"  Downloaded {len(r.content):,} bytes")
    except Exception as e:
        log.error(f"  Failed to download ZIP: {e}")
        return []

    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open(FERNANDEZ_ZIP_INNER) as f:
            text_wrapper = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            reader = csv.DictReader(text_wrapper)
            rows = list(reader)
    except Exception as e:
        log.error(f"  Failed to parse ZIP/CSV: {e}")
        return []

    log.info(f"  Parsed {len(rows):,} rows — columns: {list(rows[0].keys()) if rows else '?'}")

    articles: list[Article] = []
    skipped = 0
    for row in rows:
        raw_label = clean(row.get("Label", "")).lower()
        if raw_label == "credible":
            label = "real"
        elif raw_label in ("not credible", "not_credible"):
            label = "fake"
        else:
            skipped += 1
            continue

        title = clean(row.get("Headline", ""))
        body  = clean(row.get("Content",  ""))
        url   = clean(row.get("URL",      ""))
        date  = clean(row.get("Date",     ""))
        brand = clean(row.get("Brand",    "") or "Fernandez2019")

        if len(title) < 5 or len(body) < 100:
            skipped += 1
            continue

        articles.append(Article(
            title=title,
            text=body[:5000],
            label=label,
            source=brand,
            url=url,
            date=date,
            source_dataset="fernandez2019",
        ))

    real_count = sum(1 for a in articles if a.label == "real")
    fake_count = sum(1 for a in articles if a.label == "fake")
    log.info(f"  Real (Credible)     : {real_count:,}")
    log.info(f"  Fake (Not Credible) : {fake_count:,}")
    log.info(f"  Skipped             : {skipped:,}")
    log.info(f"  Total collected     : {len(articles):,}\n")
    return articles


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, tuple[str, Callable[[], list[Article]]]] = {
    "fernandez": (
        "Fernandez & Devaraj (2019) — Philippine Fake News Corpus (ACM WIMS)",
        download_fernandez,
    ),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download public Philippine fake news datasets")
    parser.add_argument("--only",    nargs="*", choices=list(DATASETS), metavar="KEY",
                        help="Only download specific datasets (keys: fernandez)")
    parser.add_argument("--list",    action="store_true", help="List available datasets and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without writing to CSV")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for key, (name, _) in DATASETS.items():
            print(f"  {key:12s}  {name}")
        return

    keys = args.only if args.only else list(DATASETS.keys())

    existing_urls   = load_existing_urls(OUTPUT_FILE)
    existing_titles = load_existing_titles(OUTPUT_FILE)
    log.info(f"Existing dataset: {len(existing_urls)} articles in {OUTPUT_FILE}")

    all_new: list[Article] = []
    for key in keys:
        name, fn = DATASETS[key]
        log.info(f"\n{'='*60}")
        log.info(f"Downloading: {name}")
        batch = fn()

        batch_urls   = {a.url   for a in all_new if a.url}
        batch_titles = {a.title.strip().lower() for a in all_new}

        deduped: list[Article] = []
        for a in batch:
            norm_title = a.title.strip().lower()
            if a.url and (a.url in existing_urls or a.url in batch_urls):
                continue
            if not a.url and (norm_title in existing_titles or norm_title in batch_titles):
                continue
            deduped.append(a)
            if a.url:
                batch_urls.add(a.url)
            batch_titles.add(norm_title)

        log.info(f"  After dedup: {len(deduped)} new (dropped {len(batch)-len(deduped)} duplicates)")
        all_new.extend(deduped)

    fake_new    = sum(1 for a in all_new if a.label == "fake")
    real_new    = sum(1 for a in all_new if a.label == "real")
    total_after = len(existing_urls) + len(all_new)

    log.info(f"\n{'='*60}")
    log.info(f"=== Summary ===")
    log.info(f"  New fake articles   : {fake_new}")
    log.info(f"  New real articles   : {real_new}")
    log.info(f"  New total           : {len(all_new)}")
    log.info(f"  Dataset total after : ~{total_after}")
    log.info(f"  source_dataset tags : all new rows = 'fernandez2019'")

    if args.dry_run:
        log.info("  Dry run — nothing written.")
        return

    if not all_new:
        log.info("  Nothing new to add.")
        return

    append_csv(all_new, OUTPUT_FILE)
    log.info(f"  Appended {len(all_new)} articles → {OUTPUT_FILE}")
    log.info("\nDone! Run retrain.sh to retrain the model.")


if __name__ == "__main__":
    main()

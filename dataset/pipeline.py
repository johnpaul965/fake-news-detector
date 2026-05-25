"""
Full Training Pipeline
=======================
Runs: scraper.py → features.py → train.py → copies results to API server

Usage:
  python pipeline.py

Or run individual steps:
  python scraper.py    # collect articles
  python features.py   # extract features
  python train.py      # train models, output model_results.json
"""

import subprocess
import shutil
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

API_DATA_DIR = Path("../artifacts/api-server/data")


def run(script: str):
    log.info(f"\n{'='*60}")
    log.info(f"Running {script} ...")
    log.info(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, script],
        cwd=Path(__file__).parent,
    )
    if result.returncode != 0:
        log.error(f"{script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def copy_results():
    src = Path("model_results.json")
    dst_dir = API_DATA_DIR
    dst_dir.mkdir(parents=True, exist_ok=True)

    if src.exists():
        shutil.copy(src, dst_dir / "model_results.json")
        log.info(f"Copied model_results.json → {dst_dir}/")

    csv_src = Path("dataset.csv")
    csv_dst = dst_dir / "dataset.csv"
    if csv_src.exists():
        shutil.copy(csv_src, csv_dst)
        log.info(f"Copied dataset.csv → {dst_dir}/")

    log.info("\nPipeline complete!")
    log.info("Restart the API server workflow to pick up the new data and models.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-scraper", action="store_true", help="Skip scraping step")
    args = parser.parse_args()

    if not args.skip_scraper:
        run("scraper.py")

    run("features.py")
    run("train.py")
    copy_results()

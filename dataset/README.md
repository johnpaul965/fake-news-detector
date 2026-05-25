# Philippine Fake News Dataset

Scraper for collecting labeled news articles from Philippine sources.

## Sources

| Label | Source | URL |
|-------|--------|-----|
| fake  | VERA Files Fact Check | verafiles.org/vera-files-fact-check |
| fake  | Rappler Fact Check | rappler.com/facts-first |
| real  | Inquirer News | inquirer.net |
| real  | GMA News | gmanetwork.com/news |

## How to run

**Step 1 — Install Python dependencies**
```bash
pip install requests beautifulsoup4 tqdm
```

**Step 2 — Run the scraper (from the dataset/ folder)**
```bash
cd dataset
python scraper.py
```

This will take 20–40 minutes depending on your internet speed.
It collects ~150 articles per source = ~600 total articles.

**Step 3 — Copy the output to the API server**
```bash
cp dataset.csv ../artifacts/api-server/data/dataset.csv
```

## Output format (dataset.csv)

| Column | Description |
|--------|-------------|
| title  | Article headline |
| text   | Full article body |
| label  | `fake` or `real` |
| source | Source name |
| url    | Original URL |
| date   | Publication date |

## Adjusting the target size

Edit `TARGET_PER_SOURCE` in `scraper.py` (default: 150).
Higher = bigger dataset but longer scraping time.

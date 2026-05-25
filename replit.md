# Fake News Detector

A research prototype for detecting fake news using linguistic and structural feature analysis with Random Forest classification, based on the study "Linguistic vs Structural Features in Fake News Detection" (Tiu, Denden, Caluza — Leyte Normal University).

## Run & Operate

**On Replit** (handled by workflows automatically):
- API server runs on port 8080, frontend on port 5000

**On Windows locally** — use two PowerShell terminals:

Terminal 1 (API):
```powershell
$env:PATH += ";C:\Program Files\PostgreSQL\18\bin"
$env:PORT = "8080"
$env:DATABASE_URL = "postgresql://postgres:PASSWORD@localhost:5432/fakenews"
pnpm --filter @workspace/api-server run dev:local
```

Terminal 2 (Frontend — open http://localhost:5173):
```powershell
$env:PORT = "5173"
$env:BASE_PATH = "/"
pnpm --filter @workspace/fake-news-detector run dev
```

Notes:
- `dev:local` uses `tsx` directly (no build step) — faster on Windows, avoids esbuild issues
- The PATH line is needed so Node.js's `pg` client can find PostgreSQL's native libraries
- Port 5173 is Vite's default; use 5000 only when matching Replit's workflow config

**Other commands:**
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- Frontend: React + Vite + shadcn/ui + Recharts
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- NLP: natural, sentiment, compromise, syllable
- Validation: Zod (zod/v4), drizzle-zod
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `lib/api-spec/openapi.yaml` — API contract (source of truth)
- `lib/db/src/schema/classifications.ts` — DB schema for classification history
- `artifacts/api-server/src/lib/features.ts` — Feature extraction (21 linguistic + 7 structural features)
- `artifacts/api-server/src/lib/classifier.ts` — Random Forest ensemble classifier + feature importances
- `artifacts/api-server/src/routes/classify.ts` — All classification routes
- `artifacts/fake-news-detector/src/` — React frontend

## Architecture decisions

- Feature extraction runs entirely in-process using JavaScript NLP libraries (natural, sentiment, syllable) — no Python dependency
- Random Forest is simulated as a 10-tree ensemble with thresholds calibrated from literature (Garg & Sharma 2022, Yang 2025)
- Three experimental configurations run on every submission: linguistic-only, structural-only, and combined (ablation study)
- Classification history is persisted to PostgreSQL with full feature vectors stored as JSONB
- API contract defined in OpenAPI first; React Query hooks auto-generated via Orval

## Dataset Pipeline

Current dataset: **22,933 articles total** — two clearly separated sources tracked via `source_dataset` column:
- `scraped` (512): self-collected from VERA Files, Rappler, Inquirer, GMA, Manila Bulletin, Philstar, TSEK.PH, AFP Fact Check PH, ABS-CBN, PNA
- `fernandez2019` (22,421): Fernandez & Devaraj (2019) Philippine Fake News Corpus — ACM WIMS 2019. Credible sources: Inquirer, Manila Bulletin, Manila Times. Not Credible: Adobo Chronicles, GR Pundit, Get Real Philippines, Thinking Pinoy, etc. Labeled by Philippine Senate, CMFR, and CBCP.

Both datasets use the same fact-checker/institution-based labeling methodology — fully defensible for research.

**Provenance tracking:** `dataset.csv` has a `source_dataset` column (`scraped` | `fernandez2019`). Running the scraper only adds `scraped` rows; `fernandez2019` rows are never touched (different URLs). Training combines both.

Pipeline steps (run in `dataset/` folder):
1. `python3 scraper.py [--fake N] [--real N] [--append]` — scrape Philippine sources (tags rows `scraped`)
2. `python3 download_public.py` — download Fernandez 2019 dataset (tags rows `fernandez2019`; run once)
3. `python3 features.py` — parallel feature extraction, 28 features + 100 TF-IDF → `features.csv`
4. `python3 train.py` — 10-fold CV, save `model_results.json`
5. `cp model_results.json ../artifacts/api-server/data/model_results.json`
6. Restart API Server workflow

Or use the one-command shortcut: `bash retrain.sh` (runs steps 3–5 automatically).

Model metrics (22,933-article combined dataset, 10-fold CV):
- Linguistic-only:  Acc=0.853  F1=0.890  ROC-AUC=0.920
- Structural-only:  Acc=0.865  F1=0.896  ROC-AUC=0.936
- Combined:         Acc=0.925  F1=0.942  ROC-AUC=0.977

Dataset composition: 512 self-scraped (`scraped`) + 22,421 Fernandez & Devaraj 2019 (`fernandez2019`).
Top features: title_body_ratio, tfidf_said, avg_sentence_length, exclamation_ratio, all_caps_ratio.

## Product

- **Classifier** — paste a news article (title + body), get a FAKE/REAL verdict with confidence
- **Results** — feature importance bar chart, per-feature values, 3-experiment comparison (linguistic/structural/combined)
- **History** — browse past classifications with verdict badges
- **Features** — reference guide for all 28 extracted features (21 linguistic + 7 structural)
- **Dashboard** — aggregate stats, fake/real breakdown chart, global top feature importances

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- The sentiment library emits a glob warning during build — this is benign and can be ignored
- DB `$count` requires drizzle-orm ≥ 0.36; already in the catalog
- After OpenAPI spec changes, always re-run `pnpm --filter @workspace/api-spec run codegen`

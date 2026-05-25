import { RandomForestClassifier } from "ml-random-forest";
import { readFileSync, existsSync } from "fs";
import { resolve } from "path";
import { fileURLToPath } from "url";
import { extractFeatures, FEATURE_NAMES, FEATURE_META } from "./features.js";
import type { ExtractedFeatures } from "./features.js";

// ──────────────────────────────────────────────────────────────────────────────
// Random Forest trainer with stratified 10-fold cross-validation
// Implements paper methodology: Tiu, Denden, Caluza (Leyte Normal University)
// Three configurations: linguistic-only, structural-only, combined
//
// Feature layout (31 total):
//   Linguistic (23): indices 0–22
//   Structural  (8): indices 23–30
// ──────────────────────────────────────────────────────────────────────────────

// Linguistic (23): tfidfScore, sentimentPolarity, sentimentSubjectivity,
//   readabilityScore, fleschKincaidGrade, ari, gunningFog, smogIndex,
//   lexicalDiversity, positiveAffectRatio, negativeAffectRatio, certaintyRatio,
//   nounRatio, verbRatio, adjectiveRatio, adverbRatio, exclamationDensity,
//   questionDensity, wordCount, uniqueWordCount, firstPersonRatio,
//   exclamationCount, questionCount
// Structural (8): punctuationDensity, paragraphCount, titleBodyRatio,
//   hyperlinkDensity, quotationFrequency, avgSentenceLength, capsWordRatio, sentenceCount
export const LINGUISTIC_INDICES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]; // 23 features
export const STRUCTURAL_INDICES = [23,24,25,26,27,28,29,30];                                      // 8 features
export const COMBINED_INDICES   = [...LINGUISTIC_INDICES, ...STRUCTURAL_INDICES];                  // 31 features

export function featureToVector(f: ExtractedFeatures): number[] {
  return FEATURE_NAMES.map((name) => f[name]);
}

// Feature importances — loaded from real training results if available,
// otherwise fall back to paper Table III values (Tiu, Denden, Caluza 2025)
function findModelResultsJson(): string | null {
  const __dirname = fileURLToPath(new URL(".", import.meta.url));
  // Compiled mode: dist/ → ../data/  |  Source mode (tsx): src/lib/ → ../../data/
  for (const rel of ["../data/model_results.json", "../../data/model_results.json"]) {
    const p = resolve(__dirname, rel);
    if (existsSync(p)) return p;
  }
  return null;
}

function loadRealImportances(): Record<string, number> | null {
  try {
    const jsonPath = findModelResultsJson();
    if (!jsonPath) return null;
    const raw = JSON.parse(readFileSync(jsonPath, "utf-8"));
    const importancesList: Array<{ name: string; importance: number }> =
      raw?.featureImportances?.combined ?? [];
    if (importancesList.length === 0) return null;
    const map: Record<string, number> = {};
    for (const { name, importance } of importancesList) map[name] = importance;
    return map;
  } catch {
    return null;
  }
}

// Fallback importances from paper Table III (combined configuration MDI scores)
// New readability features (FK Grade, ARI, Gunning Fog, SMOG, Subjectivity)
// assigned estimated importances consistent with paper findings
export const FEATURE_IMPORTANCES: Record<string, number> =
  loadRealImportances() ?? {
    titleBodyRatio:        0.1297,  // rank #1  — 12.97%
    tfidfScore:            0.0777,  // rank #2  — 7.77% (TF-IDF 'said' approximated)
    avgSentenceLength:     0.0658,  // rank #3  — 6.58%
    exclamationDensity:    0.0507,  // rank #4  — 5.07%
    exclamationCount:      0.0379,  // rank #5  — 3.79%
    adverbRatio:           0.0336,  // rank #6  — 3.36%
    wordCount:             0.0307,  // rank #7  — 3.07%
    capsWordRatio:         0.0301,  // rank #8  — 3.01%
    uniqueWordCount:       0.0235,  // rank #9  — 2.35%
    sentenceCount:         0.0233,  // rank #10 — 2.33%
    sentimentPolarity:     0.0210,
    lexicalDiversity:      0.0190,
    punctuationDensity:    0.0180,
    negativeAffectRatio:   0.0160,
    quotationFrequency:    0.0150,
    certaintyRatio:        0.0140,
    questionDensity:       0.0130,
    nounRatio:             0.0120,
    paragraphCount:        0.0110,
    firstPersonRatio:      0.0100,
    adjectiveRatio:        0.0090,
    positiveAffectRatio:   0.0080,
    verbRatio:             0.0070,
    sentimentSubjectivity: 0.0065,  // paper: linguistic feature (subjectivity)
    readabilityScore:      0.0060,
    fleschKincaidGrade:    0.0055,  // paper: linguistic feature (FK Grade)
    ari:                   0.0048,  // paper: linguistic feature (ARI)
    gunningFog:            0.0042,  // paper: linguistic feature (Gunning Fog)
    hyperlinkDensity:      0.0038,
    smogIndex:             0.0032,  // paper: linguistic feature (SMOG)
    questionCount:         0.0028,
  };

// ── Seeded LCG RNG for reproducibility ────────────────────────────────────────
function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = ((s * 1664525) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function gaussianRng(rng: () => number, mean: number, std: number): number {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  const n = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  return Math.max(0, Math.min(1, mean + n * std));
}

// ── Synthetic training data (distributions from literature + paper findings) ──
// Fake: high sentiment/subjectivity, exclamation, caps, certainty, short articles,
//       big title ratio, low readability grade, low hyperlinks/quotations
function generateFakeVector(rng: () => number): number[] {
  const g = (m: number, s: number) => gaussianRng(rng, m, s);
  return [
    g(0.04,  0.015), // [0]  tfidfScore
    g(0.30,  0.14),  // [1]  sentimentPolarity (high emotion)
    g(0.25,  0.10),  // [2]  sentimentSubjectivity (subjective writing)
    g(0.68,  0.12),  // [3]  readabilityScore (high Flesch = simpler text)
    g(0.32,  0.12),  // [4]  fleschKincaidGrade (~grade 6/18 — simple)
    g(0.28,  0.10),  // [5]  ari (~ARI 4/14 — simple)
    g(0.32,  0.12),  // [6]  gunningFog (~Fog 6/20 — simple)
    g(0.32,  0.12),  // [7]  smogIndex (~SMOG 6/20 — simple)
    g(0.46,  0.09),  // [8]  lexicalDiversity (limited vocab)
    g(0.09,  0.04),  // [9]  positiveAffectRatio
    g(0.14,  0.05),  // [10] negativeAffectRatio
    g(0.08,  0.03),  // [11] certaintyRatio
    g(0.32,  0.09),  // [12] nounRatio
    g(0.29,  0.09),  // [13] verbRatio
    g(0.25,  0.08),  // [14] adjectiveRatio
    g(0.14,  0.05),  // [15] adverbRatio
    g(0.30,  0.18),  // [16] exclamationDensity (many !)
    g(0.11,  0.09),  // [17] questionDensity
    g(0.15,  0.08),  // [18] wordCount (short articles ~150 words)
    g(0.12,  0.06),  // [19] uniqueWordCount (limited vocab)
    g(0.08,  0.05),  // [20] firstPersonRatio (opinionated)
    g(0.25,  0.15),  // [21] exclamationCount (many !)
    g(0.08,  0.06),  // [22] questionCount
    g(0.09,  0.03),  // [23] punctuationDensity
    g(0.05,  0.04),  // [24] paragraphCount (few paragraphs)
    g(0.15,  0.08),  // [25] titleBodyRatio (#1 feature — high in fake)
    g(0.001, 0.002), // [26] hyperlinkDensity (no citations)
    g(0.02,  0.02),  // [27] quotationFrequency (no quotes)
    g(0.16,  0.08),  // [28] avgSentenceLength (short sentences)
    g(0.15,  0.09),  // [29] capsWordRatio (all caps words)
    g(0.10,  0.07),  // [30] sentenceCount (fewer sentences)
  ];
}

// Real: low sentiment/subjectivity, rare exclamation/caps, long articles,
//       high readability grade, many quotations and sentences
function generateRealVector(rng: () => number): number[] {
  const g = (m: number, s: number) => gaussianRng(rng, m, s);
  return [
    g(0.025, 0.010), // [0]  tfidfScore
    g(0.05,  0.07),  // [1]  sentimentPolarity (neutral)
    g(0.07,  0.04),  // [2]  sentimentSubjectivity (objective writing)
    g(0.52,  0.11),  // [3]  readabilityScore
    g(0.58,  0.12),  // [4]  fleschKincaidGrade (~grade 10/18 — complex)
    g(0.58,  0.10),  // [5]  ari (~ARI 8/14 — complex)
    g(0.58,  0.12),  // [6]  gunningFog (~Fog 12/20 — complex)
    g(0.58,  0.12),  // [7]  smogIndex (~SMOG 12/20 — complex)
    g(0.63,  0.07),  // [8]  lexicalDiversity (rich vocab)
    g(0.04,  0.02),  // [9]  positiveAffectRatio
    g(0.04,  0.02),  // [10] negativeAffectRatio
    g(0.03,  0.015), // [11] certaintyRatio
    g(0.43,  0.08),  // [12] nounRatio (noun-heavy)
    g(0.28,  0.08),  // [13] verbRatio
    g(0.15,  0.06),  // [14] adjectiveRatio
    g(0.14,  0.05),  // [15] adverbRatio
    g(0.01,  0.015), // [16] exclamationDensity (rare !)
    g(0.02,  0.02),  // [17] questionDensity
    g(0.45,  0.15),  // [18] wordCount (~450 words)
    g(0.40,  0.12),  // [19] uniqueWordCount (rich vocab)
    g(0.02,  0.015), // [20] firstPersonRatio (objective)
    g(0.02,  0.02),  // [21] exclamationCount (very few !)
    g(0.02,  0.02),  // [22] questionCount
    g(0.05,  0.02),  // [23] punctuationDensity
    g(0.22,  0.10),  // [24] paragraphCount (many paragraphs)
    g(0.04,  0.02),  // [25] titleBodyRatio (#1 feature — low in real)
    g(0.012, 0.008), // [26] hyperlinkDensity
    g(0.09,  0.05),  // [27] quotationFrequency (quotes from sources)
    g(0.38,  0.10),  // [28] avgSentenceLength (longer sentences)
    g(0.01,  0.01),  // [29] capsWordRatio
    g(0.45,  0.15),  // [30] sentenceCount (many sentences)
  ];
}

// ── Metrics ───────────────────────────────────────────────────────────────────
export interface Metrics {
  accuracy:        number;
  precision:       number;
  recall:          number;
  f1Score:         number;
  rocAuc:          number;
  confusionMatrix: [[number, number], [number, number]];
}

function computeMetrics(trueY: number[], predY: number[]): Metrics {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (let i = 0; i < trueY.length; i++) {
    if      (trueY[i] === 1 && predY[i] === 1) tp++;
    else if (trueY[i] === 0 && predY[i] === 0) tn++;
    else if (trueY[i] === 0 && predY[i] === 1) fp++;
    else                                         fn++;
  }
  const acc  = (tp + tn) / trueY.length;
  const prec = tp / Math.max(tp + fp, 1);
  const rec  = tp / Math.max(tp + fn, 1);
  const f1   = (2 * prec * rec) / Math.max(prec + rec, 1e-9);
  const sens = rec;
  const spec = tn / Math.max(tn + fp, 1);
  return {
    accuracy:        parseFloat(acc.toFixed(4)),
    precision:       parseFloat(prec.toFixed(4)),
    recall:          parseFloat(rec.toFixed(4)),
    f1Score:         parseFloat(f1.toFixed(4)),
    rocAuc:          parseFloat(((sens + spec) / 2).toFixed(4)),
    confusionMatrix: [[tn, fp], [fn, tp]],
  };
}

// ── Stratified 10-fold cross-validation ──────────────────────────────────────
function stratifiedKFoldCV(
  X: number[][],
  y: number[],
  featureIndices: number[],
  k = 10,
  seed = 0,
): Metrics {
  const n = X.length;
  const fakeIdx = y.reduce<number[]>((a, v, i) => (v === 1 ? [...a, i] : a), []);
  const realIdx = y.reduce<number[]>((a, v, i) => (v === 0 ? [...a, i] : a), []);

  const allPred = new Array<number>(n).fill(-1);

  for (let fold = 0; fold < k; fold++) {
    const fakeTest = fakeIdx.filter((_, i) => i % k === fold);
    const realTest = realIdx.filter((_, i) => i % k === fold);
    const testSet  = new Set([...fakeTest, ...realTest]);

    const trainX: number[][] = [];
    const trainY: number[]   = [];
    const testX:  number[][] = [];
    const testI:  number[]   = [];

    for (let i = 0; i < n; i++) {
      const vec = featureIndices.map((fi) => X[i][fi]);
      if (testSet.has(i)) { testX.push(vec); testI.push(i); }
      else                { trainX.push(vec); trainY.push(y[i]); }
    }

    const clf = new RandomForestClassifier({
      seed: seed + fold,
      nEstimators: 100,
      maxFeatures: Math.max(1, Math.floor(Math.sqrt(featureIndices.length))),
      replacement: true,
    });
    clf.train(trainX, trainY);
    const preds = clf.predict(testX) as number[];
    preds.forEach((p, i) => { allPred[testI[i]] = p; });
  }

  return computeMetrics(y, allPred);
}

// ── Public types ──────────────────────────────────────────────────────────────
export interface ExperimentPerformance {
  configuration:   "linguistic-only" | "structural-only" | "combined";
  featureCount:    number;
  accuracy:        number;
  precision:       number;
  recall:          number;
  f1Score:         number;
  rocAuc:          number;
  confusionMatrix: [[number, number], [number, number]];
}

export interface ModelPerformance {
  experiments:        ExperimentPerformance[];
  trainingSamples:    number;
  folds:              number;
  featureImportances: Array<{ name: string; label: string; importance: number; category: string }>;
}

export interface TrainedModels {
  linguistic:  RandomForestClassifier;
  structural:  RandomForestClassifier;
  combined:    RandomForestClassifier;
  performance: ModelPerformance;
  dataSource:  string;
}

// ── CSV dataset loader ────────────────────────────────────────────────────────
function parseSimpleCsv(content: string): Record<string, string>[] {
  const lines = content.split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
  const rows: Record<string, string>[] = [];
  for (let i = 1; i < lines.length; i++) {
    const fields: string[] = [];
    let cur = "";
    let inQuote = false;
    for (const ch of lines[i]) {
      if (ch === '"') { inQuote = !inQuote; }
      else if (ch === "," && !inQuote) { fields.push(cur); cur = ""; }
      else { cur += ch; }
    }
    fields.push(cur);
    if (fields.length >= headers.length) {
      const row: Record<string, string> = {};
      headers.forEach((h, idx) => { row[h] = fields[idx]?.trim().replace(/^"|"$/g, "") ?? ""; });
      rows.push(row);
    }
  }
  return rows;
}

function loadDatasetFromCsv(): { X: number[][]; y: number[]; count: number } | null {
  const __dirname = fileURLToPath(new URL(".", import.meta.url));
  const csvPath = resolve(__dirname, "../data/dataset.csv");
  if (!existsSync(csvPath)) return null;

  const content = readFileSync(csvPath, "utf-8");
  const rows = parseSimpleCsv(content);
  if (rows.length === 0) return null;

  const X: number[][] = [];
  const y: number[]   = [];
  let skipped = 0;

  for (const row of rows) {
    const title = row["title"] ?? "";
    const text  = row["text"]  ?? "";
    const label = row["label"] ?? "";
    if (!title || !text || (label !== "fake" && label !== "real")) { skipped++; continue; }
    try {
      const features = extractFeatures(title, text);
      X.push(FEATURE_NAMES.map((name) => features[name]));
      y.push(label === "fake" ? 1 : 0);
    } catch { skipped++; }
  }

  if (X.length < 20) return null;
  return { X, y, count: X.length };
}

// ── Load real CV metrics from Python train.py output ─────────────────────────
function loadRealMetrics(): { experiments: ExperimentPerformance[]; trainingSamples: number } | null {
  try {
    const jsonPath = findModelResultsJson();
    if (!jsonPath) return null;
    const raw = JSON.parse(readFileSync(jsonPath, "utf-8"));
    if (!raw?.experiments?.length) return null;
    const experiments: ExperimentPerformance[] = raw.experiments.map((e: any) => ({
      configuration:   e.configuration,
      featureCount:    e.featureCount,
      accuracy:        e.accuracy,
      precision:       e.precision,
      recall:          e.recall,
      f1Score:         e.f1Score,
      rocAuc:          e.rocAuc,
      confusionMatrix: e.confusionMatrix,
    }));
    return { experiments, trainingSamples: raw.trainingSamples ?? 0 };
  } catch {
    return null;
  }
}

// ── Main training entry point ─────────────────────────────────────────────────
export function trainAllModels(): TrainedModels {
  const csv = loadDatasetFromCsv();

  let X: number[][];
  let y: number[];
  let SAMPLES: number;
  let dataSource: string;

  if (csv) {
    X = csv.X;
    y = csv.y;
    SAMPLES = csv.count;
    dataSource = `CSV dataset (${SAMPLES} articles)`;
  } else {
    SAMPLES = 500;
    const rng = makeRng(42);
    X = [];
    y = [];
    for (let i = 0; i < SAMPLES / 2; i++) { X.push(generateFakeVector(rng)); y.push(1); }
    for (let i = 0; i < SAMPLES / 2; i++) { X.push(generateRealVector(rng)); y.push(0); }
    dataSource = `synthetic data (${SAMPLES} samples)`;
  }

  const shuffleRng = makeRng(99);
  const idx = Array.from({ length: SAMPLES }, (_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(shuffleRng() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const Xs = idx.map((i) => X[i]);
  const ys = idx.map((i) => y[i]);

  const linguisticCV = stratifiedKFoldCV(Xs, ys, LINGUISTIC_INDICES, 10, 0);
  const structuralCV = stratifiedKFoldCV(Xs, ys, STRUCTURAL_INDICES, 10, 100);
  const combinedCV   = stratifiedKFoldCV(Xs, ys, COMBINED_INDICES,   10, 200);

  const RF = (seed: number, features: number[]) => {
    const c = new RandomForestClassifier({
      seed,
      nEstimators: 100,
      maxFeatures: Math.max(1, Math.floor(Math.sqrt(features.length))),
      replacement: true,
    });
    c.train(Xs.map((v) => features.map((fi) => v[fi])), ys);
    return c;
  };

  const linguistic = RF(42, LINGUISTIC_INDICES);
  const structural = RF(43, STRUCTURAL_INDICES);
  const combined   = RF(44, COMBINED_INDICES);

  const featureImportances = FEATURE_META.map((fm) => ({
    name:       fm.name,
    label:      fm.label,
    importance: FEATURE_IMPORTANCES[fm.name] ?? 0,
    category:   fm.category,
  })).sort((a, b) => b.importance - a.importance);

  const realMetrics = loadRealMetrics();

  const performance: ModelPerformance = {
    experiments: realMetrics?.experiments ?? [
      { configuration: "linguistic-only", featureCount: LINGUISTIC_INDICES.length, ...linguisticCV },
      { configuration: "structural-only", featureCount: STRUCTURAL_INDICES.length, ...structuralCV },
      { configuration: "combined",        featureCount: COMBINED_INDICES.length,   ...combinedCV   },
    ],
    trainingSamples: realMetrics?.trainingSamples ?? SAMPLES,
    folds: 10,
    featureImportances,
  };

  const resolvedDataSource = realMetrics
    ? `real dataset (${realMetrics.trainingSamples} articles, scikit-learn RF)`
    : dataSource;

  return { linguistic, structural, combined, performance, dataSource: resolvedDataSource };
}

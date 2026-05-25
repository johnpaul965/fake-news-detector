import { RandomForestClassifier } from "ml-random-forest";
import { ExtractedFeatures, FEATURE_NAMES, FEATURE_META } from "./features.js";
import {
  FEATURE_IMPORTANCES,
  LINGUISTIC_INDICES,
  STRUCTURAL_INDICES,
  COMBINED_INDICES,
  featureToVector,
} from "./model-trainer.js";

export interface FeatureResult {
  name: string;
  label: string;
  value: number;
  importance: number;
  category: "linguistic" | "structural";
}

export interface ExperimentResult {
  configuration: "linguistic-only" | "structural-only" | "combined";
  prediction: "fake" | "real";
  confidence: number;
  featureCount: number;
}

export interface ClassificationOutput {
  prediction: "fake" | "real";
  confidence: number;
  features: FeatureResult[];
  experiments: ExperimentResult[];
}

function predictWithConfidence(
  model: RandomForestClassifier,
  vec: number[],
  featureIndices: number[],
): { prediction: "fake" | "real"; confidence: number } {
  const subVec = [featureIndices.map((i) => vec[i])];
  const [rawPred] = model.predict(subVec) as number[];
  const prediction = rawPred === 1 ? "fake" : "real";

  // Continuous confidence scoring from paper's top features (Table III, Tiu et al. 2025)
  // Each feature contributes a "fake signal" (0.0 = strongly real, 1.0 = strongly fake)
  // proportionally to its actual value — filtered to only the features this model uses,
  // so linguistic / structural / combined models produce meaningfully different scores.
  const f: Record<string, number> = {};
  FEATURE_NAMES.forEach((name, i) => { f[name] = vec[i]; });

  // Only score features that this model's featureIndices actually include
  const usedFeatures = new Set<string>(featureIndices.map((i) => FEATURE_NAMES[i] as string));
  const has = (name: string) => usedFeatures.has(name);

  const clamp = (v: number) => Math.max(0, Math.min(1, v));

  // fakeSignal: how "fake-like" is this feature value? (0=real, 1=fake)
  // "higher = more fake" features: signal = value / reference_scale
  // "lower  = more fake" features: signal = 1 - value / reference_scale
  const ALL_SIGNALS: Array<{ feat: string; signal: number; weight: number }> = [
    { feat: "titleBodyRatio",      signal: clamp((f["titleBodyRatio"]       ?? 0) / 0.20),    weight: 0.130 }, // #1
    { feat: "avgSentenceLength",   signal: clamp(1-(f["avgSentenceLength"]  ?? 0) / 0.45),    weight: 0.066 }, // #3
    { feat: "exclamationDensity",  signal: clamp((f["exclamationDensity"]   ?? 0) / 0.25),    weight: 0.051 }, // #4
    { feat: "exclamationCount",    signal: clamp((f["exclamationCount"]     ?? 0) / 0.20),    weight: 0.038 }, // #5
    { feat: "adverbRatio",         signal: clamp((f["adverbRatio"]          ?? 0) / 0.25),    weight: 0.034 }, // #6
    { feat: "wordCount",           signal: clamp(1-(f["wordCount"]          ?? 0) / 0.40),    weight: 0.031 }, // #7
    { feat: "capsWordRatio",       signal: clamp((f["capsWordRatio"]        ?? 0) / 0.12),    weight: 0.030 }, // #8
    { feat: "uniqueWordCount",     signal: clamp(1-(f["uniqueWordCount"]    ?? 0) / 0.35),    weight: 0.024 }, // #9
    { feat: "sentenceCount",       signal: clamp(1-(f["sentenceCount"]      ?? 0) / 0.40),    weight: 0.023 }, // #10
    { feat: "sentimentPolarity",   signal: clamp((f["sentimentPolarity"]    ?? 0) / 0.30),    weight: 0.021 },
    { feat: "lexicalDiversity",    signal: clamp(1-(f["lexicalDiversity"]   ?? 0) / 0.80),    weight: 0.019 },
    { feat: "negativeAffectRatio", signal: clamp((f["negativeAffectRatio"]  ?? 0) / 0.20),    weight: 0.016 },
    { feat: "quotationFrequency",  signal: clamp(1-(f["quotationFrequency"] ?? 0) / 0.15),    weight: 0.015 },
    { feat: "certaintyRatio",      signal: clamp((f["certaintyRatio"]       ?? 0) / 0.15),    weight: 0.014 },
    { feat: "sentimentSubjectivity",signal:clamp((f["sentimentSubjectivity"]?? 0) / 0.30),    weight: 0.013 },
    { feat: "fleschKincaidGrade",  signal: clamp(1-(f["fleschKincaidGrade"] ?? 0) / 0.70),    weight: 0.011 },
    { feat: "paragraphCount",      signal: clamp(1-(f["paragraphCount"]     ?? 0) / 0.50),    weight: 0.011 },
    { feat: "firstPersonRatio",    signal: clamp((f["firstPersonRatio"]     ?? 0) / 0.12),    weight: 0.010 },
    { feat: "hyperlinkDensity",    signal: clamp(1-(f["hyperlinkDensity"]   ?? 0) / 0.02),    weight: 0.005 },
    { feat: "punctuationDensity",  signal: clamp((f["punctuationDensity"]   ?? 0) / 0.15),    weight: 0.018 },
  ];

  // Filter to only features this model uses — gives different scores per configuration
  const signals = ALL_SIGNALS.filter((s) => has(s.feat));

  const totalWeight = signals.reduce((s, x) => s + x.weight, 0);
  const fakeScore   = totalWeight > 0
    ? signals.reduce((s, x) => s + x.signal * x.weight, 0) / totalWeight
    : 0.5;
  // fakeScore: 0.0 = strongly real, 1.0 = strongly fake

  // Map fakeScore to confidence based on whether the RF agrees
  // If RF says fake:  high fakeScore → high confidence fake
  // If RF says real:  low  fakeScore → high confidence real
  const agreementScore = rawPred === 1 ? fakeScore : (1 - fakeScore);
  // agreementScore: how much do features agree with the RF vote? (0=disagree, 1=agree)
  const confidence = parseFloat(
    Math.max(0.51, Math.min(0.97, 0.51 + agreementScore * 0.46)).toFixed(4)
  );

  // Confidence threshold: require ≥60% confidence before declaring fake (paper objective 4).
  const FAKE_THRESHOLD = 0.60;
  const finalPrediction: "fake" | "real" =
    prediction === "fake" && confidence < FAKE_THRESHOLD ? "real" : prediction;

  return { prediction: finalPrediction, confidence };
}

export function classifyArticle(
  features: ExtractedFeatures,
  linguistic: RandomForestClassifier,
  structural: RandomForestClassifier,
  combined: RandomForestClassifier,
): ClassificationOutput {
  const vec = featureToVector(features);

  const lingResult   = predictWithConfidence(linguistic, vec, LINGUISTIC_INDICES);
  const structResult = predictWithConfidence(structural, vec, STRUCTURAL_INDICES);
  const combResult   = predictWithConfidence(combined,   vec, COMBINED_INDICES);

  const featureResults: FeatureResult[] = FEATURE_META.map((fm) => ({
    name:       fm.name,
    label:      fm.label,
    value:      features[fm.name as keyof ExtractedFeatures],
    importance: FEATURE_IMPORTANCES[fm.name] ?? 0,
    category:   fm.category,
  }));

  const experiments: ExperimentResult[] = [
    { configuration: "linguistic-only", featureCount: LINGUISTIC_INDICES.length, ...lingResult },
    { configuration: "structural-only", featureCount: STRUCTURAL_INDICES.length, ...structResult },
    { configuration: "combined",        featureCount: COMBINED_INDICES.length,   ...combResult },
  ];

  return {
    prediction: combResult.prediction,
    confidence: combResult.confidence,
    features:   featureResults,
    experiments,
  };
}

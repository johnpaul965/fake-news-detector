"""
Random Forest Trainer
======================
Trains 3 Random Forest classifiers using 10-fold stratified cross-validation:
  1. Linguistic features only
  2. Structural features only
  3. Combined features

Reads features.csv (output of features.py), trains the models, outputs:
  - model_results.json  — real metrics + feature importances (used by the API)
  - models/             — saved .pkl model files

Usage:
  python train.py

Requirements:
  pip install pandas scikit-learn joblib
"""

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature group definitions  (must match features.py output columns)
# ---------------------------------------------------------------------------

LINGUISTIC_FEATURES = [
    "sentiment_polarity",
    "sentiment_subjectivity",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
    "smog_index",
    "automated_readability_index",
    "lexical_diversity_ttr",
    "total_word_count",
    "unique_word_count",
    "certainty_word_ratio",
    "first_person_ratio",
    "emotional_word_ratio",
    "noun_ratio",
    "verb_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "exclamation_count",
    "question_mark_count",
    "exclamation_ratio",
    "all_caps_ratio",
]

STRUCTURAL_FEATURES = [
    "paragraph_count",
    "punctuation_density",
    "avg_sentence_length",
    "title_body_ratio",
    "quotation_ratio",
    "capitalization_ratio",
    "sentence_count",
]

RF_PARAMS = {
    "n_estimators": 100,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

K_FOLDS = 10
MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate_cv(X: np.ndarray, y: np.ndarray, label: str) -> dict:
    """Run stratified 10-fold CV and return averaged metrics."""
    log.info(f"  Running {K_FOLDS}-fold CV on {label} ({X.shape[1]} features) ...")

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    clf = RandomForestClassifier(**RF_PARAMS)

    all_true = []
    all_pred = []
    all_prob = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_clf = RandomForestClassifier(**{**RF_PARAMS, "random_state": 42 + fold})
        fold_clf.fit(X_train, y_train)
        preds = fold_clf.predict(X_test)
        probs = fold_clf.predict_proba(X_test)[:, 1]

        all_true.extend(y_test)
        all_pred.extend(preds)
        all_prob.extend(probs)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_prob  = np.array(all_prob)

    acc  = accuracy_score(all_true, all_pred)
    prec = precision_score(all_true, all_pred, zero_division=0)
    rec  = recall_score(all_true, all_pred, zero_division=0)
    f1   = f1_score(all_true, all_pred, zero_division=0)
    auc  = roc_auc_score(all_true, all_prob)
    cm   = confusion_matrix(all_true, all_pred).tolist()

    log.info(f"    Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  ROC-AUC={auc:.4f}")

    return {
        "accuracy":        round(acc,  4),
        "precision":       round(prec, 4),
        "recall":          round(rec,  4),
        "f1Score":         round(f1,   4),
        "rocAuc":          round(auc,  4),
        "confusionMatrix": cm,
    }


def train_final(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X, y)
    return clf


def feature_importances_dict(clf: RandomForestClassifier, names: list[str]) -> list[dict]:
    importances = clf.feature_importances_
    pairs = sorted(zip(names, importances), key=lambda x: -x[1])
    return [{"name": n, "importance": round(float(v), 6)} for n, v in pairs]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    features_path = Path("features.csv")
    if not features_path.exists():
        log.error("features.csv not found — run features.py first!")
        return

    log.info(f"Loading features from {features_path} ...")
    df = pd.read_csv(features_path)
    log.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Encode label
    le = LabelEncoder()
    y = le.fit_transform(df["label"])   # fake=0, real=1  (alphabetical)
    fake_label = int(le.transform(["fake"])[0])
    real_label = int(le.transform(["real"])[0])
    log.info(f"  Label encoding: fake={fake_label}, real={real_label}")
    log.info(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Identify available feature columns
    tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]

    ling_cols     = [c for c in LINGUISTIC_FEATURES if c in df.columns]
    struct_cols   = [c for c in STRUCTURAL_FEATURES if c in df.columns]
    combined_cols = ling_cols + struct_cols + tfidf_cols

    log.info(f"\n  Linguistic features  : {len(ling_cols)}")
    log.info(f"  Structural features  : {len(struct_cols)}")
    log.info(f"  TF-IDF features      : {len(tfidf_cols)}")
    log.info(f"  Combined features    : {len(combined_cols)}\n")

    X_ling   = df[ling_cols].fillna(0).values
    X_struct = df[struct_cols].fillna(0).values
    X_comb   = df[combined_cols].fillna(0).values

    # ── Cross-validation ─────────────────────────────────────────────────────
    log.info("=== Cross-Validation ===")
    ling_metrics   = evaluate_cv(X_ling,   y, "linguistic-only")
    struct_metrics = evaluate_cv(X_struct, y, "structural-only")
    comb_metrics   = evaluate_cv(X_comb,   y, "combined")

    # ── Train final models on full data ──────────────────────────────────────
    log.info("\n=== Training Final Models ===")
    MODELS_DIR.mkdir(exist_ok=True)

    log.info("  Training linguistic model ...")
    ling_clf = train_final(X_ling, y)
    joblib.dump(ling_clf, MODELS_DIR / "linguistic.pkl")

    log.info("  Training structural model ...")
    struct_clf = train_final(X_struct, y)
    joblib.dump(struct_clf, MODELS_DIR / "structural.pkl")

    log.info("  Training combined model ...")
    comb_clf = train_final(X_comb, y)
    joblib.dump(comb_clf, MODELS_DIR / "combined.pkl")

    log.info(f"  Models saved to {MODELS_DIR}/")

    # ── Feature importances ──────────────────────────────────────────────────
    ling_importances   = feature_importances_dict(ling_clf,   ling_cols)
    struct_importances = feature_importances_dict(struct_clf, struct_cols)
    comb_importances   = feature_importances_dict(comb_clf,   combined_cols)

    # Map to the JS feature names used by the API
    JS_NAME_MAP = {
        "sentiment_polarity":           "sentimentPolarity",
        "sentiment_subjectivity":       "sentimentSubjectivity",
        "flesch_reading_ease":          "readabilityScore",
        "flesch_kincaid_grade":         "readabilityGrade",
        "gunning_fog":                  "gunningFog",
        "smog_index":                   "smogIndex",
        "automated_readability_index":  "ariScore",
        "lexical_diversity_ttr":        "lexicalDiversity",
        "total_word_count":             "totalWordCount",
        "unique_word_count":            "uniqueWordCount",
        "certainty_word_ratio":         "certaintyRatio",
        "first_person_ratio":           "firstPersonRatio",
        "emotional_word_ratio":         "negativeAffectRatio",
        "noun_ratio":                   "nounRatio",
        "verb_ratio":                   "verbRatio",
        "adjective_ratio":              "adjectiveRatio",
        "adverb_ratio":                 "adverbRatio",
        "exclamation_count":            "exclamationCount",
        "question_mark_count":          "questionCount",
        "exclamation_ratio":            "exclamationDensity",
        "all_caps_ratio":               "capsWordRatio",
        "paragraph_count":              "paragraphCount",
        "punctuation_density":          "punctuationDensity",
        "avg_sentence_length":          "avgSentenceLength",
        "title_body_ratio":             "titleBodyRatio",
        "quotation_ratio":              "quotationFrequency",
        "capitalization_ratio":         "capitalizationRatio",
        "sentence_count":               "sentenceCount",
    }

    def remap(importances: list[dict]) -> list[dict]:
        remapped = []
        for item in importances:
            js_name = JS_NAME_MAP.get(item["name"], item["name"])
            remapped.append({"name": js_name, "importance": item["importance"]})
        return remapped

    # Build output
    results = {
        "trainingSamples": len(df),
        "folds": K_FOLDS,
        "labelEncoding": {"fake": fake_label, "real": real_label},
        "experiments": [
            {
                "configuration": "linguistic-only",
                "featureCount": len(ling_cols),
                **ling_metrics,
            },
            {
                "configuration": "structural-only",
                "featureCount": len(struct_cols),
                **struct_metrics,
            },
            {
                "configuration": "combined",
                "featureCount": len(combined_cols),
                **comb_metrics,
            },
        ],
        "featureImportances": {
            "linguistic": remap(ling_importances),
            "structural": remap(struct_importances),
            "combined":   remap(comb_importances[:30]),  # top 30
        },
    }

    out_path = Path("model_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved → {out_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    log.info("\n=== Final Results Summary ===")
    for exp in results["experiments"]:
        log.info(
            f"  [{exp['configuration']:20s}]  "
            f"Acc={exp['accuracy']:.4f}  "
            f"F1={exp['f1Score']:.4f}  "
            f"ROC-AUC={exp['rocAuc']:.4f}"
        )

    log.info("\n=== Top 10 Combined Feature Importances ===")
    for item in comb_importances[:10]:
        log.info(f"  {item['name']:40s}  {item['importance']:.4f}")

    log.info("\nDone! Copy model_results.json to artifacts/api-server/data/")


if __name__ == "__main__":
    main()

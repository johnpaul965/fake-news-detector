"""
Feature Extraction Script
=========================
Reads dataset.csv and computes all linguistic and structural features
described in the research paper, then outputs features.csv ready for
the Random Forest classifier.

Linguistic Features:
  - TF-IDF scores (top 100 terms)
  - Sentiment polarity & subjectivity
  - Readability metrics (Flesch, Gunning Fog, SMOG, ARI)
  - Lexical diversity (TTR, word counts)
  - Psycholinguistic indicators (certainty, first-person, emotional words)
  - Part-of-speech frequency (noun, verb, adjective, adverb ratios)
  - Stylistic patterns (exclamation marks, all-caps ratio)

Structural Features:
  - Quotation frequency
  - Capitalization behavior
  - Average sentence length
  - Title-body length ratio
  - Paragraph count
  - Punctuation density
  - Sentence count

Output: features.csv  (all features + label column)

Usage:
  pip install pandas nltk textblob textstat scikit-learn
  python features.py
"""

import re
import logging
import string
import multiprocessing
import os
from pathlib import Path

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from textblob import TextBlob
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer

for resource in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

INPUT_FILE = "dataset.csv"
OUTPUT_FILE = "features.csv"
TFIDF_MAX_FEATURES = 100

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

CERTAINTY_WORDS = {
    "always", "never", "definitely", "certainly", "absolutely", "undoubtedly",
    "clearly", "obviously", "proven", "fact", "truth", "lie", "false", "exposed",
    "confirmed", "guaranteed", "must", "impossible", "unbelievable",
}
FIRST_PERSON = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
}
EMOTIONAL_WORDS = {
    "shocking", "outrageous", "unbelievable", "amazing", "horrible", "terrible",
    "disgusting", "scandal", "crisis", "urgent", "alert", "warning", "danger",
    "fear", "hate", "love", "angry", "furious", "expose", "reveal", "secret",
    "breaking", "exclusive", "viral", "share",
}


# ---------------------------------------------------------------------------
# Boilerplate stripping
# ---------------------------------------------------------------------------

# Sentence-level patterns that are cookie banners, nav menus, footers, etc.
_BOILERPLATE_PATTERNS = [
    re.compile(r, re.IGNORECASE) for r in [
        r"\bprivacy policy\b",
        r"\bterms (of (use|service)|and conditions)\b",
        r"\bcookie(s| policy| notice| consent)\b",
        r"\bwe use cookies\b",
        r"\bby (using|continuing|browsing) (this|our) (site|website|service)\b",
        r"\byou agree to (our|the)\b",
        r"\ball rights reserved\b",
        r"\bcopyright\s*©",
        r"\bsubscribe (now|to|for)\b",
        r"\bnewsletter\b",
        r"\bclick here to (read|see|view|subscribe|download)\b",
        r"\bread (more|full (article|story))\b",
        r"\bfollow us on\b",
        r"\bshare this (article|story|post)\b",
        r"\brelated (articles?|stories|news)\b",
        r"\badvertisement\b",
        r"\bsponsored (content|post|by)\b",
        r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b",   # email addresses
        r"^\s*tags?\s*:",                                      # "Tags: politics, ..."
        r"^\s*category\s*:",
        r"^\s*source\s*:",
        r"^\s*photo\s*(credit|by|courtesy)\s*:",
    ]
]


def strip_boilerplate(text: str) -> str:
    """Remove cookie banners, footers, nav text, and other non-article boilerplate."""
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    clean_sentences = []
    for s in sentences:
        s = s.strip()
        if not s or len(s) < 15:
            continue
        if any(pat.search(s) for pat in _BOILERPLATE_PATTERNS):
            continue
        clean_sentences.append(s)
    return " ".join(clean_sentences)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    return [
        LEMMATIZER.lemmatize(t)
        for t in tokens
        if t.isalpha() and t not in STOP_WORDS
    ]


# ---------------------------------------------------------------------------
# Linguistic Features
# ---------------------------------------------------------------------------

def sentiment_features(text: str) -> dict:
    blob = TextBlob(text)
    return {
        "sentiment_polarity": blob.sentiment.polarity,
        "sentiment_subjectivity": blob.sentiment.subjectivity,
    }


def readability_features(text: str) -> dict:
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
    }


def lexical_diversity_features(text: str) -> dict:
    tokens = word_tokenize(text.lower())
    words = [t for t in tokens if t.isalpha()]
    total = len(words)
    unique = len(set(words))
    return {
        "lexical_diversity_ttr": unique / total if total > 0 else 0,
        "total_word_count": total,
        "unique_word_count": unique,
    }


def psycholinguistic_features(text: str) -> dict:
    tokens = word_tokenize(text.lower())
    words = [t for t in tokens if t.isalpha()]
    total = len(words) or 1
    return {
        "certainty_word_ratio": sum(1 for w in words if w in CERTAINTY_WORDS) / total,
        "first_person_ratio": sum(1 for w in words if w in FIRST_PERSON) / total,
        "emotional_word_ratio": sum(1 for w in words if w in EMOTIONAL_WORDS) / total,
    }


def pos_features(text: str) -> dict:
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    total = len(tags) or 1
    return {
        "noun_ratio": sum(1 for _, t in tags if t.startswith("NN")) / total,
        "verb_ratio": sum(1 for _, t in tags if t.startswith("VB")) / total,
        "adjective_ratio": sum(1 for _, t in tags if t.startswith("JJ")) / total,
        "adverb_ratio": sum(1 for _, t in tags if t.startswith("RB")) / total,
    }


def stylistic_features(text: str) -> dict:
    tokens = word_tokenize(text)
    words = [t for t in tokens if t.isalpha()]
    total_words = len(words) or 1
    exclamations = text.count("!")
    questions = text.count("?")
    all_caps = sum(1 for w in words if w.isupper() and len(w) > 1)
    return {
        "exclamation_count": exclamations,
        "question_mark_count": questions,
        "exclamation_ratio": exclamations / total_words,
        "all_caps_ratio": all_caps / total_words,
    }


# ---------------------------------------------------------------------------
# Structural Features
# ---------------------------------------------------------------------------

def structural_features(title: str, text: str) -> dict:
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    alpha_words = [w for w in words if w.isalpha()]
    total_words = len(alpha_words) or 1
    total_chars = len(text) or 1
    total_sentences = len(sentences) or 1

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    paragraph_count = max(len(paragraphs), 1)

    punct_count = sum(1 for c in text if c in string.punctuation)

    title_len = len(title.split())
    body_len = len(text.split()) or 1

    quote_count = text.count('"') + text.count("\u201c") + text.count("\u201d")

    mid_caps = sum(
        1 for w in alpha_words if w[0].isupper() and w.lower() not in STOP_WORDS
    )

    return {
        "paragraph_count": paragraph_count,
        "punctuation_density": punct_count / total_chars,
        "avg_sentence_length": total_words / total_sentences,
        "title_body_ratio": title_len / body_len,
        "quotation_ratio": quote_count / total_words,
        "capitalization_ratio": mid_caps / total_words,
        "sentence_count": total_sentences,
    }


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _worker_init():
    """Ensure NLTK data is available in each worker process."""
    for resource in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger",
                     "punkt_tab", "averaged_perceptron_tagger_eng"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


def _extract_row(args: tuple) -> dict:
    """Top-level function (picklable) — extracts all features for one article."""
    title, text, label, source, source_dataset = args
    combined = title + " " + text
    feats: dict = {}
    feats.update(sentiment_features(combined))
    feats.update(readability_features(text))
    feats.update(lexical_diversity_features(text))
    feats.update(psycholinguistic_features(combined))
    feats.update(pos_features(text))
    feats.update(stylistic_features(combined))
    feats.update(structural_features(title, text))
    feats["label"]          = label
    feats["source"]         = source
    feats["source_dataset"] = source_dataset
    return feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        log.error(f"Input file not found: {INPUT_FILE}")
        log.error("Run scraper.py first to generate the dataset.")
        return

    log.info(f"Loading dataset from {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    log.info(f"  Loaded {len(df)} articles")
    log.info(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    df = df.dropna(subset=["title", "text"])
    df["title"]          = df["title"].astype(str)
    df["text"]           = df["text"].astype(str)
    df["source_dataset"] = df.get("source_dataset", pd.Series(["unknown"] * len(df))).fillna("unknown").astype(str)
    df["text"] = df["text"].apply(strip_boilerplate)
    df = df[df["text"].str.len() >= 50]
    log.info(f"  After dropping nulls + boilerplate strip: {len(df)} articles\n")

    # Build argument tuples for parallel workers
    args_list = [
        (
            row["title"],
            row["text"],
            row["label"],
            row.get("source", ""),
            row.get("source_dataset", "unknown"),
        )
        for _, row in df.iterrows()
    ]

    n_workers = max(1, os.cpu_count() or 1)
    log.info(f"Extracting features using {n_workers} parallel workers ...")

    chunk_size = max(1, len(args_list) // (n_workers * 4))
    with multiprocessing.Pool(processes=n_workers, initializer=_worker_init) as pool:
        feature_rows = pool.map(_extract_row, args_list, chunksize=chunk_size)

    log.info(f"  Feature extraction complete — {len(feature_rows)} rows")

    features_df = pd.DataFrame(feature_rows)
    non_tfidf_count = len(features_df.columns) - 3  # exclude label, source, source_dataset

    log.info(f"\nComputing TF-IDF (top {TFIDF_MAX_FEATURES} features) ...")
    cleaned_texts = df["text"].apply(lambda t: " ".join(preprocess(t)))
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    tfidf_matrix = tfidf.fit_transform(cleaned_texts).toarray()
    tfidf_cols = [f"tfidf_{term}" for term in tfidf.get_feature_names_out()]
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_cols, index=df.index)

    final_df = pd.concat(
        [features_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)],
        axis=1,
    )

    final_df.to_csv(OUTPUT_FILE, index=False)

    scraped_n    = (final_df["source_dataset"] == "scraped").sum()
    fernandez_n  = (final_df["source_dataset"] == "fernandez2019").sum()

    log.info(f"\n=== Summary ===")
    log.info(f"  Articles processed   : {len(final_df)}")
    log.info(f"    scraped            : {scraped_n}")
    log.info(f"    fernandez2019      : {fernandez_n}")
    log.info(f"  Linguistic/structural: {non_tfidf_count} features")
    log.info(f"  TF-IDF features      : {TFIDF_MAX_FEATURES}")
    log.info(f"  Total feature columns: {len(final_df.columns) - 3}")
    log.info(f"  Saved → {OUTPUT_FILE}")
    log.info(f"\nDone! Use features.csv as input to train the Random Forest model.")


if __name__ == "__main__":
    main()

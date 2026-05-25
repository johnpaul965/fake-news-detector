#!/usr/bin/env bash
# retrain.sh — one command to re-extract features, retrain the model, and deploy it.
# Run this from the dataset/ folder after the scraper has collected more articles.
#
# Usage:
#   cd dataset && bash retrain.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo " Fake News Detector — Retrain Pipeline"
echo "=========================================="

# Step 1: Feature extraction
echo ""
echo "[1/3] Extracting features from dataset.csv..."
python3 features.py
echo "      Done — features.csv updated."

# Step 2: Train models
echo ""
echo "[2/3] Training Random Forest (10-fold CV, 3 configurations)..."
python3 train.py
echo "      Done — model_results.json updated."

# Step 3: Deploy to API server
echo ""
echo "[3/3] Deploying results to API server..."
cp model_results.json ../artifacts/api-server/data/model_results.json
echo "      Done — ../artifacts/api-server/data/model_results.json replaced."

echo ""
echo "=========================================="
echo " Retrain complete!"
echo " --> Restart the 'API Server' workflow to"
echo "     load the new model results."
echo "=========================================="

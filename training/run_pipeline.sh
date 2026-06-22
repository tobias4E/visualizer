#!/usr/bin/env bash
# Full training pipeline for the EV charger YOLOv8 model.
# Run from inside the training/ directory:  cd training && bash run_pipeline.sh
#
# Prerequisites: uv (https://docs.astral.sh/uv/), CUDA optional (CPU works but is slow).
#
# Steps executed:
#   1. Create venv and install dependencies via uv
#   2. Download Open Images EV data  (skip if dataset/ already populated)
#   3. Train YOLOv8n
#   4. Evaluate on test split
#   5. Export to TF.js
#
# Labeling (step 2 in train.md) is a manual step — see train.md for Roboflow
# or CVAT instructions.  Run this script *after* your dataset is labeled and
# dataset/images/ + dataset/labels/ are populated.

set -euo pipefail

step() { echo -e "\n\033[0;36m=== Step $1 : $2 ===\033[0m"; }

# ── 1. Install dependencies ───────────────────────────────────────────────────
step 1 "Creating venv and installing dependencies with uv"
uv venv --python 3.11
uv pip install -r requirements.txt

# ── 2. Collect Open Images data (optional) ────────────────────────────────────
if [ ! -d "dataset/images/train" ]; then
    step 2 "Downloading Open Images EV charging station data"
    uv run python collect_openimages.py
    echo ""
    echo -e "\033[0;33mMANUAL STEP REQUIRED:\033[0m"
    echo "  1. Review and fix labels (Roboflow / CVAT — see train.md)"
    echo "  2. Make sure dataset/images/{train,val,test} and dataset/labels/{train,val,test} exist"
    echo "  3. Re-run this script once the dataset is ready"
    exit 0
else
    echo "dataset/images/train already exists — skipping data download."
fi

# ── 3. Train ──────────────────────────────────────────────────────────────────
step 3 "Training YOLOv8n"
uv run python train.py

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
step 4 "Evaluating on test split"
uv run python evaluate.py

# ── 5. Export to TF.js ────────────────────────────────────────────────────────
step 5 "Exporting to TF.js"
uv run python export.py

echo ""
echo -e "\033[0;32mPipeline complete.\033[0m"
echo "Next: copy runs/detect/ev_charger_v1/weights/best_web_model/ next to index.html"
echo "      then update the tf.loadGraphModel() call in index.html (see train.md step 7)"

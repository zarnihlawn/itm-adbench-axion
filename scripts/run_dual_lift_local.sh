#!/usr/bin/env bash
# Local CPU dual lift: unsup zoo/CV/NLP + semi graveyard/SVHN polish (skip AXION).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(pwd)/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-8}"
RUN_ID="${RUN_ID:-axion_beat_paper_dual_lift}"
LOG_DIR="results/${RUN_ID}/selection"
mkdir -p "$LOG_DIR" "results/${RUN_ID}/metrics" "results/${RUN_ID}/thesis"

GRAVEYARD=(speech ALOI optdigits Waveform smtp vertebral yeast PageBlocks SpamBase Stamps satellite)
NLP=(Agnews Imdb Amazon Yelp 20newsgroups)
CV=(SVHN CIFAR10 FashionMNIST MNIST-C MVTec-AD mnist)
SEMI_POLISH=(speech ALOI SVHN vertebral yeast Waveform)

echo "== catalog assert =="
"$PY" scripts/beat_paper_catalog.py --assert-catalog

echo "== Phase A: unsup perfection select (graveyard) =="
"$PY" -u scripts/beat_paper_select.py \
  --local-cpu --perfection --setting unsupervised --seed 111 \
  --datasets "${GRAVEYARD[@]}" 2>&1 | tee "$LOG_DIR/unsup_graveyard.log"

echo "== Phase A: unsup perfection select (NLP zoo, skip axion) =="
"$PY" -u scripts/beat_paper_select.py \
  --local-cpu --perfection --setting unsupervised --seed 111 \
  --datasets "${NLP[@]}" 2>&1 | tee "$LOG_DIR/unsup_nlp.log"

echo "== Phase A: unsup perfection select (CV) =="
"$PY" -u scripts/beat_paper_select.py \
  --local-cpu --perfection --setting unsupervised --seed 111 \
  --datasets "${CV[@]}" 2>&1 | tee "$LOG_DIR/unsup_cv.log"

echo "== Phase A: unsup perfection select (remaining local campaign) =="
"$PY" -u scripts/beat_paper_select.py \
  --local-cpu --perfection --setting unsupervised --seed 111 \
  2>&1 | tee "$LOG_DIR/unsup_full_campaign.log"

echo "== write-map =="
"$PY" scripts/beat_paper_select.py --write-map

echo "== Phase A: run unsup zoo winners into dual_lift =="
"$PY" -u scripts/beat_paper_run.py \
  --local-cpu \
  --settings unsupervised \
  --seeds 111 \
  --run-id "$RUN_ID" \
  --skip-axion \
  2>&1 | tee "$LOG_DIR/unsup_run.log"

echo "== Phase B: semi polish select (graveyard+SVHN) =="
"$PY" -u scripts/beat_paper_select.py \
  --local-cpu --perfection --setting semi-supervised --seed 111 \
  --datasets "${SEMI_POLISH[@]}" 2>&1 | tee "$LOG_DIR/semi_polish.log"

"$PY" scripts/beat_paper_select.py --write-map

echo "== Phase B: run semi polish into dual_lift =="
"$PY" -u scripts/beat_paper_run.py \
  --settings semi-supervised \
  --seeds 111 \
  --run-id "$RUN_ID" \
  --skip-axion \
  --datasets "${SEMI_POLISH[@]}" \
  --map results/axion_beat_paper_local/thesis/recipe_map.json \
  2>&1 | tee "$LOG_DIR/semi_polish_run.log"

echo "== Phase C/D: assemble + macro =="
"$PY" scripts/beat_paper_dual_lift_assemble.py --assemble --macro --write-map
"$PY" scripts/beat_paper_macro.py --compare-paper --final --beyond-stage --run-id "$RUN_ID"

echo "DONE dual lift local path"

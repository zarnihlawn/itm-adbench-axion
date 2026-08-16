#!/usr/bin/env bash
# Dual-setting further lift on Vast/GPU (NLP SPEAR + locks). Do not use --skip-axion for NLP claim.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$(pwd)/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-8}"
RUN_ID="${RUN_ID:-axion_beat_paper_dual_lift}"

echo "== catalog assert =="
"$PY" scripts/beat_paper_catalog.py --assert-catalog

echo "== dual select semi NLP+locks perfection (AXION enabled) =="
"$PY" -u scripts/beat_paper_select.py \
  --setting semi-supervised \
  --perfection \
  --seed 111 \
  --datasets Agnews Imdb Amazon Yelp 20newsgroups celeba census cover fraud backdoor

echo "== dual select unsup NLP (AXION allowed for axion_roberta) =="
"$PY" -u scripts/beat_paper_select.py \
  --setting unsupervised \
  --perfection \
  --seed 111 \
  --datasets Agnews Imdb Amazon Yelp 20newsgroups

echo "== merge map =="
"$PY" scripts/beat_paper_select.py --write-map

echo "== run both settings seed 111 into dual_lift =="
"$PY" -u scripts/beat_paper_run.py \
  --full57 \
  --settings semi-supervised,unsupervised \
  --seeds 111 \
  --run-id "$RUN_ID" \
  --datasets Agnews Imdb Amazon Yelp 20newsgroups celeba census cover fraud backdoor

echo "== assemble + macro =="
"$PY" scripts/beat_paper_dual_lift_assemble.py --assemble --macro --write-map

echo "== beat_paper macro claim =="
"$PY" scripts/beat_paper_macro.py --compare-paper --final --beyond-stage --run-id "$RUN_ID"

echo "DONE dual lift GPU path"

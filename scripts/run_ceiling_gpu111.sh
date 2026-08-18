#!/usr/bin/env bash
# Back-compat wrapper. Prefer: bash scripts/run_seeds.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export RUN_ID="${RUN_ID:-axion_beat_paper_dual_lift_gpu}"
export SEEDS="${SEEDS:-111,222,333,444,555}"
exec bash scripts/run_ceiling.sh

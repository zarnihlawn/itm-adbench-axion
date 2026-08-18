#!/usr/bin/env bash
# Seed-111 only. Same ingredients as all-seed. For 5 seeds: bash scripts/run_seeds.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${FORCE_LOCAL:-0}" == "1" ]]; then
  echo "FORCE_LOCAL=1: throttled laptop config. Numbers will NOT match the ceiling." >&2
  exec bash scripts/run_seed111_full57_local.sh "$@"
fi

# CFG unset: run_ceiling.sh auto-picks gpu_beat_paper_3090.yaml when VRAM >= 20 GB.
export MAP="${MAP:-results/axion_beat_paper_dual_lift/thesis/recipe_map_57_beat.json}"
export RUN_ID="${RUN_ID:-axion_beat_paper_dual_lift_gpu}"
export SEEDS="${SEEDS:-111}"
exec bash scripts/run_ceiling.sh

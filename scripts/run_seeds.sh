#!/usr/bin/env bash
# Canonical full 57x2 for paper seeds. Same ingredients as the dual_lift ceiling.
#   bash scripts/run_seeds.sh              # 111,222,333,444,555
#   SEEDS=111 bash scripts/run_seeds.sh    # seed 111 only
#   SEEDS=222,333 bash scripts/run_seeds.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${FORCE_LOCAL:-0}" == "1" ]]; then
  echo "FORCE_LOCAL=1 is seed-111 laptop debug only. Refusing all-seed local throttle." >&2
  exit 1
fi

# CFG unset: run_ceiling.sh auto-picks gpu_beat_paper_3090.yaml when VRAM >= 20 GB.
# Override only for debugging: CFG=configs/gpu_beat_paper.yaml bash scripts/run_seeds.sh
export MAP="${MAP:-results/axion_beat_paper_dual_lift/thesis/recipe_map_57_beat.json}"
export RUN_ID="${RUN_ID:-axion_beat_paper_dual_lift_gpu}"
export SEEDS="${SEEDS:-111,222,333,444,555}"
exec bash scripts/run_ceiling.sh

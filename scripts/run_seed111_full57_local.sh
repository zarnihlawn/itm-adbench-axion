#!/usr/bin/env bash
# DEPRECATED for claim numbers. Use: bash scripts/run_seed111.sh
# Only for FORCE_LOCAL=1 CPU debug (throttled 20k cap). Does not match the ceiling.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${PY:-}" && -x "${PY}" ]]; then
  :
elif [[ -x "$(pwd)/.venv/bin/python" ]]; then
  PY="$(pwd)/.venv/bin/python"
elif [[ -x /home/zarnihlawn/Desktop/ITM/project/.venv/bin/python ]]; then
  PY=/home/zarnihlawn/Desktop/ITM/project/.venv/bin/python
else
  PY="$(command -v python3)"
fi
export PY
export PYTHONPATH=src:scripts

NPROC="$(nproc)"
# ~90% of logical CPUs, leave at least 2 for UI/OS
OMP_TARGET=$(( (NPROC * 90 + 99) / 100 ))
if (( OMP_TARGET > NPROC - 2 )); then
  OMP_TARGET=$(( NPROC - 2 ))
fi
if (( OMP_TARGET < 1 )); then
  OMP_TARGET=1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$OMP_TARGET}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$OMP_TARGET}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_TARGET}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_TARGET}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OMP_TARGET}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTHONUNBUFFERED=1

# Dense progress under tee/nohup
export AXION_VERBOSE=1
export AXION_VERBOSE_EVERY=1
export AXION_FORCE_PROGRESS=1

RUN_ID="${RUN_ID:-axion_beat_paper_dual_lift_local111}"
MAP="${MAP:-results/axion_beat_paper_dual_lift/thesis/recipe_map_57_beat.json}"
CFG="${CFG:-configs/local_seed111.yaml}"
LOG_DIR="results/${RUN_ID}/logs"
mkdir -p "$LOG_DIR" "results/${RUN_ID}/metrics" "results/${RUN_ID}/thesis"
LOG_FILE="${LOG_DIR}/full57_seed111_$(date +%Y%m%d_%H%M%S).log"

RESUME=0
if [[ "${1:-}" == "--resume" ]]; then
  RESUME=1
fi

# Symlink embeds from the sibling project tree if missing (CV/NLP recipes).
if [[ ! -e data/embeds_alt_whitened ]]; then
  if [[ -d ../project/axion/data/embeds_alt_whitened ]]; then
    ln -sfn ../../project/axion/data/embeds_alt_whitened data/embeds_alt_whitened
  fi
fi
if [[ ! -e data/embeds_alt ]]; then
  if [[ -d ../project/axion/data/embeds_alt ]]; then
    ln -sfn ../../project/axion/data/embeds_alt data/embeds_alt
  fi
fi

MEM_GB="$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)"
echo "=== local seed-111 full57 ==="
echo "host=$(hostname) nproc=${NPROC} OMP=${OMP_NUM_THREADS} RAM_GB≈${MEM_GB}"
echo "PY=${PY}"
echo "RUN_ID=${RUN_ID}"
echo "MAP=${MAP}"
echo "CFG=${CFG}"
echo "LOG=${LOG_FILE}"
echo "cuda=$("$PY" -c 'import torch; print(torch.cuda.is_available())')"
echo "adbench=$("$PY" -c 'from axion.config import load_config; from pathlib import Path; print(load_config(Path("'"${CFG}"'"))["paths"]["adbench_root"])')"

if [[ ! -f "$MAP" ]]; then
  echo "ERROR: frozen map missing: $MAP" >&2
  exit 1
fi

"$PY" scripts/beat_paper_catalog.py --assert-catalog

if [[ "$RESUME" -eq 0 ]]; then
  echo "Wiping prior metrics for ${RUN_ID} (fresh override)"
  rm -f "results/${RUN_ID}/metrics/"*__111__*.json 2>/dev/null || true
fi

# nice + ionice: stay responsive for desktop
# Do NOT use --skip-axion: claim map has ~34 axion/edge_rare slots.
set +e
nice -n 5 ionice -c2 -n5 \
  "$PY" -u scripts/beat_paper_run.py \
    --config "$CFG" \
    --map "$MAP" \
    --full57 \
    --settings unsupervised,semi-supervised \
    --seeds 111 \
    --run-id "$RUN_ID" \
    2>&1 | tee -a "$LOG_FILE"
RC=${PIPESTATUS[0]}
set -e

echo "== assemble macro on ${RUN_ID} ==" | tee -a "$LOG_FILE"
"$PY" scripts/beat_paper_dual_lift_assemble.py --macro 2>&1 | tee -a "$LOG_FILE" || true

echo "DONE rc=${RC} log=${LOG_FILE}"
exit "$RC"

#!/usr/bin/env bash
# Ceiling-ingredient rescore: frozen map + gpu_beat_paper + whitened embeds + AXION on.
# Default: all paper seeds 111,222,333,444,555 into one run id (resume-safe).
# Auto-picks gpu_beat_paper_3090.yaml when VRAM >= 20 GB (override with CFG=).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

export PYTHONPATH=src:scripts
export AXION_VERBOSE=1 AXION_FORCE_PROGRESS=1 PYTHONUNBUFFERED=1
export AXION_EPOCH_BATCH_BAR="${AXION_EPOCH_BATCH_BAR:-1}"
export TQDM_MININTERVALS="${TQDM_MININTERVALS:-0.05}"

# Max-HW Mode B: burn almost all logical CPUs on the single GPU job.
_ncpu="$(nproc 2>/dev/null || echo 8)"
_omp_default="$_ncpu"
if (( _ncpu >= 4 )); then
  _omp_default=$((_ncpu - 2))
elif (( _ncpu >= 2 )); then
  _omp_default=$((_ncpu - 1))
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$_omp_default}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"

if [[ -n "${PY:-}" && -x "${PY}" ]]; then
  :
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
elif [[ -x /home/zarnihlawn/Desktop/ITM/project/.venv/bin/python ]]; then
  PY=/home/zarnihlawn/Desktop/ITM/project/.venv/bin/python
else
  PY="$(command -v python3)"
fi
export PY

_pick_cfg() {
  if [[ -n "${CFG:-}" ]]; then
    echo "$CFG"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local vram_mb
    vram_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    if [[ -n "$vram_mb" && "$vram_mb" -ge 20000 ]]; then
      echo "configs/gpu_beat_paper_3090.yaml"
      return
    fi
  fi
  echo "configs/gpu_beat_paper.yaml"
}

CFG="$(_pick_cfg)"
MAP="${MAP:-results/axion_beat_paper_dual_lift/thesis/recipe_map_57_beat.json}"
RUN_ID="${RUN_ID:-axion_beat_paper_dual_lift_gpu}"
SEEDS="${SEEDS:-111,222,333,444,555}"
SEED111_SRC="${SEED111_SRC:-results/axion_beat_paper_dual_lift/metrics}"
OUT="results/${RUN_ID}/metrics"
LOG_DIR="results/${RUN_ID}/logs"
mkdir -p "$OUT" "$LOG_DIR" "results/${RUN_ID}/thesis"
LOG="${LOG_DIR}/seeds_$(date +%Y%m%d_%H%M%S).log"

_mem_gb="$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo '?')"
_gpu_line=""
if command -v nvidia-smi >/dev/null 2>&1; then
  _gpu_line="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)"
fi
echo "MAXHW nproc=${_ncpu} OMP=${OMP_NUM_THREADS} RAM_GB≈${_mem_gb} GPU=${_gpu_line:-none} CFG=${CFG} SEEDS=${SEEDS}"

bash scripts/verify_ceiling_parity.sh --config-only
bash scripts/vast_ceiling_setup.sh

export RUN_ID SEEDS SEED111_SRC FRESH="${FRESH:-0}" CFG

"$PY" - <<'PY'
import os, shutil
from pathlib import Path

run_id = Path("results") / os.environ.get("RUN_ID", "axion_beat_paper_dual_lift_gpu")
out = run_id / "metrics"
out.mkdir(parents=True, exist_ok=True)
src = Path(os.environ.get("SEED111_SRC", "results/axion_beat_paper_dual_lift/metrics"))
fresh = os.environ.get("FRESH", "0") == "1"
copied = 0
skipped = 0
if src.is_dir() and src.resolve() != out.resolve():
    for p in src.glob("*__111__*.json"):
        dest = out / p.name
        if fresh or not dest.exists():
            shutil.copy2(p, dest)
            copied += 1
        else:
            skipped += 1
print(f"seed111_copied={copied} seed111_skipped={skipped} fresh={int(fresh)} out={out}")
PY

bash scripts/verify_ceiling_parity.sh --seed111-only || {
  echo "ERROR: seed111 parity failed after copy. Use FRESH=1 to recopy from SEED111_SRC." >&2
  exit 1
}

echo "== rescoring seeds=${SEEDS} run_id=${RUN_ID} (skips existing metric files) ==" | tee -a "$LOG"
set +e
"$PY" -u scripts/beat_paper_run.py \
  --config "$CFG" \
  --map "$MAP" \
  --full57 \
  --seeds "$SEEDS" \
  --settings unsupervised,semi-supervised \
  --run-id "$RUN_ID" \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "== macro (does not overwrite frozen dual_lift) ==" | tee -a "$LOG"
"$PY" -u scripts/beat_paper_macro.py --compare-paper --run-id "$RUN_ID" 2>&1 | tee -a "$LOG" || true
echo "DONE rc=${RC} run_id=${RUN_ID} log=${LOG}"
exit "$RC"

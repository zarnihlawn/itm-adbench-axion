#!/usr/bin/env bash
# AXION ultimate final run (Table-1 fair, edge_rare freeze, ALL seeds).
#
# Prepare-only by default docs; THIS script is the real launch entry.
# Do not use smoke / gap-close YAMLs here.
#
# Usage:
#   bash scripts/run_final.sh assert
#   bash scripts/run_final.sh probe12          # 12 × 2 × 5
#   bash scripts/run_final.sh full57           # 57 × 2 × 5
#   bash scripts/run_final.sh all              # assert → probe12 → gates → full57
#   bash scripts/run_final.sh gates            # Track C on axion_final_12
#   bash scripts/run_final.sh status
#
# Env:
#   AXION_FINAL_CFG=configs/gpu_final.yaml
#   AXION_SKIP_EXISTING=1          # resume (default 1)
#   AXION_MAX_TRAIN_SAMPLES=50000
#   OMP_NUM_THREADS=               # default: nproc-2 (Mode B max on rented box)
#   AXION_FINAL_SOFT_GATES=1       # default: gates warn but do not abort all
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Dense nested progress + stage banners (works under tee/nohup)
export AXION_FORCE_PROGRESS="${AXION_FORCE_PROGRESS:-1}"
export AXION_VERBOSE="${AXION_VERBOSE:-1}"
export AXION_VERBOSE_EVERY="${AXION_VERBOSE_EVERY:-1}"
export AXION_EPOCH_BATCH_BAR="${AXION_EPOCH_BATCH_BAR:-1}"
# Keep tqdm bars stable when stdout is piped
export TQDM_MININTERVALS="${TQDM_MININTERVALS:-0.05}"

# Max-HW Mode B: burn almost all logical CPUs on the single GPU job.
# Default OMP = nproc-2 (leave headroom for OS + DataLoader workers).
# Target rental: RTX A4000 16GB / ~8 CPU / ~64 GiB (instance 47448156). Override via env.
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

CFG="${AXION_FINAL_CFG:-configs/gpu_final.yaml}"
MODE="${1:-help}"
MAX_TRAIN="${AXION_MAX_TRAIN_SAMPLES:-50000}"
SKIP_EXISTING="${AXION_SKIP_EXISTING:-1}"
SOFT_GATES="${AXION_FINAL_SOFT_GATES:-1}"
SEEDS=(111 222 333 444 555)
RUN12="axion_final_12"
RUN57="axion_final"
LOG_DIR="results/axion_final_logs"
mkdir -p "$LOG_DIR"

_mem_gb="$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo '?')"
echo "MAXHW nproc=${_ncpu} OMP=${OMP_NUM_THREADS} RAM_GB≈${_mem_gb} workers_cfg=from_yaml CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}"
echo "FINAL cfg=$CFG mode=$MODE skip_existing=$SKIP_EXISTING"
echo "PROGRESS AXION_FORCE_PROGRESS=${AXION_FORCE_PROGRESS} AXION_VERBOSE=${AXION_VERBOSE} AXION_VERBOSE_EVERY=${AXION_VERBOSE_EVERY} AXION_EPOCH_BATCH_BAR=${AXION_EPOCH_BATCH_BAR}"

require_cuda() {
  python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA required for ultimate final (use Vast GPU box)"
props = torch.cuda.get_device_properties(0)
vram_gb = round(props.total_memory / (1024**3), 1)
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM_GB:", vram_gb)
# Soft warn if under-provisioned for full57 CV paths
if vram_gb < 11.5:
    print("WARN: VRAM <12GB; full57 CV variants may OOM (prefer 16GB)")
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  fi
}

do_assert() {
  echo "== ASSERT freeze"
  python scripts/assert_final_config.py --config "$CFG"
}

do_manifest() {
  local run_id="$1"
  local phase="$2"
  python scripts/write_final_manifest.py --config "$CFG" --run-id "$run_id" --phase "$phase"
}

do_probe12() {
  require_cuda
  do_assert
  do_manifest "$RUN12" start
  echo "== PROBE12 12×2×5 → results/$RUN12"
  local skip_flag=()
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    skip_flag=(--skip-existing)
  fi
  python scripts/run_probe.py \
    --config "$CFG" \
    --model axion \
    --protocol paper \
    --all-probe \
    --seeds "${SEEDS[@]}" \
    --max-train-samples "$MAX_TRAIN" \
    --max-variants 0 \
    --loop-log \
    --run-id "$RUN12" \
    "${skip_flag[@]}" \
    2>&1 | tee -a "$LOG_DIR/probe12.log"
  do_manifest "$RUN12" probe12
  echo "== PROBE12 done"
}

do_gates() {
  echo "== GATES Track C on $RUN12 (dual-track; prior A/B skipped for final)"
  set +e
  python scripts/check_probe_gate.py \
    --config "$CFG" \
    --run-id "$RUN12" \
    --skip-prior-gates
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    if [[ "$SOFT_GATES" == "1" ]]; then
      echo "WARN: probe gate rc=$rc (soft; continuing). Expect ~55.45 semi PR vs paper 61.36."
    else
      echo "FATAL: probe gate failed and AXION_FINAL_SOFT_GATES=0"
      exit "$rc"
    fi
  fi
}

do_full57() {
  require_cuda
  do_assert
  do_manifest "$RUN57" start
  echo "== FULL57 57×2×5 → results/$RUN57"
  local skip_flag=()
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    skip_flag=(--skip-existing)
  fi
  python scripts/run_full57.py \
    --config "$CFG" \
    --model axion \
    --protocol paper \
    --run-id "$RUN57" \
    --seeds "${SEEDS[@]}" \
    --max-train-samples "$MAX_TRAIN" \
    "${skip_flag[@]}" \
    2>&1 | tee -a "$LOG_DIR/full57.log"
  do_manifest "$RUN57" full57
  echo "== FULL57 done → results/$RUN57/compare_to_ddae.json"
}

do_status() {
  echo "== STATUS"
  python scripts/assert_final_config.py --config "$CFG" || true
  for rid in "$RUN12" "$RUN57"; do
    local d="results/$rid"
    if [[ -d "$d/metrics" ]]; then
      local n
      n=$(find "$d/metrics" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
      echo "  $rid metrics_json=$n"
    else
      echo "  $rid MISSING"
    fi
    [[ -f "$d/probe_summary.json" ]] && echo "  $rid has probe_summary.json"
    [[ -f "$d/compare_to_ddae.json" ]] && echo "  $rid has compare_to_ddae.json"
    [[ -f "$d/MANIFEST.json" ]] && echo "  $rid MANIFEST phase=$(python -c "import json;print(json.load(open('$d/MANIFEST.json')).get('phase'))")"
  done
  if [[ -f "results/$RUN57/compare_to_ddae.json" ]]; then
    python - <<PY
import json
from pathlib import Path
s=json.loads(Path("results/$RUN57/compare_to_ddae.json").read_text())
print("MACRO", json.dumps(s.get("macro"), indent=2))
PY
  fi
}

do_all() {
  do_assert
  do_probe12
  do_gates
  do_full57
  do_manifest "$RUN57" done
  do_status
  echo "== ULTIMATE FINAL COMPLETE"
}

case "$MODE" in
  assert) do_assert ;;
  probe12|12) do_probe12 ;;
  gates) do_gates ;;
  full57|57) do_full57 ;;
  all|final) do_all ;;
  status) do_status ;;
  help|-h|--help)
    cat <<EOF
AXION ultimate final (frozen edge_rare / Table-1 fair)

  bash scripts/run_final.sh assert
  bash scripts/run_final.sh probe12
  bash scripts/run_final.sh gates
  bash scripts/run_final.sh full57
  bash scripts/run_final.sh all
  bash scripts/run_final.sh status

Config: $CFG
Seeds:  ${SEEDS[*]}
Jobs:   probe12=120  full57=570
EOF
    ;;
  *)
    echo "Unknown mode: $MODE (assert|probe12|gates|full57|all|status)" >&2
    exit 2
    ;;
esac

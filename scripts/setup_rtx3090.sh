#!/usr/bin/env bash
# One-time setup for RTX 3090 / 64 GiB / 16-thread Beat-Paper ceiling run.
# Usage (from repo root): bash scripts/setup_rtx3090.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== setup_rtx3090 =="
echo "ROOT=$ROOT"

if [[ ! -d "$ROOT/.venv" ]]; then
  python3 -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install -U pip wheel

# PyTorch: prefer CUDA 12.x wheel when nvidia-smi is present
if command -v nvidia-smi >/dev/null 2>&1; then
  pip install torch --index-url https://download.pytorch.org/whl/cu124
fi
pip install -r requirements.txt

# Fair eval: official ADBench clone + embeds only (not laptop rsync).
# Set VENDOR_ADBENCH=1 to rsync from ../ADBench instead.
if [[ "${VENDOR_ADBENCH:-0}" == "1" ]]; then
  bash scripts/vendor_data.sh
else
  LINK_IN_REPO=1 bash scripts/clone_adbench.sh
  SKIP_ADBENCH=1 bash scripts/vendor_data.sh
fi

export PYTHONPATH=src:scripts
export CFG=configs/gpu_beat_paper_3090.yaml
bash scripts/vast_ceiling_setup.sh

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("vram_gb:", round(props.total_memory / (1024**3), 1))
PY

echo ""
echo "SETUP_RTX3090_OK"
echo "code: $ROOT"
echo "datasets: $ROOT/data/adbench/datasets  (or ../ADBench/adbench/datasets)"
echo "embeds: $ROOT/data/embeds_alt  and  $ROOT/data/embeds_alt_whitened"
echo "Launch full run: bash scripts/run_seeds.sh"
echo "Monitor: watch -n2 nvidia-smi"

#!/usr/bin/env bash
# GPU host setup for dual_lift ceiling rescoring (itm-adbench-axion).
# Usage (from repo root):
#   bash scripts/vast_ceiling_setup.sh
# Optional env:
#   ADBENCH_ROOT=...   EMBEDS_SRC=...   (fallback link/copy hints)
#
# YAML (configs/gpu_beat_paper*.yaml) expects, from repo root:
#   data/adbench/datasets
#   data/embeds_alt
#   data/embeds_alt_whitened
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ITM_ROOT="${ITM_ROOT:-$(cd "$ROOT/.." && pwd)}"
LOCAL_ADBENCH="$ROOT/data/adbench/datasets"
LOCAL_ALT="$ROOT/data/embeds_alt"
LOCAL_WHITE="$ROOT/data/embeds_alt_whitened"
ADBENCH_ROOT="${ADBENCH_ROOT:-$LOCAL_ADBENCH}"
EMBEDS_SRC="${EMBEDS_SRC:-$ITM_ROOT/project/axion/data}"
PY="${PY:-python3}"

echo "== ceiling setup =="
echo "ROOT=$ROOT"
echo "ADBENCH_ROOT=$ADBENCH_ROOT"
echo "EMBEDS_ALT=$LOCAL_ALT"
echo "EMBEDS_WHITE=$LOCAL_WHITE"

mkdir -p data results configs

# Prefer in-repo vendored embeds (real dirs)
if [[ ! -d "$LOCAL_WHITE/CV_by_ViT" ]]; then
  if [[ -d "$EMBEDS_SRC/embeds_alt_whitened" ]]; then
    echo "WARN: vendored embeds missing; run: bash scripts/vendor_data.sh" >&2
    echo "      Or link from EMBEDS_SRC=$EMBEDS_SRC" >&2
    if [[ ! -e "$LOCAL_ALT" ]]; then
      ln -sfn "$EMBEDS_SRC/embeds_alt" "$LOCAL_ALT"
    fi
    if [[ ! -e "$LOCAL_WHITE" ]]; then
      ln -sfn "$EMBEDS_SRC/embeds_alt_whitened" "$LOCAL_WHITE"
    fi
    echo "linked embeds from $EMBEDS_SRC"
  else
    echo "ERROR: missing whitened embeds at $LOCAL_WHITE/CV_by_ViT" >&2
    echo "Run: bash scripts/vendor_data.sh" >&2
    exit 1
  fi
fi

# ADBench classical / CV / NLP folders
if [[ ! -d "$LOCAL_ADBENCH/Classical" ]]; then
  for fallback in \
    "$ITM_ROOT/ADBench/adbench/datasets" \
    "$ROOT/ADBench/adbench/datasets"; do
    if [[ -d "$fallback/Classical" ]]; then
      echo "WARN: $LOCAL_ADBENCH missing Classical; using $fallback" >&2
      mkdir -p "$(dirname "$LOCAL_ADBENCH")"
      if [[ ! -e "$LOCAL_ADBENCH" ]]; then
        ln -sfn "$fallback" "$LOCAL_ADBENCH"
        echo "linked $LOCAL_ADBENCH -> $fallback"
      fi
      ADBENCH_ROOT="$LOCAL_ADBENCH"
      if [[ ! -d "$ADBENCH_ROOT/Classical" ]]; then
        ADBENCH_ROOT="$fallback"
      fi
      break
    fi
  done
fi
if [[ ! -d "$ADBENCH_ROOT/Classical" ]]; then
  echo "ERROR: ADBench datasets missing (expected data/adbench/datasets or ../ADBench/adbench/datasets)" >&2
  echo "Run: LINK_IN_REPO=1 bash scripts/clone_adbench.sh" >&2
  exit 1
fi

# Config sanity
for f in configs/gpu_beat_paper.yaml configs/gpu_beat_paper_3090.yaml configs/gpu_final.yaml \
         results/axion_beat_paper_dual_lift/thesis/recipe_map_57_beat.json; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

export PYTHONPATH=src:scripts
CFG_CHECK="${CFG:-configs/gpu_beat_paper.yaml}"
"$PY" - <<PY
from pathlib import Path
from axion.config import load_config
from beat_paper_catalog import embed_folders, RECIPES
import torch

cfg = load_config(Path("$CFG_CHECK"))
folders = embed_folders(cfg)
checks = {
    "cv_vit_whitened": folders["cv_vit_whitened"],
    "nlp_e5_whitened": folders["nlp_e5_whitened"],
    "classical": folders["classical"],
}
ok = True
for name, path in checks.items():
    n = len(list(path.glob("*.npz"))) if path.exists() else 0
    print(f"{name}: {path} exists={path.exists()} npz={n}")
    if n == 0:
        ok = False
assert "gmm_pca64_vit" in RECIPES and RECIPES["gmm_pca64_vit"].n_components == 1
assert "axion" in RECIPES and "axion_edge_rare" in RECIPES
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
if not ok:
    raise SystemExit("embed pools incomplete")
print("SETUP_OK")
print("code:", r"$ROOT")
print("adbench_root:", cfg["paths"]["adbench_root"])
print("embeds_alt_root:", cfg["paths"].get("embeds_alt_root"))
print("embeds_alt_whitened_root:", cfg["paths"].get("embeds_alt_whitened_root"))
PY

echo "Next: bash scripts/run_seeds.sh"

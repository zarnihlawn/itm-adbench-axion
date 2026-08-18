#!/usr/bin/env bash
# One-time copy: ADBench datasets + alt embeds into itm-adbench-axion/data/
# Idempotent (rsync). Data stays local; not committed to git.
#
# Usage (from repo root):
#   bash scripts/vendor_data.sh
#
# Optional env:
#   ADBENCH_SRC=../ADBench/adbench/datasets
#   EMBEDS_SRC=../project/axion/data
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ITM_ROOT="${ITM_ROOT:-$(cd "$ROOT/.." && pwd)}"
ADBENCH_SRC="${ADBENCH_SRC:-$ITM_ROOT/ADBench/adbench/datasets}"
EMBEDS_SRC="${EMBEDS_SRC:-$ITM_ROOT/project/axion/data}"

DEST_ADBENCH="$ROOT/data/adbench/datasets"
DEST_ALT="$ROOT/data/embeds_alt"
DEST_WHITE="$ROOT/data/embeds_alt_whitened"

echo "== vendor_data =="
echo "ROOT=$ROOT"
echo "ADBENCH_SRC=$ADBENCH_SRC -> $DEST_ADBENCH"
echo "EMBEDS_SRC=$EMBEDS_SRC -> $DEST_ALT + $DEST_WHITE"

if [[ ! -d "$ADBENCH_SRC/Classical" ]]; then
  echo "ERROR: ADBench source missing at $ADBENCH_SRC" >&2
  echo "Set ADBENCH_SRC to a checkout of ADBench/adbench/datasets" >&2
  exit 1
fi
if [[ ! -d "$EMBEDS_SRC/embeds_alt_whitened" ]]; then
  echo "ERROR: embeds source missing at $EMBEDS_SRC/embeds_alt_whitened" >&2
  echo "Set EMBEDS_SRC to project/axion/data (or equivalent)" >&2
  exit 1
fi

mkdir -p "$DEST_ADBENCH" "$(dirname "$DEST_ALT")"

# Replace symlinks with real directories
for name in embeds_alt embeds_alt_whitened; do
  target="$ROOT/data/$name"
  if [[ -L "$target" ]]; then
    rm -f "$target"
    echo "removed symlink $target"
  fi
done

echo "== rsync ADBench datasets (~2 GB) =="
rsync -aP "$ADBENCH_SRC/" "$DEST_ADBENCH/"

echo "== rsync embeds_alt (~300 MB) =="
rsync -aP "$EMBEDS_SRC/embeds_alt/" "$DEST_ALT/"

echo "== rsync embeds_alt_whitened (~1.6 GB) =="
rsync -aP "$EMBEDS_SRC/embeds_alt_whitened/" "$DEST_WHITE/"

echo "== verify =="
for check in "$DEST_ADBENCH/Classical" "$DEST_WHITE/CV_by_ViT"; do
  if [[ ! -d "$check" ]]; then
    echo "ERROR: expected directory missing: $check" >&2
    exit 1
  fi
done

du -sh "$DEST_ADBENCH" "$DEST_ALT" "$DEST_WHITE" 2>/dev/null || true
echo "VENDOR_OK"

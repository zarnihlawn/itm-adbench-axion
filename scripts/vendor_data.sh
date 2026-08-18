#!/usr/bin/env bash
# Copy alt embeds into itm-adbench-axion/data/ (and optionally ADBench NPZs).
# Idempotent (rsync). Data stays local; not committed to git.
#
# Fair eval (recommended on fresh hosts): clone official ADBench first, skip copy:
#   LINK_IN_REPO=1 bash scripts/clone_adbench.sh
#   SKIP_ADBENCH=1 bash scripts/vendor_data.sh
#
# Usage (from repo root):
#   bash scripts/vendor_data.sh
#
# Optional env:
#   SKIP_ADBENCH=1     do not rsync ADBench; require datasets via clone or symlink
#   ADBENCH_SRC=../ADBench/adbench/datasets
#   EMBEDS_SRC=../project/axion/data
#
# Paths from repo root: data/adbench/datasets, data/embeds_alt, data/embeds_alt_whitened
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ITM_ROOT="${ITM_ROOT:-$(cd "$ROOT/.." && pwd)}"
SKIP_ADBENCH="${SKIP_ADBENCH:-0}"
ADBENCH_SRC="${ADBENCH_SRC:-$ITM_ROOT/ADBench/adbench/datasets}"
EMBEDS_SRC="${EMBEDS_SRC:-$ITM_ROOT/project/axion/data}"

DEST_ADBENCH="$ROOT/data/adbench/datasets"
DEST_ALT="$ROOT/data/embeds_alt"
DEST_WHITE="$ROOT/data/embeds_alt_whitened"

_resolve_adbench() {
  for candidate in "$DEST_ADBENCH" "$ADBENCH_SRC" "$ITM_ROOT/ADBench/adbench/datasets" "$ROOT/ADBench/adbench/datasets"; do
    if [[ -d "$candidate/Classical" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

_rel_from_root() {
  python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$1" "$ROOT"
}

_ensure_inrepo_link() {
  local src="$1"
  if [[ -d "$DEST_ADBENCH/Classical" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$DEST_ADBENCH")"
  if [[ -e "$DEST_ADBENCH" && ! -L "$DEST_ADBENCH" ]]; then
    return 0
  fi
  ln -sfn "$src" "$DEST_ADBENCH"
  echo "linked $DEST_ADBENCH -> $src"
}

echo "== vendor_data =="
echo "ROOT=$ROOT"
echo "SKIP_ADBENCH=$SKIP_ADBENCH"
echo "ADBENCH_SRC=$ADBENCH_SRC -> $DEST_ADBENCH"
echo "EMBEDS_SRC=$EMBEDS_SRC -> $DEST_ALT + $DEST_WHITE"

if [[ ! -d "$EMBEDS_SRC/embeds_alt_whitened" ]]; then
  echo "ERROR: embeds source missing at $EMBEDS_SRC/embeds_alt_whitened" >&2
  echo "Set EMBEDS_SRC to project/axion/data (or equivalent)" >&2
  exit 1
fi

mkdir -p "$(dirname "$DEST_ADBENCH")" "$(dirname "$DEST_ALT")"

# Replace symlinks with real directories (embeds only; keep ADBench symlink from clone)
for name in embeds_alt embeds_alt_whitened; do
  target="$ROOT/data/$name"
  if [[ -L "$target" ]]; then
    rm -f "$target"
    echo "removed symlink $target"
  fi
done

if [[ "$SKIP_ADBENCH" == "1" ]]; then
  if ! resolved="$(_resolve_adbench)"; then
    echo "ERROR: SKIP_ADBENCH=1 but no ADBench datasets found" >&2
    echo "Run: LINK_IN_REPO=1 bash scripts/clone_adbench.sh" >&2
    exit 1
  fi
  echo "== skip ADBench rsync (using $resolved) =="
  if [[ "$resolved" != "$DEST_ADBENCH" ]]; then
    _ensure_inrepo_link "$resolved"
  fi
else
  if [[ ! -d "$ADBENCH_SRC/Classical" ]]; then
    echo "ERROR: ADBench source missing at $ADBENCH_SRC" >&2
    echo "Fair path: LINK_IN_REPO=1 bash scripts/clone_adbench.sh && SKIP_ADBENCH=1 bash scripts/vendor_data.sh" >&2
    exit 1
  fi
  mkdir -p "$DEST_ADBENCH"
  echo "== rsync ADBench datasets (~2 GB) =="
  rsync -aP "$ADBENCH_SRC/" "$DEST_ADBENCH/"
fi

echo "== rsync embeds_alt (~300 MB) =="
rsync -aP "$EMBEDS_SRC/embeds_alt/" "$DEST_ALT/"

echo "== rsync embeds_alt_whitened (~1.6 GB) =="
rsync -aP "$EMBEDS_SRC/embeds_alt_whitened/" "$DEST_WHITE/"

echo "== verify =="
if ! resolved="$(_resolve_adbench)"; then
  echo "ERROR: ADBench Classical folder missing" >&2
  exit 1
fi
for check in "$resolved/Classical" "$DEST_WHITE/CV_by_ViT"; do
  if [[ ! -d "$check" ]]; then
    echo "ERROR: expected directory missing: $check" >&2
    exit 1
  fi
done

du -sh "$resolved" "$DEST_ALT" "$DEST_WHITE" 2>/dev/null || true
echo "VENDOR_OK"
echo "  code=$ROOT"
echo "  adbench=$resolved  ($(_rel_from_root "$resolved"))"
echo "  embeds_alt=$DEST_ALT  (data/embeds_alt)"
echo "  embeds_alt_whitened=$DEST_WHITE  (data/embeds_alt_whitened)"

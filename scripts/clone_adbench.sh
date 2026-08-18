#!/usr/bin/env bash
# Fair ADBench path: clone official Minqi824/ADBench and download NPZ datasets
# (do not rsync a laptop copy). Idempotent: pull + skip existing files.
#
# Usage (from repo root):
#   bash scripts/clone_adbench.sh
#
# Layout options:
#   Option A (default): sibling ../ADBench under ITM parent
#     ADBENCH_CLONE_DIR=../ADBench
#   Option B: clone beside this repo (any path)
#     ADBENCH_CLONE_DIR=/path/to/ADBench bash scripts/clone_adbench.sh
#   Option B in-repo: clone inside itm-adbench-axion
#     ADBENCH_CLONE_DIR=./ADBench bash scripts/clone_adbench.sh
#
# Optional env:
#   ADBENCH_REPO       git remote (default: https://github.com/Minqi824/ADBench.git)
#   ADBENCH_REF        branch/tag (default: main)
#   ADBENCH_DL_REPO    github|jihulab (default: github; jihulab for CN mainland)
#   ADBENCH_FOLDERS    space-separated folder names under adbench/datasets/
#   LINK_IN_REPO=1     symlink data/adbench/datasets -> clone datasets (for configs)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ITM_ROOT="${ITM_ROOT:-$(cd "$ROOT/.." && pwd)}"
ADBENCH_CLONE_DIR="${ADBENCH_CLONE_DIR:-$ITM_ROOT/ADBench}"
ADBENCH_REPO="${ADBENCH_REPO:-https://github.com/Minqi824/ADBench.git}"
ADBENCH_REF="${ADBENCH_REF:-main}"
ADBENCH_DL_REPO="${ADBENCH_DL_REPO:-github}"
ADBENCH_FOLDERS="${ADBENCH_FOLDERS:-Classical CV_by_ResNet18 NLP_by_BERT CV_by_ViT NLP_by_RoBERTa}"
LINK_IN_REPO="${LINK_IN_REPO:-0}"

DATASETS_DIR="$ADBENCH_CLONE_DIR/adbench/datasets"
LINK_TARGET="$ROOT/data/adbench/datasets"

echo "== clone_adbench =="
echo "ROOT=$ROOT"
echo "ADBENCH_CLONE_DIR=$ADBENCH_CLONE_DIR"
echo "ADBENCH_DL_REPO=$ADBENCH_DL_REPO"
echo "ADBENCH_FOLDERS=$ADBENCH_FOLDERS"

if [[ -d "$ADBENCH_CLONE_DIR/.git" ]]; then
  echo "== git pull =="
  git -C "$ADBENCH_CLONE_DIR" fetch origin "$ADBENCH_REF"
  git -C "$ADBENCH_CLONE_DIR" checkout "$ADBENCH_REF"
  git -C "$ADBENCH_CLONE_DIR" pull --ff-only origin "$ADBENCH_REF" || true
else
  echo "== git clone =="
  git clone --depth 1 --branch "$ADBENCH_REF" "$ADBENCH_REPO" "$ADBENCH_CLONE_DIR"
fi

mkdir -p "$DATASETS_DIR"

need_download=0
for folder in $ADBENCH_FOLDERS; do
  dir="$DATASETS_DIR/$folder"
  count=0
  if [[ -d "$dir" ]]; then
    count="$(find "$dir" -maxdepth 1 -name '*.npz' | wc -l)"
  fi
  if [[ "$count" -eq 0 ]]; then
    need_download=1
    break
  fi
done

if [[ "$need_download" -eq 0 ]]; then
  echo "== datasets present; skip download =="
else
  echo "== download NPZ datasets (official manifest) =="
  ADBENCH_CLONE_DIR="$ADBENCH_CLONE_DIR" \
  ADBENCH_DL_REPO="$ADBENCH_DL_REPO" \
  ADBENCH_FOLDERS="$ADBENCH_FOLDERS" \
  python3 - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

clone = Path(os.environ["ADBENCH_CLONE_DIR"])
datasets = clone / "adbench" / "datasets"
repo = os.environ.get("ADBENCH_DL_REPO", "github")
folders = os.environ.get("ADBENCH_FOLDERS", "").split()


def download_jihulab() -> None:
    base = (
        "https://jihulab.com/BraudoCC/ADBench_datasets/-/raw/"
        "339d2ab2d53416854f6535442a67393634d1a778"
    )
    manifest_url = f"{base}/datasets_files_name.json"
    with urllib.request.urlopen(manifest_url, timeout=120) as resp:
        manifest = json.loads(resp.read().decode())
    for folder in folders:
        names = manifest.get(folder)
        if not names:
            print(f"WARN: folder not in manifest: {folder}", file=sys.stderr)
            continue
        dest = datasets / folder
        dest.mkdir(parents=True, exist_ok=True)
        print(f"folder {folder} ({len(names)} files) -> {dest}")
        for name in names:
            out = dest / name
            if out.exists():
                continue
            url = f"{base}/{folder}/{name}"
            print(f"  fetch {name}")
            with urllib.request.urlopen(url, timeout=300) as resp:
                out.write_bytes(resp.read())


def download_github() -> None:
    api = "https://api.github.com/repos/Minqi824/ADBench/contents/adbench/datasets"
    for folder in folders:
        dest = datasets / folder
        dest.mkdir(parents=True, exist_ok=True)
        url = f"{api}/{folder}"
        print(f"folder {folder} -> {dest}")
        req = urllib.request.Request(url, headers={"User-Agent": "itm-adbench-axion"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            entries = json.loads(resp.read().decode())
        if not isinstance(entries, list):
            raise RuntimeError(f"unexpected GitHub listing for {folder}: {entries}")
        for entry in entries:
            name = str(entry.get("name") or "")
            if not name.endswith(".npz"):
                continue
            out = dest / name
            if out.exists():
                continue
            dl = entry.get("download_url")
            if not dl:
                continue
            print(f"  fetch {name}")
            dreq = urllib.request.Request(dl, headers={"User-Agent": "itm-adbench-axion"})
            with urllib.request.urlopen(dreq, timeout=300) as resp:
                out.write_bytes(resp.read())


def try_github() -> None:
    try:
        download_github()
    except Exception as exc:
        print(f"github download failed: {exc}", file=sys.stderr)
        raise


def try_jihulab() -> None:
    try:
        download_jihulab()
    except Exception as exc:
        print(f"jihulab download failed: {exc}", file=sys.stderr)
        raise


if repo == "github":
    try:
        try_github()
    except Exception:
        print("retrying from jihulab...", file=sys.stderr)
        try_jihulab()
else:
    try:
        try_jihulab()
    except Exception:
        print("retrying from github...", file=sys.stderr)
        try_github()
PY
fi

echo "== verify NPZ folders =="
fail=0
for folder in $ADBENCH_FOLDERS; do
  dir="$DATASETS_DIR/$folder"
  count=0
  if [[ -d "$dir" ]]; then
    count="$(find "$dir" -maxdepth 1 -name '*.npz' | wc -l)"
  fi
  echo "  $folder: npz=$count path=$dir"
  if [[ "$count" -eq 0 ]]; then
    echo "ERROR: no NPZ in $dir" >&2
    fail=1
  fi
done
if [[ "$fail" -ne 0 ]]; then
  echo "Download incomplete. Try ADBENCH_DL_REPO=github or manual fetch from:" >&2
  echo "  https://github.com/Minqi824/ADBench/tree/main/adbench/datasets" >&2
  exit 1
fi

if [[ "$LINK_IN_REPO" == "1" ]]; then
  mkdir -p "$(dirname "$LINK_TARGET")"
  if [[ -e "$LINK_TARGET" && ! -L "$LINK_TARGET" ]]; then
    echo "ERROR: $LINK_TARGET exists and is not a symlink; remove or relocate it" >&2
    exit 1
  fi
  ln -sfn "$DATASETS_DIR" "$LINK_TARGET"
  echo "linked $LINK_TARGET -> $DATASETS_DIR"
fi

du -sh "$DATASETS_DIR"/* 2>/dev/null || true
echo "CLONE_ADBENCH_OK"
echo "  code=$ROOT"
echo "  clone=$ADBENCH_CLONE_DIR"
echo "  datasets=$DATASETS_DIR"
if [[ -e "$LINK_TARGET" ]]; then
  echo "  yaml adbench_root=$LINK_TARGET  (data/adbench/datasets)"
else
  echo "  yaml: set LINK_IN_REPO=1 so configs/gpu_beat_paper*.yaml can use data/adbench/datasets"
fi

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
#
# Campus / flaky GitHub (HTTP/2 CANCEL, "N bytes of body are still expected"):
#   git clone/pull use HTTP/1.1 for that command only (does NOT write ~/.gitconfig).
#   Clone retries 3 times. Each NPZ retries 3 times; GitHub then falls back to jihulab.
#   If HTTPS clone already failed mid-way:
#     git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/Minqi824/ADBench.git ../ADBench
#     ADBENCH_DL_REPO=jihulab LINK_IN_REPO=1 bash scripts/clone_adbench.sh
#   Or: git clone git@github.com:Minqi824/ADBench.git ../ADBench
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

# Per-invocation only: do not change the user's global git config.
# Campus GitHub often cancels HTTP/2: curl 92 HTTP/2 stream 5 was closed cleanly: CANCEL.
git_http11() {
  git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 "$@"
}

print_https_clone_fallback() {
  echo "HTTPS clone/pull failed (Git HTTP/2 cancel / flaky GitHub is common on campus)." >&2
  echo "try SSH git@github.com:Minqi824/ADBench.git or ADBENCH_DL_REPO=jihulab" >&2
  echo "Resume one-liners (from itm-adbench-axion root):" >&2
  echo "  git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/Minqi824/ADBench.git ../ADBench" >&2
  echo "  ADBENCH_DL_REPO=jihulab LINK_IN_REPO=1 bash scripts/clone_adbench.sh" >&2
}

clone_adbench_git() {
  local dest="$1"
  local attempt
  mkdir -p "$(dirname "$dest")"
  if [[ -e "$dest" && ! -d "$dest/.git" ]]; then
    echo "incomplete clone dir without .git; removing $dest"
    rm -rf "$dest"
  fi
  for attempt in 1 2 3; do
    echo "== git clone attempt ${attempt}/3 (HTTP/1.1, --depth 1) =="
    if [[ -e "$dest" ]]; then
      echo "removing incomplete dest $dest"
      rm -rf "$dest"
    fi
    if git_http11 clone --depth 1 --branch "$ADBENCH_REF" "$ADBENCH_REPO" "$dest"; then
      return 0
    fi
    echo "clone attempt ${attempt}/3 failed" >&2
    rm -rf "$dest"
    if [[ "$attempt" -lt 3 ]]; then
      sleep $((attempt * 2))
    fi
  done
  print_https_clone_fallback
  return 1
}

pull_adbench_git() {
  local dest="$1"
  local attempt
  echo "== git pull (HTTP/1.1) =="
  for attempt in 1 2 3; do
    echo "fetch/pull attempt ${attempt}/3"
    if git_http11 -C "$dest" fetch origin "$ADBENCH_REF" \
      && git_http11 -C "$dest" checkout "$ADBENCH_REF" \
      && git_http11 -C "$dest" pull --ff-only origin "$ADBENCH_REF"; then
      return 0
    fi
    echo "fetch/pull attempt ${attempt}/3 failed" >&2
    if [[ "$attempt" -lt 3 ]]; then
      sleep $((attempt * 2))
    fi
  done
  echo "git pull failed after 3 attempts; continuing with existing clone" >&2
  print_https_clone_fallback
  return 0
}

if [[ -d "$ADBENCH_CLONE_DIR/.git" ]]; then
  pull_adbench_git "$ADBENCH_CLONE_DIR"
else
  clone_adbench_git "$ADBENCH_CLONE_DIR"
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
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# urllib/http.client already speaks HTTP/1.1; set explicitly for campus proxies.
http.client.HTTPConnection._http_vsn = 11
http.client.HTTPConnection._http_vsn_str = "HTTP/1.1"

clone = Path(os.environ["ADBENCH_CLONE_DIR"])
datasets = clone / "adbench" / "datasets"
repo = os.environ.get("ADBENCH_DL_REPO", "github")
folders = os.environ.get("ADBENCH_FOLDERS", "").split()
UA = {"User-Agent": "itm-adbench-axion"}
ATTEMPTS = 3


def fetch_bytes(url: str, timeout: int, what: str, source: str) -> bytes:
    last = None
    for i in range(1, ATTEMPTS + 1):
        req = urllib.request.Request(url, headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            last = exc
            print(
                f"  {source} retry {i}/{ATTEMPTS} for {what}: {exc}",
                file=sys.stderr,
            )
            if i < ATTEMPTS:
                time.sleep(i * 2)
    raise last


def write_url(url: str, out: Path, timeout: int, what: str, source: str) -> None:
    data = fetch_bytes(url, timeout, what, source)
    part = out.with_name(out.name + ".part")
    try:
        part.write_bytes(data)
        part.replace(out)
    finally:
        if part.exists():
            part.unlink()


def download_jihulab() -> None:
    base = (
        "https://jihulab.com/BraudoCC/ADBench_datasets/-/raw/"
        "339d2ab2d53416854f6535442a67393634d1a778"
    )
    manifest_url = f"{base}/datasets_files_name.json"
    print("NPZ source: jihulab (3 retries per file)")
    manifest = json.loads(fetch_bytes(manifest_url, 120, "manifest", "jihulab").decode())
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
            write_url(url, out, 300, name, "jihulab")


def download_github() -> None:
    api = "https://api.github.com/repos/Minqi824/ADBench/contents/adbench/datasets"
    print("NPZ source: GitHub API (HTTP/1.1 urllib, 3 retries per file)")
    for folder in folders:
        dest = datasets / folder
        dest.mkdir(parents=True, exist_ok=True)
        url = f"{api}/{folder}"
        print(f"folder {folder} -> {dest}")
        entries = json.loads(
            fetch_bytes(url, 120, f"GitHub listing {folder}", "github").decode()
        )
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
            write_url(dl, out, 300, name, "github")


def try_github() -> None:
    try:
        download_github()
    except Exception as exc:
        print(f"github download failed after {ATTEMPTS} retries per file: {exc}", file=sys.stderr)
        raise


def try_jihulab() -> None:
    try:
        download_jihulab()
    except Exception as exc:
        print(f"jihulab download failed after {ATTEMPTS} retries per file: {exc}", file=sys.stderr)
        raise


if repo == "github":
    try:
        try_github()
    except Exception:
        print(
            "GitHub NPZ download failed after 3 retries per file; falling back to jihulab...",
            file=sys.stderr,
        )
        try_jihulab()
else:
    try:
        try_jihulab()
    except Exception:
        print(
            "jihulab NPZ download failed after 3 retries per file; falling back to github...",
            file=sys.stderr,
        )
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
  echo "Download incomplete. Try ADBENCH_DL_REPO=jihulab or manual fetch from:" >&2
  echo "  https://github.com/Minqi824/ADBench/tree/main/adbench/datasets" >&2
  echo "  ADBENCH_DL_REPO=jihulab LINK_IN_REPO=1 bash scripts/clone_adbench.sh" >&2
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

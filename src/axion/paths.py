"""Path resolution for standalone itm-adbench-axion checkout.

Code lives in AXION_ROOT (this repo). Datasets are not in git:

  (a) sibling official clone:  ../ADBench/adbench/datasets
  (b) in-repo copy/symlink:    data/adbench/datasets   (Beat-Paper YAML)

Whitened ViT/E5 embeds (not in official ADBench):

  data/embeds_alt
  data/embeds_alt_whitened
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root (itm-adbench-axion/)
AXION_ROOT = Path(__file__).resolve().parents[2]
# Parent of repo (usually ITM/)
ITM_ROOT = AXION_ROOT.parent
# Back-compat alias used by older scripts
PROJECT_ROOT = AXION_ROOT

# In-repo vendor/symlink first, then in-repo clone, then sibling official clone
_LOCAL_ADBENCH = AXION_ROOT / "data" / "adbench" / "datasets"
_INREPO_CLONE = AXION_ROOT / "ADBench" / "adbench" / "datasets"
_SIBLING_ADBENCH = ITM_ROOT / "ADBench" / "adbench" / "datasets"


def _first_existing(*candidates: Path) -> Path:
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[-1]


def resolve_adbench_datasets() -> Path:
    """Resolve ADBench NPZ root: env override, vendored, clone, or sibling."""
    env = os.environ.get("ADBENCH_DATASETS") or os.environ.get("ADBENCH_ROOT")
    if env:
        p = Path(env).expanduser()
        if not p.is_absolute():
            p = (AXION_ROOT / p).resolve()
        return p
    return _first_existing(_LOCAL_ADBENCH, _INREPO_CLONE, _SIBLING_ADBENCH)


DEFAULT_ADBENCH_DATASETS = resolve_adbench_datasets()
# Official AnoDDAE reference (read-only protocol twin)
ANODDAE_SRC = ITM_ROOT / "AnoDDAE" / "AnoDDAE" / "src"

DATA_DIR = AXION_ROOT / "data"
RESULTS_DIR = AXION_ROOT / "results"
ATLAS_CSV = DATA_DIR / "atlas_57.csv"
EMBEDS_ALT = DATA_DIR / "embeds_alt"
EMBEDS_ALT_WHITENED = DATA_DIR / "embeds_alt_whitened"

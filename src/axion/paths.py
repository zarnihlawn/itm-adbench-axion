"""Path resolution for standalone itm-adbench-axion checkout."""
from __future__ import annotations

from pathlib import Path

# Repo root (itm-adbench-axion/)
AXION_ROOT = Path(__file__).resolve().parents[2]
# Parent of repo (usually ITM/)
ITM_ROOT = AXION_ROOT.parent
# Back-compat alias used by older scripts
PROJECT_ROOT = AXION_ROOT

# In-repo vendor path first; sibling ITM/ADBench fallback for dev checkouts
_LOCAL_ADBENCH = AXION_ROOT / "data" / "adbench" / "datasets"
_SIBLING_ADBENCH = ITM_ROOT / "ADBench" / "adbench" / "datasets"
DEFAULT_ADBENCH_DATASETS = (
    _LOCAL_ADBENCH if _LOCAL_ADBENCH.is_dir() else _SIBLING_ADBENCH
)
# Official AnoDDAE reference (read-only protocol twin)
ANODDAE_SRC = ITM_ROOT / "AnoDDAE" / "AnoDDAE" / "src"

DATA_DIR = AXION_ROOT / "data"
RESULTS_DIR = AXION_ROOT / "results"
ATLAS_CSV = DATA_DIR / "atlas_57.csv"

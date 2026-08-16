"""Path resolution for AXION (lives at ITM/project/axion)."""
from __future__ import annotations

from pathlib import Path

# ITM/project/axion/
AXION_ROOT = Path(__file__).resolve().parents[2]
# ITM/project/
PROJECT_ROOT = AXION_ROOT.parent
# ITM/
ITM_ROOT = PROJECT_ROOT.parent
# ADBench is a sibling of project under ITM/
DEFAULT_ADBENCH_DATASETS = ITM_ROOT / "ADBench" / "adbench" / "datasets"
# Official AnoDDAE reference (read-only protocol twin)
ANODDAE_SRC = ITM_ROOT / "AnoDDAE" / "AnoDDAE" / "src"

DATA_DIR = AXION_ROOT / "data"
RESULTS_DIR = AXION_ROOT / "results"
ATLAS_CSV = DATA_DIR / "atlas_57.csv"

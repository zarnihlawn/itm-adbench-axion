"""Load AXION YAML configs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from axion.paths import AXION_ROOT, DEFAULT_ADBENCH_DATASETS


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else AXION_ROOT / "configs" / "default.yaml"
    if not cfg_path.is_absolute():
        cfg_path = AXION_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    paths = cfg.setdefault("paths", {})
    adbench = paths.get("adbench_root", "../../ADBench/adbench/datasets")
    adbench_path = Path(adbench)
    if not adbench_path.is_absolute():
        adbench_path = (AXION_ROOT / adbench_path).resolve()
    if not adbench_path.exists():
        adbench_path = DEFAULT_ADBENCH_DATASETS
    paths["adbench_root"] = str(adbench_path)

    results = paths.get("results_dir", "results")
    results_path = Path(results)
    if not results_path.is_absolute():
        results_path = AXION_ROOT / results_path
    paths["results_dir"] = str(results_path)
    return cfg

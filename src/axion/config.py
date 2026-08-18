"""Load AXION YAML configs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from axion.paths import (
    AXION_ROOT,
    DEFAULT_ADBENCH_DATASETS,
    EMBEDS_ALT,
    EMBEDS_ALT_WHITENED,
)


def _resolve_under_root(raw: str | Path, *, fallback: Optional[Path] = None) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (AXION_ROOT / p).resolve()
    if fallback is not None and not p.exists():
        return fallback
    return p


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else AXION_ROOT / "configs" / "default.yaml"
    if not cfg_path.is_absolute():
        cfg_path = AXION_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    paths = cfg.setdefault("paths", {})
    adbench = paths.get("adbench_root", "data/adbench/datasets")
    paths["adbench_root"] = str(
        _resolve_under_root(adbench, fallback=DEFAULT_ADBENCH_DATASETS)
    )
    paths["embeds_alt_root"] = str(
        _resolve_under_root(paths.get("embeds_alt_root", "data/embeds_alt"), fallback=EMBEDS_ALT)
    )
    paths["embeds_alt_whitened_root"] = str(
        _resolve_under_root(
            paths.get("embeds_alt_whitened_root", "data/embeds_alt_whitened"),
            fallback=EMBEDS_ALT_WHITENED,
        )
    )

    results = paths.get("results_dir", "results")
    results_path = Path(results)
    if not results_path.is_absolute():
        results_path = AXION_ROOT / results_path
    paths["results_dir"] = str(results_path)
    return cfg

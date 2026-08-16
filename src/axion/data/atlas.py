"""Build atlas_57.csv from ADBench NPZs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from axion.data.registry import DatasetSpec, build_registry, load_npz
from axion.paths import ATLAS_CSV, DATA_DIR, DEFAULT_ADBENCH_DATASETS


def _difficulty(n: int, d: int, anom_rate: float) -> str:
    tags: List[str] = []
    if anom_rate < 0.01:
        tags.append("extreme_imbalance")
    elif anom_rate >= 0.30:
        tags.append("high_anom_rate")
    if d >= 400:
        tags.append("high_d")
    if n >= 100_000:
        tags.append("large_n")
    elif n < 500:
        tags.append("tiny_n")
    if not tags:
        tags.append("standard")
    return "|".join(tags)


def summarize_spec(spec: DatasetSpec, adbench_root: Path) -> Dict[str, Any]:
    n_total = 0
    d = None
    n_anom = 0
    n_files = len(spec.relative_paths)
    for rel in spec.relative_paths:
        X, y = load_npz(adbench_root / rel)
        n_total += int(X.shape[0])
        d = int(X.shape[1]) if d is None else d
        if d != int(X.shape[1]):
            raise ValueError(f"Inconsistent d in {spec.name}: {d} vs {X.shape[1]}")
        n_anom += int((y == 1).sum())
    anom_rate = float(n_anom) / float(n_total) if n_total else 0.0
    # For multi-file families, paper Table 3 often cites a representative / pooled size.
    # We keep both pooled totals and per-file mean size.
    mean_n = n_total / n_files
    return {
        "name": spec.name,
        "modality": spec.modality,
        "paper_category": spec.paper_category,
        "embedding": spec.embedding,
        "n_files": n_files,
        "n_pooled": n_total,
        "n_mean_per_file": round(mean_n, 2),
        "d": d,
        "n_anom_pooled": n_anom,
        "anom_rate_pooled": round(anom_rate, 6),
        "difficulty": _difficulty(int(mean_n), int(d or 0), anom_rate),
        "relative_paths": ";".join(spec.relative_paths),
    }


def build_atlas(
    adbench_root: Optional[Path] = None,
    out_csv: Optional[Path] = None,
    *,
    cv_folder: str = "CV_by_ResNet18",
    nlp_folder: str = "NLP_by_BERT",
) -> pd.DataFrame:
    root = Path(adbench_root) if adbench_root is not None else DEFAULT_ADBENCH_DATASETS
    out = Path(out_csv) if out_csv is not None else ATLAS_CSV
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        summarize_spec(s, root)
        for s in build_registry(root, cv_folder=cv_folder, nlp_folder=nlp_folder)
    ]
    df = pd.DataFrame(rows)
    if len(df) != 57:
        raise RuntimeError(f"Atlas expected 57 rows, got {len(df)}")
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--adbench-root", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cv-folder", default="CV_by_ResNet18")
    p.add_argument("--nlp-folder", default="NLP_by_BERT")
    args = p.parse_args()
    out = args.out or ATLAS_CSV
    df = build_atlas(
        args.adbench_root,
        out,
        cv_folder=args.cv_folder,
        nlp_folder=args.nlp_folder,
    )
    print(f"Wrote {out} ({len(df)} datasets)")
    print(df.groupby("modality").size().to_string())
    print(df.groupby("embedding").size().to_string())

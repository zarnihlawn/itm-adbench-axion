#!/usr/bin/env python3
"""F3 new heads on remaining fair unpaid: PatchCore-lite (CV) + DTE-lite (classical).

No ECOD/COPOD/GOAD. Promote only if PR ≥ lock + 0.5.

  python scripts/fair_new_heads.py --datasets CIFAR10 SVHN speech ALOI Waveform optdigits
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from axion.data.registry import CLASSICAL_FILES, _classical_display_name, registry_by_name  # noqa: E402
from metric_drag import append_drag, rewrite_report  # noqa: E402
from shallow_two_step import _pr_roc, _split_semi  # noqa: E402

PAPER_IDS_CV = ["P24", "P25"]
PAPER_IDS_DTE = ["P05", "P06"]
PROMOTE_DELTA = 0.5
LOCKS = {
    "CIFAR10": 20.35,
    "SVHN": 15.25,
    "FashionMNIST": 60.08,
    "speech": 4.94,
    "ALOI": 8.04,
    "Waveform": 9.08,
    "optdigits": 22.73,
}


def _zscore(X_tr: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = np.maximum(X_tr.std(axis=0, keepdims=True), 1e-6)
    return ((X_tr - mu) / sd).astype(np.float32), ((X_te - mu) / sd).astype(np.float32)


def patchcore_lite_scores(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    memory_cap: int = 8000,
    n_neighbors: int = 5,
) -> np.ndarray:
    """kNN distance to subsampled train-normal memory (PatchCore-lite on frozen feats)."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.RandomState(0)
    n = len(X_train)
    if n > memory_cap:
        idx = rng.choice(n, size=memory_cap, replace=False)
        mem = X_train[idx]
    else:
        mem = X_train
    k = min(n_neighbors, max(1, len(mem)))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(mem)
    dists, _ = nn.kneighbors(X_test)
    return dists.mean(axis=1)


def dte_lite_scores(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_scales: int = 4,
) -> np.ndarray:
    """DTE-lite: multi-scale Gaussian noise residual energy vs train normals (P05-ish)."""
    rng = np.random.RandomState(0)
    X_tr, X_te = _zscore(X_train, X_test)
    # center of train normals
    center = X_tr.mean(axis=0)
    scores = np.zeros(len(X_te), dtype=np.float64)
    for s in range(1, n_scales + 1):
        sigma = 0.1 * s
        noise = rng.randn(*X_te.shape).astype(np.float32) * sigma
        # score = distance of noised point from train center (higher = more OD)
        scores += np.linalg.norm((X_te + noise) - center, axis=1)
    return scores / float(n_scales)


def _load_family(adbench: Path, dataset: str) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    reg = registry_by_name(adbench)
    if dataset not in reg:
        # classical by name
        fname = None
        for f in CLASSICAL_FILES:
            if _classical_display_name(f) == dataset:
                fname = f
                break
        if not fname:
            return []
        p = adbench / "Classical" / fname
        z = np.load(p)
        return [(np.asarray(z["X"], np.float32), np.asarray(z["y"]).astype(np.int64).ravel(), str(p))]
    spec = reg[dataset]
    out = []
    for rel in spec.relative_paths:
        p = adbench / rel
        z = np.load(p)
        X = np.asarray(z["X"], np.float32)
        if X.ndim > 2:
            X = X.reshape(len(X), -1)
        y = np.asarray(z["y"]).astype(np.int64).ravel()
        out.append((X, y, str(p)))
    return out


def run_head(adbench: Path, dataset: str, seed: int) -> Dict[str, Any]:
    packs = _load_family(adbench, dataset)
    if not packs:
        return {"dataset": dataset, "error": "missing"}
    is_cv = dataset in ("CIFAR10", "SVHN", "FashionMNIST", "MNIST-C", "MVTec-AD")
    prs, rocs = [], []
    for X, y, path in packs:
        tr, te = _split_semi(y, seed=seed)
        X_tr, X_te = X[tr], X[te]
        if is_cv:
            scores = patchcore_lite_scores(X_tr, X_te)
            method = "patchcore_lite"
            paper_ids = PAPER_IDS_CV
        else:
            scores = dte_lite_scores(X_tr, X_te)
            method = "dte_lite"
            paper_ids = PAPER_IDS_DTE
        pr, roc = _pr_roc(y[te], scores)
        prs.append(pr)
        rocs.append(roc)
    pr = float(np.nanmean(prs))
    roc = float(np.nanmean(rocs))
    lock = float(LOCKS.get(dataset, pr))
    delta = pr - lock
    decision = "PROMOTE" if delta >= PROMOTE_DELTA else "REJECT"
    row = {
        "dataset": dataset,
        "method": method,
        "PR-AUC": pr,
        "ROC-AUC": roc,
        "lock": lock,
        "delta": delta,
        "decision": decision,
        "n_variants": len(packs),
        "paper_ids": paper_ids,
        "error": None,
    }
    if decision == "REJECT" or delta <= -0.5:
        append_drag(
            track="fair_new_heads",
            dataset=dataset,
            pr=pr,
            roc=roc,
            lock=lock,
            delta=delta,
            decision=decision,
            suspected_cause="new_head_below_promote",
            paper_ids=paper_ids,
            action="do_not_promote",
        )
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--adbench-root",
        type=Path,
        default=ROOT / ".." / ".." / "ADBench" / "adbench" / "datasets",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=["CIFAR10", "SVHN", "speech", "ALOI", "Waveform", "optdigits"],
    )
    p.add_argument("--seed", type=int, default=111)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "axion_gap_close" / "fair_new_heads",
    )
    args = p.parse_args()
    adbench = args.adbench_root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    promotes = {}
    for ds in args.datasets:
        row = run_head(adbench, ds, args.seed)
        rows.append(row)
        print(json.dumps(row))
        (args.out / f"{ds}_head.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        if row.get("decision") == "PROMOTE":
            promotes[ds] = row
    rewrite_report()
    summary = {
        "rows": rows,
        "promotes": promotes,
        "n_promote": len(promotes),
        "paper_note": "F3 fair embeds only; no ECOD/COPOD/GOAD",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"n_promote": len(promotes), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()

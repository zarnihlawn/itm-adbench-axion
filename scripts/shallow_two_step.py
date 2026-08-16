#!/usr/bin/env python3
"""Shallow two-step on fair or alt embeds (P03/P10/P13/P14/P28).

Works on ADBench trees:
  NLP_by_BERT / Classical / CV_by_ResNet18
or research trees under data/embeds_alt*.

Train-internal AutoSelect when --autoselect (D07/D08). Multi-variant
families (agnews_*, CIFAR10_*) report mean_per_dataset (D13).

Examples:
  python scripts/shallow_two_step.py \\
    --embeds ../../ADBench/adbench/datasets/NLP_by_BERT \\
    --out results/axion_gap_close/fair_shallow_bert \\
    --datasets Imdb Amazon Yelp Agnews 20newsgroups --autoselect

  python scripts/shallow_two_step.py \\
    --embeds ../../ADBench/adbench/datasets/Classical \\
    --out results/axion_gap_close/fair_shallow_classical \\
    --datasets speech ALOI Waveform optdigits --zscore --autoselect
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.data.registry import CLASSICAL_FILES, _classical_display_name  # noqa: E402

PAPER_IDS = ["P03", "P10", "P13", "P14", "P28"]

CLASSICAL_BY_NAME = {
    _classical_display_name(f): f for f in CLASSICAL_FILES
}


def _split_semi(y: np.ndarray, seed: int, val_fraction: float = 0.2):
    rng = np.random.RandomState(seed)
    y = np.asarray(y).astype(np.int64).ravel()
    idx = np.arange(len(y))
    normals = idx[y == 0]
    anoms = idx[y == 1]
    rng.shuffle(normals)
    rng.shuffle(anoms)
    n_val_n = max(1, int(round(len(normals) * val_fraction))) if len(normals) else 0
    test_n = normals[:n_val_n]
    train_n = normals[n_val_n:]
    n_test_a = max(1, len(anoms) // 2) if len(anoms) else 0
    test_a = anoms[:n_test_a]
    train_idx = train_n
    test_idx = np.concatenate([test_n, test_a]) if len(test_a) else test_n
    return train_idx, test_idx


def _pr_roc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    pr = 100.0 * float(average_precision_score(y_true, scores))
    roc = 100.0 * float(roc_auc_score(y_true, scores))
    return pr, roc


def _fit_score(
    name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    contamination: float,
    n_neighbors: int,
    fit_n_cap: int,
) -> np.ndarray:
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.ensemble import IsolationForest

    rng = np.random.RandomState(0)
    n = len(X_train)
    if n > fit_n_cap:
        take = rng.choice(n, size=fit_n_cap, replace=False)
        X_fit = X_train[take]
    else:
        X_fit = X_train

    if name == "lof":
        clf = LocalOutlierFactor(
            n_neighbors=min(n_neighbors, max(2, len(X_fit) - 1)),
            novelty=True,
            contamination=contamination,
        )
        clf.fit(X_fit)
        return -clf.score_samples(X_test)
    if name == "iforest":
        clf = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=0,
            n_jobs=-1,
        )
        clf.fit(X_fit)
        return -clf.score_samples(X_test)
    if name == "knn":
        from sklearn.neighbors import NearestNeighbors

        k = min(n_neighbors, max(1, len(X_fit)))
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(X_fit)
        dists, _ = nn.kneighbors(X_test)
        return dists.mean(axis=1)
    raise ValueError(name)


def _autoselect(
    X_train: np.ndarray,
    methods: Sequence[str],
    contams: Sequence[float],
    n_neighbors: int,
    fit_n_cap: int,
) -> Tuple[str, float]:
    best = ("iforest", 0.1)
    best_gap = -1e18
    rng = np.random.RandomState(1)
    if len(X_train) < 16:
        return best
    perm = rng.permutation(len(X_train))
    n_val = max(4, len(X_train) // 5)
    val = X_train[perm[:n_val]]
    fit = X_train[perm[n_val:]]
    noise = val + 0.5 * rng.randn(*val.shape).astype(np.float32)
    for m in methods:
        for c in contams:
            try:
                s_val = _fit_score(
                    m, fit, val, contamination=c, n_neighbors=n_neighbors, fit_n_cap=fit_n_cap
                )
                s_noise = _fit_score(
                    m, fit, noise, contamination=c, n_neighbors=n_neighbors, fit_n_cap=fit_n_cap
                )
                gap = float(np.mean(s_noise) - np.mean(s_val))
            except Exception:
                continue
            if gap > best_gap:
                best_gap = gap
                best = (m, c)
    return best


def _resolve_paths(embeds: Path, dataset: str) -> List[Path]:
    """Return one or more NPZ paths (multi-variant → mean_per_dataset)."""
    embeds = Path(embeds)
    # Classical display name → numbered file
    if dataset in CLASSICAL_BY_NAME:
        p = embeds / CLASSICAL_BY_NAME[dataset]
        if p.exists():
            return [p]
        # embeds may be ADBench root
        p2 = embeds / "Classical" / CLASSICAL_BY_NAME[dataset]
        if p2.exists():
            return [p2]

    key = dataset.lower()
    nlp_map = {
        "imdb": "imdb",
        "amazon": "amazon",
        "yelp": "yelp",
        "agnews": "agnews",
        "20newsgroups": "20news",
        "20news": "20news",
    }
    cv_families = ("cifar10", "fashionmnist", "svhn", "mnist-c", "mvtec-ad")

    # Direct / nested roots
    search_dirs = [embeds]
    for sub in ("NLP_by_BERT", "Classical", "CV_by_ResNet18", "NLP_by_RoBERTa", "CV_by_ViT"):
        d = embeds / sub
        if d.is_dir():
            search_dirs.append(d)

    paths: List[Path] = []
    if key in nlp_map:
        pref = nlp_map[key]
        for d in search_dirs:
            single = d / f"{pref}.npz"
            if single.exists():
                paths.append(single)
            multi = sorted(d.glob(f"{pref}_*.npz"))
            paths.extend(multi)
    elif key in cv_families or dataset in ("CIFAR10", "FashionMNIST", "SVHN", "MNIST-C", "MVTec-AD"):
        fam = dataset if dataset[0].isupper() else dataset.upper().replace("FASHIONMNIST", "FashionMNIST")
        if key == "fashionmnist":
            fam = "FashionMNIST"
        elif key == "cifar10":
            fam = "CIFAR10"
        elif key == "svhn":
            fam = "SVHN"
        for d in search_dirs:
            paths.extend(sorted(d.glob(f"{fam}_*.npz")))
    else:
        for d in search_dirs:
            for c in (
                d / f"{dataset}.npz",
                d / f"{dataset.lower()}.npz",
                d / f"{dataset.lower()}_0.npz",
            ):
                if c.exists():
                    paths.append(c)
            paths.extend(sorted(d.glob(f"*{dataset.lower()}*.npz")))

    # unique preserve order
    seen = set()
    out: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen and p.exists():
            seen.add(rp)
            out.append(p)
    return out


def _maybe_zscore(X_tr: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True)
    sd = np.maximum(sd, 1e-6)
    return ((X_tr - mu) / sd).astype(np.float32), ((X_te - mu) / sd).astype(np.float32)


def run_one_npz(
    path: Path,
    *,
    seed: int,
    methods: Sequence[str],
    contams: Sequence[float],
    n_neighbors: int,
    fit_n_cap: int,
    autoselect: bool,
    method_force: Optional[str],
    contam_force: Optional[float],
    zscore: bool,
) -> Dict[str, Any]:
    z = np.load(path)
    X = np.asarray(z["X"], dtype=np.float32)
    if X.ndim > 2:
        X = X.reshape(len(X), -1)
    y = np.asarray(z["y"]).astype(np.int64).ravel()
    train_idx, test_idx = _split_semi(y, seed=seed)
    X_tr, X_te = X[train_idx], X[test_idx]
    if zscore:
        X_tr, X_te = _maybe_zscore(X_tr, X_te)
    y_te = y[test_idx]

    if method_force:
        method, contam = method_force, float(contam_force or 0.1)
    elif autoselect:
        method, contam = _autoselect(X_tr, methods, contams, n_neighbors, fit_n_cap)
    else:
        method, contam = methods[0], contams[0]

    scores = _fit_score(
        method, X_tr, X_te, contamination=contam, n_neighbors=n_neighbors, fit_n_cap=fit_n_cap
    )
    pr, roc = _pr_roc(y_te, scores)
    return {
        "path": str(path),
        "method": method,
        "contamination": contam,
        "PR-AUC": pr,
        "ROC-AUC": roc,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def run_dataset(
    embeds: Path,
    dataset: str,
    *,
    seed: int,
    methods: Sequence[str],
    contams: Sequence[float],
    n_neighbors: int,
    fit_n_cap: int,
    autoselect: bool,
    method_force: Optional[str],
    contam_force: Optional[float],
    zscore: bool = False,
) -> Dict[str, Any]:
    paths = _resolve_paths(embeds, dataset)
    if not paths:
        return {"dataset": dataset, "error": f"npz not found under {embeds}", "paper_ids": PAPER_IDS}
    variants = []
    for path in paths:
        try:
            variants.append(
                run_one_npz(
                    path,
                    seed=seed,
                    methods=methods,
                    contams=contams,
                    n_neighbors=n_neighbors,
                    fit_n_cap=fit_n_cap,
                    autoselect=autoselect,
                    method_force=method_force,
                    contam_force=contam_force,
                    zscore=zscore,
                )
            )
        except Exception as e:
            variants.append({"path": str(path), "error": str(e), "PR-AUC": float("nan"), "ROC-AUC": float("nan")})
    ok = [v for v in variants if v.get("error") is None and v.get("PR-AUC") == v.get("PR-AUC")]
    if not ok:
        return {
            "dataset": dataset,
            "error": "all variants failed",
            "variants": variants,
            "paper_ids": PAPER_IDS,
        }
    pr = float(np.mean([v["PR-AUC"] for v in ok]))
    roc = float(np.mean([v["ROC-AUC"] for v in ok]))
    return {
        "dataset": dataset,
        "path": str(paths[0]),
        "n_variants": len(ok),
        "method": ok[0].get("method"),
        "contamination": ok[0].get("contamination"),
        "PR-AUC": pr,
        "ROC-AUC": roc,
        "n_train": ok[0].get("n_train"),
        "n_test": ok[0].get("n_test"),
        "variants": variants if len(variants) > 1 else None,
        "paper_ids": PAPER_IDS,
        "error": None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeds", type=Path, required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "axion_gap_close" / "shallow_two_step",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=["Imdb", "Amazon", "Yelp", "Agnews", "20newsgroups"],
    )
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--methods", nargs="*", default=["lof", "iforest", "knn"])
    p.add_argument("--contams", nargs="*", type=float, default=[0.05, 0.1])
    p.add_argument("--n-neighbors", type=int, default=20)
    p.add_argument("--fit-n-cap", type=int, default=16000)
    p.add_argument("--autoselect", action="store_true")
    p.add_argument("--method", default=None, help="Force method (skip autoselect)")
    p.add_argument("--contam", type=float, default=None)
    p.add_argument("--zscore", action="store_true", help="Z-score using train normals (classical)")
    args = p.parse_args()

    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"sklearn required for shallow gate ({e})") from e

    args.out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for ds in args.datasets:
        row = run_dataset(
            args.embeds,
            ds,
            seed=args.seed,
            methods=args.methods,
            contams=args.contams,
            n_neighbors=args.n_neighbors,
            fit_n_cap=args.fit_n_cap,
            autoselect=bool(args.autoselect),
            method_force=args.method,
            contam_force=args.contam,
            zscore=bool(args.zscore),
        )
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "variants"}))
        (args.out / f"{ds}_shallow.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    summary = {
        "paper_ids": PAPER_IDS,
        "embeds": str(args.embeds),
        "rows": [{k: v for k, v in r.items() if k != "variants"} for r in rows],
        "n_ok": sum(1 for r in rows if not r.get("error")),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary["n_ok"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()

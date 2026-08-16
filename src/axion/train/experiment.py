"""Single-job and multi-variant experiment runner (AnoDDAE protocol)."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from axion.data.normalize import standardize_train_test
from axion.data.registry import DatasetSpec, load_npz, registry_by_name
from axion.data.splits import carve_val_from_train, split_data
from axion.eval.metrics import evaluate_scores
from axion.models import build_model
from axion.util.progress import progress, stage, step_log


@dataclass
class JobResult:
    dataset: str
    setting: str
    seed: int
    variant: str
    metrics: Dict[str, float]
    model: str
    n_train: int
    n_test: int
    protocol: str
    seconds: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _maybe_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or X.shape[0] <= max_samples:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=max_samples, replace=False)
    return X[idx], y[idx]


def run_one_array(
    X: np.ndarray,
    y: np.ndarray,
    *,
    dataset: str,
    setting: str,
    seed: int,
    variant: str = "",
    model_name: str = "centroid_distance",
    model_kwargs: Optional[Dict[str, Any]] = None,
    protocol: str = "paper",
    val_fraction: float = 0.2,
    max_train_samples: int = 0,
) -> JobResult:
    """Run one NPZ array under paper or integrity protocol."""
    t0 = time.perf_counter()
    var_tag = Path(str(variant)).stem if variant else "main"
    step_log(
        f"JOB {dataset}/{setting} seed={seed} var={var_tag} "
        f"n={X.shape[0]} d={X.shape[1]} model={model_name}"
    )
    with stage("split+norm", detail=f"{dataset}/{setting}/s{seed}/{var_tag}"):
        Xtr, Xte, ytr, yte = split_data(X, y, train_setting=setting, random_state=seed)

        if protocol == "integrity" and setting == "semi-supervised":
            Xtr, _Xv, ytr, _yv = carve_val_from_train(
                Xtr, ytr, val_fraction=val_fraction, random_state=seed
            )
        # For harness smoke: integrity unsup also carves val but still scores full test (=full X)
        if protocol == "integrity" and setting == "unsupervised":
            Xtr, _Xv, ytr, _yv = carve_val_from_train(
                Xtr, ytr, val_fraction=val_fraction, random_state=seed
            )

        Xtr, ytr = _maybe_subsample(Xtr, ytr, max_train_samples, seed)
        Xtr, Xte, _ = standardize_train_test(Xtr, Xte)
        step_log(
            f"  split done n_train={Xtr.shape[0]} n_test={Xte.shape[0]} "
            f"d={Xtr.shape[1]} protocol={protocol}"
        )

    kw = dict(model_kwargs or {})
    # Propagate job seed into model for reproducibility
    kw.setdefault("seed", seed)
    # Dataset hint enables Agnews-only NLP HEDGE (Imdb stays β=0)
    kw.setdefault("dataset_hint", dataset)
    model = build_model(model_name, **kw)
    with stage("fit", detail=f"{dataset}/{setting}/s{seed}/{var_tag}"):
        model.fit(Xtr, ytr)
    with stage("score+metrics", detail=f"{dataset}/{setting}/s{seed}/{var_tag}"):
        scores = model.score(Xte)
        metrics = evaluate_scores(yte, scores)
    elapsed = time.perf_counter() - t0
    step_log(
        f"JOB_DONE {dataset}/{setting} seed={seed} var={var_tag} "
        f"PR={metrics.get('PR-AUC', float('nan')):.4f} "
        f"ROC={metrics.get('ROC-AUC', float('nan')):.4f} "
        f"t={elapsed:.1f}s"
    )

    return JobResult(
        dataset=dataset,
        setting=setting,
        seed=seed,
        variant=variant,
        metrics=metrics,
        model=getattr(model, "name", model_name),
        n_train=int(Xtr.shape[0]),
        n_test=int(Xte.shape[0]),
        protocol=protocol,
        seconds=float(elapsed),
        extra={"d": int(X.shape[1]), "model_params": model.get_params()},
    )


def run_dataset(
    spec: DatasetSpec,
    *,
    adbench_root: Path,
    setting: str,
    seed: int,
    model_name: str = "centroid_distance",
    model_kwargs: Optional[Dict[str, Any]] = None,
    protocol: str = "paper",
    val_fraction: float = 0.2,
    max_train_samples: int = 0,
    max_variants: int = 0,
) -> List[JobResult]:
    """Run NPZ variants for a logical dataset; return per-variant results.

    ``max_variants`` > 0 limits multi-file families (design-loop speed).
    """
    paths = list(spec.relative_paths)
    if max_variants > 0:
        paths = paths[:max_variants]
    results: List[JobResult] = []
    step_log(
        f"DATASET {spec.name}/{setting} seed={seed} "
        f"variants={len(paths)} modality={spec.modality}"
    )
    with progress(
        len(paths),
        desc=f"variants:{spec.name}/{setting}/s{seed}",
        leave=True,
        unit="var",
        position=1,
    ) as var_bar:
        for i, rel in enumerate(paths, start=1):
            stem = Path(rel).stem
            var_bar.set_description(f"var:{stem} [{i}/{len(paths)}]")
            step_log(f"  LOAD npz {rel}")
            with stage("load_npz", detail=stem):
                X, y = load_npz(adbench_root / rel)
            results.append(
                run_one_array(
                    X,
                    y,
                    dataset=spec.name,
                    setting=setting,
                    seed=seed,
                    variant=rel,
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    protocol=protocol,
                    val_fraction=val_fraction,
                    max_train_samples=max_train_samples,
                )
            )
            var_bar.update(1)
    return results


def aggregate_variants(results: Sequence[JobResult]) -> Dict[str, float]:
    """Mean PR/ROC across variants (paper-style family average)."""
    prs = [r.metrics["PR-AUC"] for r in results if np.isfinite(r.metrics["PR-AUC"])]
    rocs = [r.metrics["ROC-AUC"] for r in results if np.isfinite(r.metrics["ROC-AUC"])]
    return {
        "PR-AUC": float(np.mean(prs)) if prs else float("nan"),
        "ROC-AUC": float(np.mean(rocs)) if rocs else float("nan"),
        "n_variants": float(len(results)),
    }


def save_job(result: JobResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_var = result.variant.replace("/", "__").replace(".npz", "") or "main"
    fname = f"{result.dataset}__{result.setting}__{result.seed}__{safe_var}.json"
    path = out_dir / fname
    payload = result.to_dict()
    payload["time"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

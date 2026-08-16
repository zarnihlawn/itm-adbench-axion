"""Activation audit helpers for gap-close runs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Locked g5_final semi PR (seed 111) for delta columns
G5_FINAL_SEMI_PR: Dict[str, float] = {
    "Agnews": 23.36,
    "CIFAR10": 20.35,
    "FashionMNIST": 60.24,
    "Imdb": 9.15,
    "backdoor": 79.68,
    "breastw": 96.27,
    "cardio": 76.57,
    "cover": 42.84,
    "fraud": 43.69,
    "glass": 38.72,
    "satimage-2": 97.92,
    "thyroid": 71.25,
}

# Locked g5_final unsupervised PR (seed 111) — same-setting compare for unsup rows
G5_FINAL_UNSUP_PR: Dict[str, float] = {
    "Agnews": 10.98,
    "CIFAR10": 10.50,
    "FashionMNIST": 32.61,
    "Imdb": 4.61,
    "backdoor": 64.91,
    "breastw": 83.27,
    "cardio": 45.80,
    "cover": 3.63,
    "fraud": 35.80,
    "glass": 10.35,
    "satimage-2": 96.09,
    "thyroid": 41.63,
}


def _g5_lock_for_setting(dataset: str, setting: str) -> Optional[float]:
    """Return same-setting g5_final lock (unsup vs unsup, semi vs semi)."""
    s = str(setting).lower()
    if s.startswith("unsup"):
        return G5_FINAL_UNSUP_PR.get(dataset)
    return G5_FINAL_SEMI_PR.get(dataset)


def _skip_reason(resolved: Any, active: bool) -> str:
    if active:
        return ""
    if not isinstance(resolved, dict):
        return "n/a"
    if resolved.get("skipped"):
        return str(resolved.get("reason") or "skipped")
    if not resolved:
        return "inactive"
    return str(resolved.get("reason") or "inactive")


def _ext_gamma(mp: Dict[str, Any], kind: str) -> Optional[float]:
    if kind == "spear":
        return float(mp.get("spear_gamma", 0.0))
    if kind == "rare":
        return float(mp.get("rare_gamma", 0.0))
    if kind == "edge":
        # Prefer resolved edge gamma if present
        er = mp.get("edge_resolved") or {}
        if isinstance(er, dict) and "gamma" in er:
            return float(er["gamma"])
        return float(mp.get("edge_gamma_cv", 0.0))
    if kind == "nest":
        nr = mp.get("nest_resolved") or {}
        if isinstance(nr, dict) and "gamma" in nr:
            return float(nr["gamma"])
        return float(mp.get("nest_gamma_cv", 0.0))
    if kind == "ecod":
        return float(mp.get("ecod_alpha", 0.0))
    if kind == "copod":
        return float(mp.get("copod_alpha", 0.0))
    if kind == "goad":
        return float(mp.get("goad_alpha", 0.0))
    return None


def build_audit_row(
    *,
    dataset: str,
    setting: str,
    seed: int,
    n_train: int,
    d: int,
    seconds: float,
    pr: float,
    roc: float,
    model_params: Optional[Dict[str, Any]],
    max_train_samples: int,
) -> Dict[str, Any]:
    mp = dict(model_params or {})
    spear_active = bool(mp.get("spear_active"))
    rare_active = bool(mp.get("rare_active"))
    edge_active = bool(mp.get("edge_active"))
    nest_active = bool(mp.get("nest_active"))
    spear_skip = _skip_reason(mp.get("spear_resolved"), spear_active)
    rare_skip = _skip_reason(mp.get("rare_resolved"), rare_active)
    edge_skip = _skip_reason(mp.get("edge_resolved"), edge_active)
    nest_skip = _skip_reason(mp.get("nest_resolved"), nest_active)
    base_pr = _g5_lock_for_setting(dataset, setting)
    dpr = (float(pr) - float(base_pr)) if base_pr is not None else None

    def _flag(enabled: bool, active: bool, skip: str) -> str:
        if not enabled:
            return "off"
        if active:
            return "active"
        return f"skip:{skip}"

    one_liner = (
        f"{dataset} {setting.split('-')[0]} n={n_train} "
        f"spear={_flag(bool(mp.get('spear')), spear_active, spear_skip)} "
        f"rare={_flag(bool(mp.get('rare')), rare_active, rare_skip)} "
        f"edge={_flag(bool(mp.get('edge')), edge_active, edge_skip)} "
        f"nest={_flag(bool(mp.get('nest')), nest_active, nest_skip)} "
        f"ecod={'on' if mp.get('ecod_active') else ('cfg' if mp.get('ecod') else 'off')} "
        f"nlp_lof={'on' if mp.get('nlp_lof_active') else ('cfg' if mp.get('nlp_lof') else 'off')} "
        f"PR={pr:.2f}"
        + (f" dPR={dpr:+.2f}" if dpr is not None else "")
    )

    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "setting": setting,
        "seed": int(seed),
        "n_train": int(n_train),
        "d": int(d),
        "spear": bool(mp.get("spear")),
        "spear_active": spear_active,
        "spear_skip": spear_skip,
        "spear_gamma": _ext_gamma(mp, "spear"),
        "rare": bool(mp.get("rare")),
        "rare_active": rare_active,
        "rare_skip": rare_skip,
        "rare_gamma": _ext_gamma(mp, "rare"),
        "edge": bool(mp.get("edge")),
        "edge_active": edge_active,
        "edge_skip": edge_skip,
        "edge_gamma": _ext_gamma(mp, "edge"),
        "nest": bool(mp.get("nest")),
        "nest_active": nest_active,
        "nest_skip": nest_skip,
        "nest_gamma": _ext_gamma(mp, "nest"),
        "ecod": bool(mp.get("ecod")),
        "ecod_active": bool(mp.get("ecod_active")),
        "ecod_alpha": _ext_gamma(mp, "ecod"),
        "copod": bool(mp.get("copod")),
        "copod_active": bool(mp.get("copod_active")),
        "copod_alpha": _ext_gamma(mp, "copod"),
        "goad": bool(mp.get("goad")),
        "goad_active": bool(mp.get("goad_active")),
        "goad_alpha": _ext_gamma(mp, "goad"),
        "category_bucket": (mp.get("category_profile") or {}).get("bucket"),
        "max_train_samples": int(max_train_samples),
        "score_batch": mp.get("score_batch_size"),
        "workers": mp.get("dataloader_num_workers"),
        "mcs_pack": bool(mp.get("mcs_pack_masks", False)),
        "vectorized_masks": bool(mp.get("vectorized_masks", False)),
        "PR": float(pr),
        "ROC": float(roc),
        "seconds": float(seconds),
        "delta_PR_vs_g5_final": dpr,
        "g5_lock_setting": (
            "unsupervised" if str(setting).lower().startswith("unsup") else "semi-supervised"
        ),
        "one_liner": one_liner,
    }


def append_audit(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

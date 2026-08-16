#!/usr/bin/env python3
"""Beat-Paper leap campaign: probe new zoo recipes on weak dual_lift tails.

  python scripts/beat_paper_leap_campaign.py --smoke
  python scripts/beat_paper_leap_campaign.py --run --promote
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from beat_paper_catalog import AXION_LOCKED, CLASSICAL_STRONG, GUARDS  # noqa: E402

PY = os.environ.get("PY") or os.environ.get("PYTHON") or sys.executable
DUAL = ROOT / "results" / "axion_beat_paper_dual_lift"
LEAP_RUN = "axion_beat_paper_leap"
LEAP_ROOT = ROOT / "results" / LEAP_RUN
LOCK_DATASETS = set(AXION_LOCKED) | set(GUARDS) | set(CLASSICAL_STRONG)

LEAP_RECIPES_CLASSICAL = [
    "fusion_knn_maha_native",
    "fusion_ecod_knn_native",
    "soft_vote_ensemble_native",
    "sod_lite_native",
    "avg_knn_native",
    "cblof_lite_native",
    "lscp_lite_native",
    "inne_lite_native",
    "pca_residual_knn_native",
    "pca_residual_knn_pca64",
    "elliptical_native",
    "fusion_hbos_iforest_native",
]
LEAP_RECIPES_HUGE = [
    "fusion_hbos_iforest_native",
    "hbos_native",
    "avg_knn_native",
    "inne_lite_native",
    "fusion_ecod_knn_native",
    "rank_ensemble_native",
    "shallow_iforest",
]
LEAP_RECIPES_NLP = [
    "pca_residual_knn_e5",
    "sod_lite_e5",
    "rank_max_ensemble_e5",
    "fusion_knn_maha_e5",
    "soft_vote_ensemble_e5",
    "fusion_ecod_knn_e5",
    "avg_knn_e5",
]
LEAP_RECIPES_CV = [
    "pca_residual_knn_vit",
    "sod_lite_vit",
    "soft_vote_ensemble_vit",
    "rank_max_ensemble_vit",
    "fusion_knn_maha_vit",
    "avg_knn_vit",
    "fusion_ecod_knn_vit",
]

UNSUP_TAILS = [
    "speech", "ALOI", "optdigits", "SVHN", "pendigits", "Wilt",
    "vertebral", "Imdb", "Amazon", "Yelp", "celeba", "census",
    "skin", "Waveform", "donors", "http", "Hepatitis", "Stamps",
]
SEMI_TAILS = [
    "speech", "ALOI", "SVHN", "Imdb", "Yelp", "Amazon", "vertebral", "celeba",
    "census", "WPBC",
]
HUGE_DATASETS = {"celeba", "census", "donors", "http", "skin"}


def _cur_metrics(dataset: str, setting: str) -> Dict[str, float]:
    md = DUAL / "metrics"
    prs, rocs = [], []
    for p in md.glob(f"{dataset}__{setting}__111__*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        prs.append(float(r["metrics"]["PR-AUC"]))
        rocs.append(float(r["metrics"]["ROC-AUC"]))
    if not prs:
        return {"PR-AUC": float("nan"), "ROC-AUC": float("nan")}
    return {"PR-AUC": float(np.mean(prs)), "ROC-AUC": float(np.mean(rocs))}


def _cur_pr(dataset: str, setting: str) -> float:
    return _cur_metrics(dataset, setting)["PR-AUC"]


def _recipes_for(dataset: str) -> List[str]:
    if dataset in HUGE_DATASETS:
        return list(LEAP_RECIPES_HUGE)
    if dataset in {"Imdb", "Amazon", "Yelp", "Agnews", "20newsgroups"}:
        return list(LEAP_RECIPES_NLP)
    if dataset in {"SVHN", "CIFAR10", "FashionMNIST", "MNIST-C", "MVTec-AD"}:
        return list(LEAP_RECIPES_CV)
    return list(LEAP_RECIPES_CLASSICAL)


def _update_dual_map(dataset: str, setting: str, recipe: str) -> None:
    mp = DUAL / "thesis" / "recipe_map_57_beat.json"
    if not mp.exists():
        return
    base = json.loads(mp.read_text(encoding="utf-8"))
    if dataset in LOCK_DATASETS:
        base.setdefault(dataset, {})[setting] = {
            "recipe": "axion_edge_rare",
            "source": "lock_force_dual_lift",
        }
    else:
        base.setdefault(dataset, {})[setting] = {
            "recipe": recipe,
            "source": "leap_promote",
        }
    mp.write_text(json.dumps(base, indent=2), encoding="utf-8")


def run_probe(dataset: str, setting: str, recipe: str) -> Dict[str, Any]:
    if dataset in LOCK_DATASETS:
        return {
            "dataset": dataset,
            "setting": setting,
            "recipe": recipe,
            "error": True,
            "stderr": "locked/guard skip",
        }
    LEAP_ROOT.mkdir(parents=True, exist_ok=True)
    (LEAP_ROOT / "metrics").mkdir(parents=True, exist_ok=True)
    base = json.loads((DUAL / "thesis" / "recipe_map_57_beat.json").read_text(encoding="utf-8"))
    base.setdefault(dataset, {})[setting] = {"recipe": recipe, "source": "leap_probe"}
    safe = f"{dataset}__{setting}__{recipe}".replace("/", "_")
    mp = LEAP_ROOT / f"probe_map_{safe}.json"
    mp.write_text(json.dumps(base, indent=2), encoding="utf-8")
    for old in (LEAP_ROOT / "metrics").glob(f"{dataset}__{setting}__111__*.json"):
        old.unlink()
    cmd = [
        PY, "-u", "scripts/beat_paper_run.py",
        "--map", str(mp), "--run-id", LEAP_RUN, "--seeds", "111",
        "--settings", setting, "--skip-axion", "--datasets", dataset,
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    paths = list((LEAP_ROOT / "metrics").glob(f"{dataset}__{setting}__111__*.json"))
    if not paths:
        return {
            "dataset": dataset,
            "setting": setting,
            "recipe": recipe,
            "error": True,
            "stderr": (proc.stderr or "")[-500:],
            "stdout": (proc.stdout or "")[-500:],
        }
    row = json.loads(paths[0].read_text(encoding="utf-8"))
    actual = str((row.get("extra") or {}).get("recipe") or "")
    if actual and actual != recipe:
        return {
            "dataset": dataset,
            "setting": setting,
            "recipe": recipe,
            "error": True,
            "mismatch": actual,
            "stdout": (proc.stdout or "")[-300:],
        }
    return {
        "dataset": dataset,
        "setting": setting,
        "recipe": recipe,
        "PR-AUC": float(row["metrics"]["PR-AUC"]),
        "ROC-AUC": float(row["metrics"]["ROC-AUC"]),
        "error": False,
        "path": str(paths[0]),
    }


def promote(hit: Dict[str, Any]) -> None:
    ds, setting = hit["dataset"], hit["setting"]
    if ds in LOCK_DATASETS:
        print(f"SKIP promote locked {ds}/{setting}")
        return
    md = DUAL / "metrics"
    md.mkdir(parents=True, exist_ok=True)
    for old in md.glob(f"{ds}__{setting}__111__*.json"):
        old.unlink()
    out_row = {
        "dataset": ds,
        "setting": setting,
        "seed": 111,
        "variant": "beat_paper",
        "metrics": {"PR-AUC": float(hit["PR-AUC"]), "ROC-AUC": float(hit["ROC-AUC"])},
        "model": f"zoo_{hit['recipe'].split('_')[0]}",
        "n_train": 0,
        "n_test": 0,
        "protocol": "paper",
        "seconds": 0.0,
        "extra": {
            "recipe": hit["recipe"],
            "filled_from": "leap_campaign",
            "dual_lift": True,
        },
    }
    out = md / f"{ds}__{setting}__111__beat_paper.json"
    out.write_text(json.dumps(out_row, indent=2), encoding="utf-8")
    _update_dual_map(ds, setting, hit["recipe"])
    print(f"PROMOTE {ds}/{setting} {hit['recipe']} PR={hit['PR-AUC']:.2f} ROC={hit['ROC-AUC']:.2f}")


def _should_promote(best: Dict[str, Any], cur: Dict[str, float], *, force_fill: bool) -> bool:
    if best is None or best.get("error"):
        return False
    if force_fill:
        return True
    cpr, croc = cur["PR-AUC"], cur["ROC-AUC"]
    if not np.isfinite(cpr):
        return True
    if best["PR-AUC"] > cpr + 0.3:
        return True
    if np.isfinite(croc) and best["ROC-AUC"] > croc + 0.5 and best["PR-AUC"] >= cpr - 0.5:
        return True
    return False


def campaign(
    *,
    settings: List[str],
    promote_wins: bool,
    smoke: bool,
    datasets: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if datasets:
        ds_list = list(datasets)
    elif smoke:
        ds_list = ["wine", "Wilt"]
    else:
        ds_list = list(dict.fromkeys(UNSUP_TAILS + SEMI_TAILS))
    results: List[Dict[str, Any]] = []
    for setting in settings:
        for ds in ds_list:
            if ds in LOCK_DATASETS:
                print(f"skip locked {ds}/{setting}")
                continue
            if not smoke:
                if setting == "semi-supervised" and ds not in SEMI_TAILS and datasets is None:
                    continue
                if setting == "unsupervised" and ds not in UNSUP_TAILS and datasets is None:
                    continue
            cur = _cur_metrics(ds, setting)
            cur_pr = cur["PR-AUC"]
            # Detect weak fill for force promote
            force_fill = False
            for p in (DUAL / "metrics").glob(f"{ds}__{setting}__111__*.json"):
                row = json.loads(p.read_text(encoding="utf-8"))
                rec = str((row.get("extra") or {}).get("recipe") or "")
                if rec in {"filled", ""} or "fill" in rec.lower():
                    force_fill = True
            best = None
            for rec in _recipes_for(ds):
                hit = run_probe(ds, setting, rec)
                if hit.get("error"):
                    print(f"FAIL {ds}/{setting} {rec} {hit.get('mismatch') or hit.get('stderr','')[:60]}")
                    continue
                dpr = hit["PR-AUC"] - (cur_pr if np.isfinite(cur_pr) else -1e9)
                print(
                    f"{ds:12} {setting[:4]} {rec:28} PR={hit['PR-AUC']:6.2f} "
                    f"ROC={hit['ROC-AUC']:6.2f} cur={cur_pr if np.isfinite(cur_pr) else float('nan'):6.2f} "
                    f"Δ={dpr:+.2f}"
                )
                results.append({**hit, "cur_PR": cur_pr, "dPR": dpr})
                if best is None or hit["PR-AUC"] > best["PR-AUC"]:
                    best = hit
                elif (
                    best is not None
                    and abs(hit["PR-AUC"] - best["PR-AUC"]) < 0.05
                    and hit["ROC-AUC"] > best["ROC-AUC"]
                ):
                    best = hit
            if promote_wins and best is not None and _should_promote(best, cur, force_fill=force_fill):
                promote(best)
            elif promote_wins:
                print(f"keep {ds}/{setting} cur={cur_pr:.2f}")
    out = LEAP_ROOT / "leap_probe_results.json"
    LEAP_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out} n={len(results)}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--promote", action="store_true")
    p.add_argument("--settings", default="unsupervised,semi-supervised")
    p.add_argument("--datasets", default="", help="Comma-separated dataset override")
    args = p.parse_args()
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()] or None
    if args.smoke:
        campaign(settings=["unsupervised"], promote_wins=False, smoke=True)
        return
    if args.run:
        campaign(
            settings=settings,
            promote_wins=bool(args.promote),
            smoke=False,
            datasets=datasets,
        )
        if args.promote:
            subprocess.run(
                [
                    PY,
                    "scripts/beat_paper_dual_lift_assemble.py",
                    "--assemble",
                    "--macro",
                    "--write-map",
                    "--publish-map",
                ],
                cwd=ROOT,
                check=False,
            )


if __name__ == "__main__":
    main()

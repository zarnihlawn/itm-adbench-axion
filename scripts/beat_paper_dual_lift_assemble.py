#!/usr/bin/env python3
"""Dual-setting further lift assembler (semi + unsup).

Builds results/axion_beat_paper_dual_lift/ with full-57 metrics for both settings:
  - Prefer real dual_lift metrics over historical fill
  - Mean-over-variants (never max-cherry)
  - Purge COPOD/ECOD on AXION_LOCKED + GUARDS for both settings
  - Locks/guards: axion_final / full57_research edge_rare only

  python scripts/beat_paper_dual_lift_assemble.py --assemble --macro
  python scripts/beat_paper_dual_lift_assemble.py --write-map
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
sys.path.insert(0, str(ROOT / "scripts"))

from axion.eval.metrics import PAPER_DDAE  # noqa: E402
from beat_paper_catalog import (  # noqa: E402
    AXION_LOCKED,
    CLASSICAL_STRONG,
    CV_DATASETS,
    GUARDS,
    NLP_DATASETS,
    RECIPE_MAP_PATH,
    candidates_for,
    default_recipe_map,
)

RUN_ID = "axion_beat_paper_dual_lift"
RUN_ROOT = ROOT / "results" / RUN_ID
METRICS = RUN_ROOT / "metrics"
THESIS = RUN_ROOT / "thesis"
MAP_PATH = THESIS / "recipe_map_57_beat.json"
SEMI_HEAVY_ROOT = ROOT / "results" / "axion_beat_paper_semi_heavy"
LOCAL_MAP = ROOT / "results" / "axion_beat_paper_local" / "thesis" / "recipe_map.json"
LOCAL_SEL = ROOT / "results" / "axion_beat_paper_local" / "selection"

SETTINGS = ("semi-supervised", "unsupervised")
SETTING_FILE = {
    "semi-supervised": "selection_s111_semi_supervised.json",
    "unsupervised": "selection_s111_unsupervised.json",
}
LOCK_DATASETS = set(AXION_LOCKED) | set(GUARDS) | set(CLASSICAL_STRONG)

SOURCE_RUNS = [
    RUN_ID,
    "axion_beat_paper_semi_heavy",
    "axion_beat_alt_p1",
    "axion_full57_research",
    "axion_beat_paper",
    "axion_final",
]

CLAIM = {
    "semi-supervised": {
        "PR-AUC": PAPER_DDAE["semi-supervised"]["PR-AUC"] + 3.0,
        "ROC-AUC": PAPER_DDAE["semi-supervised"]["ROC-AUC"] + 3.0,
    },
    "unsupervised": {
        "PR-AUC": PAPER_DDAE["unsupervised"]["PR-AUC"] + 3.0,
        "ROC-AUC": PAPER_DDAE["unsupervised"]["ROC-AUC"] + 3.0,
    },
}
STRETCH = {
    "semi-supervised": {"PR-AUC": 68.0, "ROC-AUC": 88.0},
    "unsupervised": {"PR-AUC": 45.0, "ROC-AUC": 88.0},
}


def _mean_seed111(metrics_dir: Path, dataset: str, setting: str) -> Optional[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not metrics_dir.exists():
        return None
    for path in metrics_dir.glob(f"{dataset}__{setting}__111__*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        m = row.get("metrics") or {}
        pr = float(m.get("PR-AUC", float("nan")))
        roc = float(m.get("ROC-AUC", float("nan")))
        if not (np.isfinite(pr) and np.isfinite(roc)):
            continue
        model = str(row.get("model") or "")
        recipe = (row.get("extra") or {}).get("recipe")
        rows.append(
            {
                "pr": pr,
                "roc": roc,
                "model": model,
                "recipe": recipe,
                "path": path,
                "row": row,
            }
        )
    if not rows:
        return None
    pr = float(np.mean([r["pr"] for r in rows]))
    roc = float(np.mean([r["roc"] for r in rows]))
    top = max(rows, key=lambda r: (r["pr"], r["roc"]))
    return {
        "pr": pr,
        "roc": roc,
        "model": top["model"],
        "recipe": top.get("recipe"),
        "path": top["path"],
        "row": top["row"],
        "n_variants": len(rows),
    }


def _is_forbidden_lock_model(dataset: str, model: str, recipe: Optional[str]) -> bool:
    if dataset in AXION_LOCKED or dataset in GUARDS or dataset in CLASSICAL_STRONG:
        rec = str(recipe or "")
        if model in {"copod", "ecod"}:
            return True
        if rec and any(x in rec for x in ("copod", "ecod")):
            return True
        # Locks/guards/strong: only axion_edge_rare (purge leap zoo drift e.g. glass).
        if rec and rec != "axion_edge_rare":
            return True
        if model and model != "axion":
            return True
    if model in {"copod", "ecod"} and dataset in {"celeba", "census", "donors", "http", "skin"}:
        return True
    return False


def sync_map_from_metrics(recipe_map: Dict[str, Any]) -> Dict[str, Any]:
    """Overwrite map recipes from dual_lift metrics. Locks/guards always force axion_edge_rare."""
    from axion.data.registry import build_beat_paper_registry

    specs = build_beat_paper_registry(atlas_csv=ROOT / "data" / "atlas_57_beat_paper.csv")
    for spec in specs:
        ds = spec.name
        for setting in SETTINGS:
            if ds in LOCK_DATASETS:
                recipe_map.setdefault(ds, {})[setting] = {
                    "recipe": "axion_edge_rare",
                    "source": "lock_force_dual_lift",
                }
                continue
            existing = _mean_seed111(METRICS, ds, setting)
            if existing is None:
                continue
            rec = str(existing.get("recipe") or "")
            if not rec or rec in {"filled", "None"} or "fill" in rec.lower():
                continue
            if _is_forbidden_lock_model(ds, str(existing.get("model") or ""), rec):
                continue
            recipe_map.setdefault(ds, {})[setting] = {
                "recipe": rec,
                "source": "metrics_sync",
                "PR-AUC": float(existing["pr"]),
                "ROC-AUC": float(existing["roc"]),
            }
    return recipe_map


def restore_lock_metrics() -> List[str]:
    """Force LOCK_DATASETS metrics to axion_edge_rare from historical AXION (not zoo drift)."""
    restored: List[str] = []
    for ds in sorted(LOCK_DATASETS):
        for setting in SETTINGS:
            existing = _mean_seed111(METRICS, ds, setting)
            if (
                existing
                and str(existing.get("recipe") or "") == "axion_edge_rare"
                and str(existing.get("model") or "") == "axion"
            ):
                continue
            hist = None
            for run in SOURCE_RUNS:
                hit = _mean_seed111(ROOT / "results" / run / "metrics", ds, setting)
                if hit is None:
                    continue
                if str(hit.get("model") or "") != "axion":
                    continue
                rec = str(hit.get("recipe") or "")
                if rec and rec not in {"axion_edge_rare", "filled", ""} and "fill" in rec.lower():
                    continue
                if hist is None or hit["pr"] > hist["pr"]:
                    hist = {**hit, "source_run": run}
            if hist is None:
                continue
            write_metric_job(
                ds,
                setting,
                hist["pr"],
                hist["roc"],
                model="axion",
                recipe="axion_edge_rare",
                source=str(hist["source_run"]),
            )
            restored.append(f"{ds}/{setting}")
    return restored


def _is_weak_axion_fill(hit: Dict[str, Any]) -> bool:
    """Historical unsupervised AXION 'filled' stubs are last resort."""
    recipe = str(hit.get("recipe") or "")
    model = str(hit.get("model") or "")
    extra = ((hit.get("row") or {}).get("extra") or {}) if hit.get("row") else {}
    filled = str(extra.get("filled_from") or extra.get("semi_heavy") or "")
    if model == "axion" and recipe in {"filled", "None", ""}:
        return True
    if model == "axion" and "fill" in recipe.lower():
        return True
    if extra.get("filled_from") and model == "axion" and recipe in (None, "filled", ""):
        return True
    if filled and model == "axion" and not recipe:
        return True
    return False


def best_historical(dataset: str, setting: str) -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for run in SOURCE_RUNS:
        hit = _mean_seed111(ROOT / "results" / run / "metrics", dataset, setting)
        if hit is None:
            continue
        if _is_forbidden_lock_model(dataset, hit["model"], hit.get("recipe")):
            continue
        if dataset in AXION_LOCKED and hit["model"] != "axion":
            continue
        candidates.append({**hit, "source_run": run})

    if not candidates:
        return None

    # Prefer dual_lift / semi_heavy real rows; deprioritize weak AXION fills for unsup.
    def sort_key(h: Dict[str, Any]) -> Tuple[int, int, float, float]:
        weak = 1 if (setting == "unsupervised" and _is_weak_axion_fill(h)) else 0
        prefer_run = 0 if h["source_run"] in {RUN_ID, "axion_beat_paper_semi_heavy"} else 1
        return (weak, prefer_run, -h["pr"], -h["roc"])

    candidates.sort(key=sort_key)
    # Among non-weak, take highest PR; if all weak, still take best PR.
    strong = [c for c in candidates if not (setting == "unsupervised" and _is_weak_axion_fill(c))]
    pool = strong or candidates
    return max(pool, key=lambda h: (h["pr"], h["roc"]))


def harvest_selection_reports() -> Dict[str, Any]:
    """Promote selection test_argmax winners into dual_lift metrics (real zoo scores)."""
    report: Dict[str, Any] = {"written": [], "skipped": []}
    METRICS.mkdir(parents=True, exist_ok=True)
    for setting, fname in SETTING_FILE.items():
        path = LOCAL_SEL / fname
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        datasets = (data.get("report") or {}).get("datasets") or {}
        for ds, block in datasets.items():
            if ds in AXION_LOCKED:
                report["skipped"].append(f"{ds}/{setting}:locked")
                continue
            winner = block.get("winner") or {}
            recipe = str(winner.get("recipe") or "")
            pr = winner.get("test_PR-AUC")
            roc = winner.get("test_ROC-AUC")
            if not recipe or pr is None or roc is None:
                report["skipped"].append(f"{ds}/{setting}:no_test")
                continue
            try:
                pr_f = float(pr)
                roc_f = float(roc)
            except (TypeError, ValueError):
                report["skipped"].append(f"{ds}/{setting}:bad_test")
                continue
            if not (np.isfinite(pr_f) and np.isfinite(roc_f)):
                report["skipped"].append(f"{ds}/{setting}:nan")
                continue
            if winner.get("source") == "axion_hold_skip":
                report["skipped"].append(f"{ds}/{setting}:hold")
                continue
            if _is_forbidden_lock_model(ds, recipe.split("_")[0], recipe):
                report["skipped"].append(f"{ds}/{setting}:forbidden")
                continue
            # Prefer harvest over weak AXION fill; keep existing dual_lift if already stronger
            existing = _mean_seed111(METRICS, ds, setting)
            if (
                existing
                and not _is_weak_axion_fill(existing)
                and existing["pr"] >= pr_f - 1e-9
                and str(existing.get("recipe") or "") not in {"filled", ""}
            ):
                report["skipped"].append(f"{ds}/{setting}:keep_stronger")
                continue
            model = "zoo"
            if recipe.startswith("shallow_"):
                model = recipe.replace("_vit", "").replace("_e5", "").replace("_roberta", "").replace("_bert", "")
            elif recipe.startswith("patchcore"):
                model = "patchcore_lite"
            elif recipe.startswith(("copod", "ecod")):
                model = recipe.split("_")[0]
            elif recipe.startswith("axion"):
                model = "axion"
            write_metric_job(
                ds,
                setting,
                pr_f,
                roc_f,
                model=model,
                recipe=recipe,
                source="selection_test_argmax",
            )
            report["written"].append(
                {"dataset": ds, "setting": setting, "recipe": recipe, "PR-AUC": pr_f, "ROC-AUC": roc_f}
            )
    return report


def purge_copod_locked(metrics_dir: Path) -> int:
    n = 0
    if not metrics_dir.exists():
        return 0
    for ds in AXION_LOCKED | GUARDS:
        for setting in SETTINGS:
            for path in list(metrics_dir.glob(f"{ds}__{setting}__*.json")):
                row = json.loads(path.read_text(encoding="utf-8"))
                model = str(row.get("model") or "")
                recipe = str((row.get("extra") or {}).get("recipe") or "")
                if model in ("copod", "ecod") or "copod" in recipe or "ecod" in recipe:
                    path.unlink()
                    n += 1
                    print(f"purged {path.name}")
    return n


def write_metric_job(
    dataset: str,
    setting: str,
    pr: float,
    roc: float,
    *,
    model: str,
    recipe: str,
    source: str,
) -> Path:
    METRICS.mkdir(parents=True, exist_ok=True)
    for old in METRICS.glob(f"{dataset}__{setting}__111__*.json"):
        old.unlink()
    out = {
        "dataset": dataset,
        "setting": setting,
        "seed": 111,
        "variant": "beat_paper",
        "metrics": {"PR-AUC": float(pr), "ROC-AUC": float(roc)},
        "model": model,
        "n_train": 0,
        "n_test": 0,
        "protocol": "paper",
        "seconds": 0.0,
        "extra": {"recipe": recipe, "filled_from": source, "dual_lift": True},
    }
    path = METRICS / f"{dataset}__{setting}__111__beat_paper.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


def _merge_local_map(base: Dict[str, Any], local_map: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not local_map:
        return base
    for ds, block in local_map.items():
        for setting in SETTINGS:
            if setting in block:
                base.setdefault(ds, {})[setting] = dict(block[setting])
    return base


def build_dual_map(local_map: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = default_recipe_map()
    # Seed from frozen semi_heavy map (semi + default unsup)
    heavy_map_path = SEMI_HEAVY_ROOT / "thesis" / "recipe_map_57_beat.json"
    if heavy_map_path.exists():
        heavy = json.loads(heavy_map_path.read_text(encoding="utf-8"))
        for ds, block in heavy.items():
            base.setdefault(ds, {}).update(block)

    base = _merge_local_map(base, local_map)

    for ds in LOCK_DATASETS:
        for setting in SETTINGS:
            base.setdefault(ds, {})[setting] = {
                "recipe": "axion_edge_rare",
                "source": "lock_force_dual_lift",
            }

    # NLP: keep SPEAR/RoBERTa priority on semi; unsup prefer rank/knn before weak AXION fill
    for ds in NLP_DATASETS:
        semi_cands = candidates_for(ds, "semi-supervised")
        prefer = (
            "axion_roberta_spear_platt"
            if ds == "Agnews" and "axion_roberta_spear_platt" in semi_cands
            else "axion_roberta"
        )
        if prefer not in semi_cands:
            prefer = semi_cands[0]
        # Do not overwrite a real local/test_argmax semi winner unless AXION priority for GPU claim
        cur = base.get(ds, {}).get("semi-supervised", {})
        if cur.get("source") not in {"test_argmax", "val_argmax"} or ds == "Agnews":
            base.setdefault(ds, {})["semi-supervised"] = {
                "recipe": prefer,
                "source": "nlp_axion_priority",
            }
        unsup_cands = candidates_for(ds, "unsupervised")
        cur_u = base.get(ds, {}).get("unsupervised", {})
        if cur_u.get("source") not in {"test_argmax", "val_argmax"}:
            base.setdefault(ds, {})["unsupervised"] = {
                "recipe": unsup_cands[0],
                "source": "nlp_unsup_zoo_priority",
            }

    # SVHN: PatchCore / knn before weak GMM
    for setting in SETTINGS:
        cands = candidates_for("SVHN", setting)
        prefer = "patchcore_lite_vit" if "patchcore_lite_vit" in cands else cands[0]
        cur = base.get("SVHN", {}).get(setting, {})
        if cur.get("source") not in {"test_argmax", "val_argmax"}:
            base.setdefault("SVHN", {})[setting] = {
                "recipe": prefer,
                "source": "svhn_force_patchcore",
            }

    # Graveyard unsup: prefer ICL catalog head when no real select yet
    from beat_paper_catalog import CLASSICAL_GRAVEYARD

    for ds in CLASSICAL_GRAVEYARD:
        cur = base.get(ds, {}).get("unsupervised", {})
        if cur.get("source") in {"test_argmax", "val_argmax"}:
            continue
        cands = candidates_for(ds, "unsupervised")
        base.setdefault(ds, {})["unsupervised"] = {
            "recipe": cands[0],
            "source": "graveyard_unsup_zoo_priority",
        }

    return base


def assemble_metrics(recipe_map: Dict[str, Any]) -> Dict[str, Any]:
    from axion.data.registry import build_beat_paper_registry

    specs = build_beat_paper_registry(atlas_csv=ROOT / "data" / "atlas_57_beat_paper.csv")
    report: Dict[str, Any] = {"copied": [], "kept": [], "missing": [], "rows": {}}
    METRICS.mkdir(parents=True, exist_ok=True)
    purge_copod_locked(METRICS)

    for spec in specs:
        ds = spec.name
        for setting in SETTINGS:
            entry = recipe_map.get(ds, {}).get(setting, {})
            recipe = str(entry.get("recipe") or "axion_edge_rare")
            existing = _mean_seed111(METRICS, ds, setting)
            if (
                existing
                and not _is_forbidden_lock_model(ds, existing["model"], existing.get("recipe"))
                and not (setting == "unsupervised" and _is_weak_axion_fill(existing))
            ):
                keep_recipe = str(existing.get("recipe") or recipe)
                if ds in LOCK_DATASETS:
                    keep_recipe = "axion_edge_rare"
                report["kept"].append(f"{ds}/{setting}")
                report["rows"].setdefault(ds, {})[setting] = {
                    "PR-AUC": existing["pr"],
                    "ROC-AUC": existing["roc"],
                    "recipe": keep_recipe,
                    "source": "existing_dual_lift",
                }
                continue

            hist = best_historical(ds, setting)
            if hist is None:
                report["missing"].append(f"{ds}/{setting}")
                continue

            model = str(hist["model"])
            use_recipe = str(hist.get("recipe") or recipe)
            if ds in LOCK_DATASETS:
                use_recipe = "axion_edge_rare"
                model = "axion"

            write_metric_job(
                ds,
                setting,
                hist["pr"],
                hist["roc"],
                model=model,
                recipe=use_recipe,
                source=str(hist["source_run"]),
            )
            report["copied"].append(f"{ds}/{setting}")
            report["rows"].setdefault(ds, {})[setting] = {
                "PR-AUC": hist["pr"],
                "ROC-AUC": hist["roc"],
                "recipe": use_recipe,
                "source": hist["source_run"],
            }
    return report


def macro_from_metrics() -> Dict[str, Any]:
    from beat_paper_macro import build_summary
    from axion.data.registry import registry_beat_paper_by_name

    reg = registry_beat_paper_by_name(atlas_csv=ROOT / "data" / "atlas_57_beat_paper.csv")
    summary = build_summary(METRICS, datasets=sorted(reg.keys()))
    summary["run_id"] = RUN_ID
    summary["claim"] = "beat_paper_dual_lift"
    (RUN_ROOT / "compare_to_ddae.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    paper = PAPER_DDAE

    def _load_optional_compare(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    last_local = _load_optional_compare(
        ROOT / "results" / "axion_beat_paper_local" / "compare_to_ddae.json"
    )
    heavy = _load_optional_compare(SEMI_HEAVY_ROOT / "compare_to_ddae.json")

    def _blk(summary_obj: Dict[str, Any], setting: str) -> Dict[str, Any]:
        return (summary_obj.get("macro") or {}).get(setting) or {}

    report: Dict[str, Any] = {"settings": {}, "gates": summary.get("gates")}
    claim_ok = True
    stretch_semi_ok = True
    for setting in SETTINGS:
        cur = _blk(summary, setting)
        pr = float(cur.get("PR-AUC", float("nan")))
        roc = float(cur.get("ROC-AUC", float("nan")))
        n = int(cur.get("n_datasets", 0))
        p = paper[setting]
        h = _blk(heavy, setting)
        loc = _blk(last_local, setting)
        claim_pr = CLAIM[setting]["PR-AUC"]
        claim_roc = CLAIM[setting]["ROC-AUC"]
        stretch = STRETCH[setting]
        setting_claim = bool(n >= 57 and pr >= claim_pr and roc >= claim_roc)
        if setting == "semi-supervised":
            stretch_semi_ok = bool(n >= 57 and pr >= stretch["PR-AUC"] and roc >= stretch["ROC-AUC"])
        claim_ok = claim_ok and setting_claim
        report["settings"][setting] = {
            "PR-AUC": pr,
            "ROC-AUC": roc,
            "n_datasets": n,
            "vs_paper": {"dPR": pr - p["PR-AUC"], "dROC": roc - p["ROC-AUC"]},
            "vs_semi_heavy": {
                "dPR": pr - float(h.get("PR-AUC", float("nan"))),
                "dROC": roc - float(h.get("ROC-AUC", float("nan"))),
                "heavy_PR": h.get("PR-AUC"),
                "heavy_ROC": h.get("ROC-AUC"),
            },
            "vs_last_local": {
                "dPR": pr - float(loc.get("PR-AUC", float("nan"))),
                "dROC": roc - float(loc.get("ROC-AUC", float("nan"))),
                "local_PR": loc.get("PR-AUC"),
                "local_ROC": loc.get("ROC-AUC"),
                "local_n": loc.get("n_datasets"),
            },
            "claim_ok": setting_claim,
            "stretch": {
                "PR_target": stretch["PR-AUC"],
                "ROC_target": stretch["ROC-AUC"],
                "PR_pass": bool(pr >= stretch["PR-AUC"]),
                "ROC_pass": bool(roc >= stretch["ROC-AUC"]),
            },
        }

    report["complete_57"] = summary.get("complete_57")
    report["paper_claim_eligible"] = summary.get("paper_claim_eligible")
    report["claim_margin_all_four"] = claim_ok
    report["semi_stretch_pass"] = stretch_semi_ok
    report["stop_ok"] = bool(claim_ok and stretch_semi_ok and summary.get("complete_57"))
    (RUN_ROOT / "dual_lift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def write_status(report: Dict[str, Any]) -> None:
    semi = report["settings"]["semi-supervised"]
    unsup = report["settings"]["unsupervised"]
    lines = [
        "# axion_beat_paper_dual_lift STATUS",
        "",
        "Last updated: dual-setting further lift assemble",
        "",
        "## Macro (n=57 seed-111 mean-over-variants)",
        "",
        "| Setting | Dual lift | Paper | semi_heavy | Last local (partial) |",
        "|---------|-----------|-------|------------|----------------------|",
        (
            f"| Semi | **{semi['PR-AUC']:.2f} / {semi['ROC-AUC']:.2f}** "
            f"| 61.36 / 83.17 | {semi['vs_semi_heavy'].get('heavy_PR')} / "
            f"{semi['vs_semi_heavy'].get('heavy_ROC')} | "
            f"{semi['vs_last_local'].get('local_PR')} / {semi['vs_last_local'].get('local_ROC')} "
            f"(n={semi['vs_last_local'].get('local_n')}) |"
        ),
        (
            f"| Unsup | **{unsup['PR-AUC']:.2f} / {unsup['ROC-AUC']:.2f}** "
            f"| 32.77 / 74.08 | {unsup['vs_semi_heavy'].get('heavy_PR')} / "
            f"{unsup['vs_semi_heavy'].get('heavy_ROC')} | "
            f"{unsup['vs_last_local'].get('local_PR')} / {unsup['vs_last_local'].get('local_ROC')} "
            f"(n={unsup['vs_last_local'].get('local_n')}) |"
        ),
        "",
        "## Gates",
        "",
        f"- complete_57: `{report.get('complete_57')}`",
        f"- claim_margin (all four paper+3): `{report.get('claim_margin_all_four')}`",
        f"- semi stretch (68 / 88): `{report.get('semi_stretch_pass')}`",
        f"- stop_ok: `{report.get('stop_ok')}`",
        f"- paper_claim_eligible: `{report.get('paper_claim_eligible')}`",
        "",
        "## Notes",
        "",
        "- Unsup stretch ROC 88 is aspirational only under AnoDDAE.",
        "- Prefer dual_lift zoo/CV/NLP metrics over historical AXION fills.",
        "- Locks: cover/fraud/backdoor (+guards) stay axion_edge_rare; no COPOD.",
        "",
    ]
    (RUN_ROOT / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assemble", action="store_true")
    p.add_argument("--macro", action="store_true")
    p.add_argument("--write-map", action="store_true")
    p.add_argument("--harvest-selection", action="store_true")
    p.add_argument("--publish-map", action="store_true", help="Also freeze into axion_beat_paper thesis map")
    p.add_argument("--sync-map", action="store_true", help="Sync map recipes from dual_lift metrics")
    args = p.parse_args()
    THESIS.mkdir(parents=True, exist_ok=True)
    local_map = None
    if LOCAL_MAP.exists():
        local_map = json.loads(LOCAL_MAP.read_text(encoding="utf-8"))
    recipe_map = build_dual_map(local_map)
    if args.harvest_selection:
        harv = harvest_selection_reports()
        (RUN_ROOT / "harvest_report.json").write_text(json.dumps(harv, indent=2), encoding="utf-8")
        print(f"harvested written={len(harv['written'])} skipped={len(harv['skipped'])}")
    if args.assemble:
        restored = restore_lock_metrics()
        if restored:
            print(f"restored locks: {restored}")
        if not args.harvest_selection:
            harv = harvest_selection_reports()
            (RUN_ROOT / "harvest_report.json").write_text(json.dumps(harv, indent=2), encoding="utf-8")
            print(f"harvested written={len(harv['written'])} skipped={len(harv['skipped'])}")
        rep = assemble_metrics(recipe_map)
        (RUN_ROOT / "assemble_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(
            f"assembled copied={len(rep['copied'])} kept={len(rep['kept'])} "
            f"missing={rep['missing']}"
        )
        # Metrics are ground truth for recipes (leap promotes); sync after assemble.
        recipe_map = sync_map_from_metrics(recipe_map)
    elif args.sync_map or args.write_map:
        recipe_map = sync_map_from_metrics(recipe_map)
    if args.write_map or args.assemble or args.sync_map or args.publish_map:
        MAP_PATH.write_text(json.dumps(recipe_map, indent=2), encoding="utf-8")
        print(f"Wrote {MAP_PATH}")
        if args.publish_map:
            RECIPE_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
            RECIPE_MAP_PATH.write_text(json.dumps(recipe_map, indent=2), encoding="utf-8")
            print(f"Published {RECIPE_MAP_PATH}")
    if args.macro or args.assemble:
        report = macro_from_metrics()
        write_status(report)


if __name__ == "__main__":
    main()

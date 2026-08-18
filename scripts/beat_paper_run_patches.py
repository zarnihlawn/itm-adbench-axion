"""Post-load patches for beat_paper_run (run_id / map ROOT fixes)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


_CLAIM_METRICS = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "axion_beat_paper_dual_lift"
    / "metrics"
)


def _claim_hparams(dataset: str, setting: str, seed: int = 111) -> Dict[str, Any]:
    """Pull k / n_components from frozen dual_lift claim metrics when present."""
    p = _CLAIM_METRICS / f"{dataset}__{setting}__{seed}__beat_paper.json"
    if not p.exists():
        return {}
    try:
        extra = json.loads(p.read_text(encoding="utf-8")).get("extra") or {}
    except Exception:
        return {}
    out: Dict[str, Any] = {}
    for key in ("n_components", "k", "n_neighbors"):
        if extra.get(key) is not None:
            out[key] = extra[key]
    return out


def _enrich_recipe_entry(
    entry: Dict[str, Any],
    *,
    dataset: str,
    setting: str,
    seed: int,
) -> Dict[str, Any]:
    """Merge claim extras + catalog RecipeSpec defaults into a map recipe dict."""
    from beat_paper_catalog import RECIPES

    out = dict(entry or {})
    recipe_id = str(out.get("recipe") or "")
    claim = _claim_hparams(dataset, setting, seed=seed)
    for key, val in claim.items():
        if out.get(key) is None:
            out[key] = val
    rec = RECIPES.get(recipe_id)
    if rec is not None:
        if out.get("n_components") is None and getattr(rec, "n_components", None) is not None:
            out["n_components"] = int(rec.n_components)
        if out.get("pca_dim") is None and getattr(rec, "pca_dim", None) is not None:
            out["pca_dim"] = int(rec.pca_dim)
    return out


def apply_run_patches(mod: Any) -> None:
    if getattr(mod, "__semi_heavy_run_patched__", False):
        return

    _orig_main = mod.main
    _orig_run_map = getattr(mod, "run_map", None)
    ROOT = mod.ROOT

    def main() -> None:
        import argparse
        import json

        from axion.config import load_config
        from beat_paper_catalog import RECIPE_MAP_PATH, local_campaign_datasets

        p = argparse.ArgumentParser(description=mod.__doc__)
        p.add_argument("--config", type=Path, default=ROOT / "configs" / "gpu_beat_paper.yaml")
        p.add_argument("--map", type=Path, default=RECIPE_MAP_PATH)
        p.add_argument("--probe12", action="store_true")
        p.add_argument("--full57", action="store_true")
        p.add_argument("--seeds", default="111")
        p.add_argument("--settings", default="semi-supervised,unsupervised")
        p.add_argument("--skip-axion", action="store_true")
        p.add_argument("--datasets", nargs="*", default=None)
        p.add_argument("--run-id", default=None)
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--local-cpu", action="store_true")
        args = p.parse_args()
        cfg = load_config(args.config)

        if args.smoke:
            args.skip_axion = True
            args.settings = "semi-supervised,unsupervised"
            if not args.datasets:
                from beat_paper_select import LOCAL_SMOKE_DATASETS

                args.datasets = list(LOCAL_SMOKE_DATASETS)
            if args.map == RECIPE_MAP_PATH:
                from beat_paper_select import SMOKE_MAP_PATH

                args.map = SMOKE_MAP_PATH
        if args.local_cpu:
            args.skip_axion = True
            args.settings = args.settings or "semi-supervised,unsupervised"
            if not args.datasets:
                args.datasets = local_campaign_datasets()
            if args.map == RECIPE_MAP_PATH:
                from beat_paper_select import LOCAL_MAP_PATH

                args.map = LOCAL_MAP_PATH

        run_id = args.run_id or cfg.get("run_id", "axion_beat_paper")
        if args.smoke and not args.run_id:
            run_id = "axion_beat_paper_smoke"
        elif args.local_cpu and not args.run_id:
            run_id = "axion_beat_paper_local"

        metrics_dir = Path(cfg["paths"]["results_dir"]) / run_id / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        seeds = mod._parse_seeds(args.seeds)
        settings = [s.strip() for s in str(args.settings).split(",") if s.strip()]
        if args.datasets:
            datasets = list(args.datasets)
        elif args.probe12:
            from beat_paper_catalog import PROBE12

            datasets = list(PROBE12)
        else:
            from axion.data.registry import registry_beat_paper_by_name

            atlas = ROOT / cfg["paths"]["atlas_csv"]
            datasets = sorted(registry_beat_paper_by_name(atlas_csv=atlas).keys())

        mod.run_map(
            cfg,
            datasets=datasets,
            seeds=seeds,
            settings=settings,
            map_path=Path(args.map),
            metrics_dir=metrics_dir,
            skip_axion=bool(args.skip_axion),
        )

    # Fix _parse_seeds / _load_map if incomplete
    if not callable(getattr(mod, "_parse_seeds", None)) or getattr(mod._parse_seeds, "__code__", None) and False:
        pass

    def _parse_seeds(s: str):
        return [int(x.strip()) for x in str(s).split(",") if x.strip()]

    def _load_map(path: Path):
        import json

        from beat_paper_catalog import default_recipe_map

        if not path.exists():
            m = default_recipe_map()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(m, indent=2), encoding="utf-8")
            print(f"WARNING: wrote default map to {path}")
            return m
        return json.loads(path.read_text(encoding="utf-8"))

    def run_map(
        cfg,
        *,
        datasets,
        seeds,
        settings,
        map_path,
        metrics_dir,
        skip_axion=False,
    ):
        """Enrich frozen map entries with claim/catalog GMM hparams, then score."""
        import copy

        raw = _load_map(Path(map_path))
        enriched = copy.deepcopy(raw)
        seed0 = int(seeds[0]) if seeds else 111
        for ds, by_setting in enriched.items():
            if not isinstance(by_setting, dict):
                continue
            for setting, entry in list(by_setting.items()):
                if not isinstance(entry, dict):
                    continue
                by_setting[setting] = _enrich_recipe_entry(
                    entry, dataset=str(ds), setting=str(setting), seed=seed0
                )
        # Write a sidecar so audits see the exact hparams used for this rescore.
        metrics_dir = Path(metrics_dir)
        thesis = metrics_dir.parent / "thesis"
        thesis.mkdir(parents=True, exist_ok=True)
        side = thesis / "recipe_map_enriched_hparams.json"
        side.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
        return _orig_run_map(
            cfg,
            datasets=datasets,
            seeds=seeds,
            settings=settings,
            map_path=side,
            metrics_dir=metrics_dir,
            skip_axion=skip_axion,
        )

    mod._parse_seeds = _parse_seeds
    mod._load_map = _load_map
    mod.main = main
    if callable(_orig_run_map):
        mod.run_map = run_map
    mod.__semi_heavy_run_patched__ = True

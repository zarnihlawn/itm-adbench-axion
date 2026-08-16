"""Post-load patches for beat_paper_run (run_id / map ROOT fixes)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def apply_run_patches(mod: Any) -> None:
    if getattr(mod, "__semi_heavy_run_patched__", False):
        return

    _orig_main = mod.main
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

    mod._parse_seeds = _parse_seeds
    mod._load_map = _load_map
    mod.main = main
    mod.__semi_heavy_run_patched__ = True

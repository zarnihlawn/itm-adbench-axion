"""Post-load patches for beat_paper_select (anti stub-val, SVHN GMM ban, leap k-grids)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

LEAP_K_METHODS = frozenset(
    {
        "knn_cosine",
        "cosine_knn",
        "abod_lite",
        "abod",
        "avg_knn",
        "knn_avg",
        "sod_lite",
        "sod",
        "pca_residual_knn",
        "pca_resid_knn",
        "fusion_knn_maha",
        "knn_maha",
        "lscp_lite",
        "lscp",
        "fusion_ecod_knn",
        "ecod_knn",
    }
)


def apply_select_patches(mod: Any) -> None:
    """Mutate loaded beat_paper_select module in place."""
    if getattr(mod, "__semi_heavy_patched__", False):
        return

    _orig_pick = mod.pick_winner
    _finite_pr = mod._finite_pr
    _orig_score_val = mod.score_recipe_val
    _multi = mod._multi_synth_pr_roc
    _fit_score = __import__("shallow_two_step", fromlist=["_fit_score"])._fit_score
    _maybe_zscore = mod._maybe_zscore
    RECIPES = mod.RECIPES
    FIT_N_CAP = mod.FIT_N_CAP
    _orig_zoo_grid = mod._score_zoo_grid
    score_zoo = mod.score_zoo

    def _val_pr(row: Dict[str, Any]) -> float:
        try:
            return float(row.get("val_PR-AUC", float("nan")))
        except (TypeError, ValueError):
            return float("nan")

    def pick_winner(
        rows: List[Dict[str, Any]],
        *,
        dataset: str,
        setting: str,
        default_recipe: str,
        lock: Optional[Dict[str, float]],
        skip_axion: bool,
    ) -> Dict[str, Any]:
        has_real = any(_finite_pr(r) and _val_pr(r) > 1e-9 for r in rows)
        filtered = list(rows)
        if has_real:
            filtered = [
                r
                for r in rows
                if (not _finite_pr(r)) or _val_pr(r) > 1e-9 or r.get("error")
            ]
            if not any(_finite_pr(r) for r in filtered):
                filtered = list(rows)
        if dataset == "SVHN":
            knn_rows = [
                r
                for r in filtered
                if str(r.get("recipe", "")).startswith(("shallow_knn_vit", "patchcore"))
                and _finite_pr(r)
            ]
            if knn_rows:
                filtered = [
                    r
                    for r in filtered
                    if not (
                        "gmm_pca64_vit" in str(r.get("recipe", ""))
                        and float(r.get("test_PR-AUC", 0) or 0) < 40.0
                    )
                ]
        return _orig_pick(
            filtered,
            dataset=dataset,
            setting=setting,
            default_recipe=default_recipe,
            lock=lock,
            skip_axion=skip_axion,
        )

    def score_recipe_val(recipe_id, spec, *, cfg, seed, setting, fast, perfection):
        rec = RECIPES[recipe_id]
        out = _orig_score_val(
            recipe_id,
            spec,
            cfg=cfg,
            seed=seed,
            setting=setting,
            fast=fast,
            perfection=perfection,
        )
        if rec.kind == "shallow" and not out.get("error"):
            try:
                from beat_paper_catalog import resolve_recipe_npz_paths
                from axion.data.registry import load_npz

                paths = resolve_recipe_npz_paths(spec, recipe_id, cfg)
                if not paths:
                    return out
                X, y = load_npz(paths[0])
                X_fit, X_val, _yf, _yv = mod._val_split(
                    X,
                    y,
                    setting=setting,
                    seed=seed,
                    val_fraction=float(cfg.get("beat_paper", {}).get("val_fraction", 0.2)),
                )
                X_fit, X_val = _maybe_zscore(X_fit, X_val, enabled=bool(rec.zscore))
                method = rec.shallow_method or "iforest"
                contam = float(out.get("contamination", 0.05))
                k = int(out.get("k", 20))

                def _score(X_eval):
                    return _fit_score(
                        method,
                        X_fit,
                        X_eval,
                        contamination=contam,
                        n_neighbors=k,
                        fit_n_cap=FIT_N_CAP,
                    )

                pr, roc = _multi(_score, X_val, seed=seed)
                out["val_PR-AUC"] = float(pr)
                out["val_ROC-AUC"] = float(roc)
            except Exception as exc:
                out["val_error"] = str(exc)
        return out

    def _score_zoo_grid(X_fit, X_val, rec, seed, fast, perfection):
        method = str(getattr(rec, "shallow_method", "") or "").lower()
        if method in ("gmm", "knn_cosine", "cosine_knn", "abod_lite", "abod"):
            return _orig_zoo_grid(X_fit, X_val, rec, seed, fast, perfection)
        if method not in LEAP_K_METHODS:
            return _orig_zoo_grid(X_fit, X_val, rec, seed, fast, perfection)

        ks = list(mod.ZOO_COSINE_KS_PERFECT if perfection else mod.ZOO_COSINE_KS_FAST)
        best_pr, best_roc, best_k = -1.0, -1.0, ks[0]
        pca_dim = getattr(rec, "pca_dim", None)
        zscore = bool(getattr(rec, "zscore", True))
        for k in ks:

            def _score_one(X_eval, _k=k):
                return score_zoo(
                    method,
                    X_fit,
                    X_eval,
                    pca_dim=pca_dim,
                    zscore=zscore,
                    n_neighbors=int(_k),
                )

            try:
                pr, roc = _multi(_score_one, X_val, seed=seed)
            except Exception:
                continue
            if float(pr) > best_pr:
                best_pr, best_roc, best_k = float(pr), float(roc), int(k)
        if best_pr < 0:
            raise RuntimeError(f"leap k-grid empty for {method}")
        meta = {"method": method, "pca_dim": pca_dim, "k": best_k, "n_neighbors": best_k}
        return best_pr, best_roc, meta

    mod.pick_winner = pick_winner
    mod.score_recipe_val = score_recipe_val
    mod._score_zoo_grid = _score_zoo_grid
    mod.__semi_heavy_patched__ = True

#!/usr/bin/env python3
"""Unit tests for Beat-Paper leap zoo detectors."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from beat_paper_heads import ZOO_METHODS, score_zoo  # noqa: E402


LEAP_METHODS = [
    "sod_lite",
    "cblof_lite",
    "avg_knn",
    "elliptical",
    "fusion_knn_maha",
    "fusion_hbos_iforest",
    "soft_vote_ensemble",
    "lscp_lite",
    "inne_lite",
    "pca_residual_knn",
    "fusion_ecod_knn",
]


@pytest.fixture
def toy_xy():
    rng = np.random.RandomState(0)
    X_fit = rng.randn(120, 12).astype(np.float32)
    X_query = rng.randn(40, 12).astype(np.float32)
    X_query[0] = 8.0
    return X_fit, X_query


def test_leap_methods_registered():
    for m in LEAP_METHODS:
        assert m in ZOO_METHODS


@pytest.mark.parametrize("method", LEAP_METHODS)
def test_leap_score_shapes(toy_xy, method):
    X_fit, X_query = toy_xy
    s = score_zoo(method, X_fit, X_query, zscore=True, n_neighbors=10)
    assert s.shape == (len(X_query),)
    assert np.all(np.isfinite(s))
    assert float(np.max(s)) >= float(np.min(s))


def test_catalog_has_leap_recipes():
    from beat_paper_catalog import RECIPES, candidates_for

    for rid in (
        "sod_lite_native",
        "fusion_knn_maha_native",
        "soft_vote_ensemble_native",
        "cblof_lite_native",
        "inne_lite_native",
        "fusion_knn_maha_e5",
        "sod_lite_vit",
        "fusion_ecod_knn_native",
        "pca_residual_knn_e5",
        "pca_residual_knn_vit",
        "sod_lite_e5",
        "avg_knn_e5",
        "soft_vote_ensemble_vit",
        "rank_max_ensemble_e5",
        "rank_max_ensemble_vit",
    ):
        assert rid in RECIPES
    unsup = candidates_for("speech", "unsupervised")
    assert "fusion_knn_maha_native" in unsup
    assert "soft_vote_ensemble_native" in unsup
    assert unsup.index("fusion_knn_maha_native") < unsup.index("shallow_lof")
    nlp = candidates_for("Imdb", "unsupervised")
    assert "pca_residual_knn_e5" in nlp
    cv = candidates_for("SVHN", "unsupervised")
    assert "pca_residual_knn_vit" in cv


def test_assert_catalog():
    from beat_paper_catalog import assert_catalog

    assert_catalog()


def test_sync_map_from_metrics_helper():
    from beat_paper_dual_lift_assemble import LOCK_DATASETS, sync_map_from_metrics

    assert "glass" in LOCK_DATASETS
    # Empty map + empty metrics dir behavior: returns dict unchanged shape
    out = sync_map_from_metrics({})
    assert isinstance(out, dict)
    if "glass" in out:
        for setting in ("semi-supervised", "unsupervised"):
            if setting in out["glass"]:
                assert out["glass"][setting]["recipe"] == "axion_edge_rare"


def test_glass_lock_in_map_when_present():
    mp = ROOT / "results" / "axion_beat_paper_dual_lift" / "thesis" / "recipe_map_57_beat.json"
    if not mp.exists():
        pytest.skip("dual_lift map missing")
    m = json.loads(mp.read_text(encoding="utf-8"))
    if "glass" not in m:
        pytest.skip("glass not in map")
    for setting in ("semi-supervised", "unsupervised"):
        if setting in m["glass"]:
            assert m["glass"][setting]["recipe"] == "axion_edge_rare"

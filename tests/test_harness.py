"""Phase-1 harness unit tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.config import load_config
from axion.data.registry import registry_by_name
from axion.eval.metrics import PAPER_DDAE, evaluate_scores
from axion.models import build_model
from axion.train.experiment import aggregate_variants, run_dataset, run_one_array


def test_evaluate_scores_perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.9, 0.8])
    m = evaluate_scores(y, s)
    assert m["PR-AUC"] == pytest.approx(100.0)
    assert m["ROC-AUC"] == pytest.approx(100.0)


def test_evaluate_scores_percent_scale():
    y = np.array([0, 0, 0, 1])
    s = np.array([0.0, 0.1, 0.2, 0.9])
    m = evaluate_scores(y, s)
    assert 0.0 <= m["PR-AUC"] <= 100.0
    assert 0.0 <= m["ROC-AUC"] <= 100.0


def test_centroid_fit_score():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 4).astype(np.float32)
    model = build_model("centroid_distance")
    model.fit(X)
    s = model.score(X)
    assert s.shape == (50,)
    assert np.all(np.isfinite(s))


def test_run_one_array_semi():
    rng = np.random.RandomState(1)
    X = rng.randn(80, 5).astype(np.float32)
    y = np.zeros(80, dtype=np.int64)
    y[-10:] = 1
    r = run_one_array(
        X, y, dataset="toy", setting="semi-supervised", seed=111, protocol="paper"
    )
    assert r.n_train == 35  # 70 normals // 2
    assert r.n_test == 45  # 35 normals + 10 anom
    assert np.isfinite(r.metrics["PR-AUC"])


def test_run_breastw_smoke():
    cfg = load_config()
    adbench = Path(cfg["paths"]["adbench_root"])
    reg = registry_by_name(adbench)
    spec = reg["breastw"]
    results = run_dataset(
        spec,
        adbench_root=adbench,
        setting="semi-supervised",
        seed=111,
        protocol="paper",
    )
    assert len(results) == 1
    agg = aggregate_variants(results)
    assert np.isfinite(agg["PR-AUC"])
    assert "semi-supervised" in PAPER_DDAE


def test_config_resolves_adbench():
    cfg = load_config()
    assert Path(cfg["paths"]["adbench_root"]).exists()
    assert "probe" in cfg
    assert len(cfg["probe"]["datasets"]) == 12

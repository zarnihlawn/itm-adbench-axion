"""NEST extension unit tests (kNN soft-tail; Imdb off)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.models import build_model
from axion.models.extensions.nest import NEST_VIEW_NAMES, NestExtension


def _toy_semi(n_n: int = 120, n_a: int = 24, d: int = 64, seed: int = 0):
    rng = np.random.RandomState(seed)
    Xn = rng.randn(n_n, d).astype(np.float32)
    Xa = rng.randn(n_a, d).astype(np.float32) + 2.0
    X = np.vstack([Xn, Xa])
    y_train = np.zeros(n_n, dtype=np.int64)
    return X, Xn, y_train


def test_nest_off_by_default():
    X, Xn, y_train = _toy_semi(seed=1)
    m = build_model(
        "axion",
        epochs=8,
        patience=3,
        batch_size=32,
        seed=2,
        device="cpu",
        nest=False,
        edge=False,
        spear=False,
        rare=False,
        true_ensemble=False,
        ensemble_heads=1,
        dataset_hint="CIFAR10",
    )
    m.fit(Xn, y_train)
    assert m._nest_ext is None
    assert m.get_params().get("nest") is False


def test_nest_skips_classical_low_d():
    X, Xn, y_train = _toy_semi(d=8, seed=3)
    m = build_model(
        "axion",
        epochs=6,
        patience=2,
        batch_size=32,
        seed=4,
        device="cpu",
        nest=True,
        nest_epochs=8,
        nest_pairs=32,
        edge=False,
        spear=False,
        rare=False,
        true_ensemble=False,
        ensemble_heads=1,
        dataset_hint="breastw",
        dataloader_num_workers=0,
    )
    m.fit(Xn, y_train)
    assert m._nest_ext is not None
    assert not m._nest_ext.fitted_
    assert m._nest_ext.resolved_.get("reason") == "not_embed_target"
    assert not m._nest_ext.active(m)


def test_nest_fits_high_d_semi_and_scores_finite():
    X, Xn, y_train = _toy_semi(d=96, seed=5)
    common = dict(
        epochs=6,
        patience=2,
        batch_size=32,
        seed=8,
        device="cpu",
        spear=False,
        rare=False,
        edge=False,
        true_ensemble=False,
        ensemble_heads=1,
        nest_epochs=15,
        nest_pairs=48,
        nest_gamma_cv=0.10,
        nest_gamma_nlp=0.08,
        nest_knn_k=10,
        dataset_hint="CIFAR10",
        dataloader_num_workers=0,
        pca_dim=0,
    )
    base = build_model("axion", nest=False, **common)
    base.fit(Xn, y_train)
    s0 = base.score(X)

    nest_m = build_model("axion", nest=True, **common)
    nest_m.fit(Xn, y_train)
    assert nest_m._nest_ext is not None
    assert nest_m._nest_ext.fitted_
    assert nest_m._nest_ext.active(nest_m)
    assert abs(float(nest_m._nest_ext.gamma) - 0.10) < 1e-9
    s1 = nest_m.score(X)
    assert s1.shape == s0.shape
    assert np.all(np.isfinite(s1))
    assert float(np.max(np.abs(s1 - s0))) > 1e-6


def test_nest_inactive_on_imdb_hint():
    X, Xn, y_train = _toy_semi(d=128, seed=6)
    m = build_model(
        "axion",
        epochs=5,
        patience=2,
        batch_size=32,
        seed=9,
        device="cpu",
        nest=True,
        nest_skip_imdb=True,
        nest_epochs=10,
        nest_pairs=32,
        edge=False,
        spear=False,
        rare=False,
        true_ensemble=False,
        ensemble_heads=1,
        dataset_hint="Imdb",
        dataloader_num_workers=0,
        pca_dim=0,
    )
    m.fit(Xn, y_train)
    assert m._nest_ext is not None
    assert not m._nest_ext.active(m)
    assert not m._nest_ext.fitted_ or abs(float(m._nest_ext.gamma)) < 1e-12


def test_nest_agnews_uses_nlp_gamma():
    X, Xn, y_train = _toy_semi(d=128, seed=7)
    m = build_model(
        "axion",
        epochs=5,
        patience=2,
        batch_size=32,
        seed=10,
        device="cpu",
        nest=True,
        nest_gamma_cv=0.15,
        nest_gamma_nlp=0.08,
        nest_epochs=12,
        nest_pairs=40,
        edge=False,
        spear=False,
        rare=False,
        true_ensemble=False,
        ensemble_heads=1,
        dataset_hint="Agnews",
        dataloader_num_workers=0,
        pca_dim=0,
    )
    m.fit(Xn, y_train)
    assert m._nest_ext is not None
    assert m._nest_ext.fitted_
    assert m._nest_ext.active(m)
    assert abs(float(m._nest_ext.gamma) - 0.08) < 1e-9
    assert len(NEST_VIEW_NAMES) == 4
    assert NestExtension.name == "nest"

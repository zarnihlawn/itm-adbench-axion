"""EDGE extension unit tests (embed density ranker; Imdb off)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.models import build_model
from axion.models.extensions.edge import EDGE_VIEW_NAMES, EdgeExtension


def _toy_semi(n_n: int = 120, n_a: int = 24, d: int = 64, seed: int = 0):
    rng = np.random.RandomState(seed)
    Xn = rng.randn(n_n, d).astype(np.float32)
    Xa = rng.randn(n_a, d).astype(np.float32) + 2.0
    X = np.vstack([Xn, Xa])
    y_train = np.zeros(n_n, dtype=np.int64)
    return X, Xn, y_train


def test_edge_off_by_default():
    X, Xn, y_train = _toy_semi(seed=1)
    m = build_model(
        "axion",
        epochs=8,
        patience=3,
        batch_size=32,
        seed=2,
        device="cpu",
        edge=False,
        spear=False,
        rare=False,
        true_ensemble=False,
        ensemble_heads=1,
        dataset_hint="CIFAR10",
    )
    m.fit(Xn, y_train)
    assert m._edge_ext is None
    assert m.get_params().get("edge") is False


def test_edge_skips_classical_low_d():
    X, Xn, y_train = _toy_semi(d=8, seed=3)
    m = build_model(
        "axion",
        epochs=6,
        patience=2,
        batch_size=32,
        seed=4,
        device="cpu",
        edge=True,
        edge_epochs=8,
        edge_pairs=32,
        spear=False,
        rare=False,
        true_ensemble=False,
        ensemble_heads=1,
        dataset_hint="breastw",
        dataloader_num_workers=0,
    )
    m.fit(Xn, y_train)
    assert m._edge_ext is not None
    assert not m._edge_ext.fitted_
    assert m._edge_ext.resolved_.get("reason") == "not_embed_target"
    assert not m._edge_ext.active(m)


def test_edge_fits_cifar_hint_and_fuses():
    X, Xn, y_train = _toy_semi(d=96, seed=5)
    common = dict(
        epochs=6,
        patience=2,
        batch_size=32,
        seed=8,
        device="cpu",
        spear=False,
        rare=False,
        true_ensemble=False,
        ensemble_heads=1,
        edge_epochs=15,
        edge_pairs=48,
        edge_gamma_cv=0.2,
        edge_gamma_nlp=0.1,
        dataset_hint="CIFAR10",
        dataloader_num_workers=0,
        pca_dim=0,
    )
    base = build_model("axion", edge=False, **common)
    base.fit(Xn, y_train)
    s0 = base.score(X)

    edge_m = build_model("axion", edge=True, **common)
    edge_m.fit(Xn, y_train)
    assert edge_m._edge_ext is not None
    assert edge_m._edge_ext.fitted_
    assert edge_m._edge_ext.active(edge_m)
    assert abs(float(edge_m._edge_ext.gamma) - 0.2) < 1e-9
    s1 = edge_m.score(X)
    assert s1.shape == s0.shape
    assert np.all(np.isfinite(s1))
    assert float(np.max(np.abs(s1 - s0))) > 1e-6


def test_edge_skips_imdb():
    X, Xn, y_train = _toy_semi(d=128, seed=6)
    m = build_model(
        "axion",
        epochs=5,
        patience=2,
        batch_size=32,
        seed=9,
        device="cpu",
        edge=True,
        edge_skip_imdb=True,
        edge_epochs=10,
        edge_pairs=32,
        spear=False,
        rare=False,
        true_ensemble=False,
        ensemble_heads=1,
        dataset_hint="Imdb",
        dataloader_num_workers=0,
    )
    m.fit(Xn, y_train)
    assert m._edge_ext is not None
    # Imdb blocked either as not_embed_target or gamma_zero
    assert not m._edge_ext.active(m)
    assert not m._edge_ext.fitted_ or abs(float(m._edge_ext.gamma)) < 1e-12


def test_edge_agnews_uses_nlp_gamma():
    X, Xn, y_train = _toy_semi(d=128, seed=7)
    m = build_model(
        "axion",
        epochs=5,
        patience=2,
        batch_size=32,
        seed=10,
        device="cpu",
        edge=True,
        edge_gamma_cv=0.15,
        edge_gamma_nlp=0.10,
        edge_epochs=12,
        edge_pairs=40,
        spear=False,
        rare=False,
        true_ensemble=False,
        ensemble_heads=1,
        dataset_hint="Agnews",
        dataloader_num_workers=0,
    )
    m.fit(Xn, y_train)
    assert m._edge_ext is not None
    assert m._edge_ext.fitted_
    assert m._edge_ext.active(m)
    assert abs(float(m._edge_ext.gamma) - 0.10) < 1e-9


def test_edge_view_count():
    assert len(EDGE_VIEW_NAMES) == 5


def test_edge_extension_protocol():
    ext = EdgeExtension(
        enabled=True,
        epochs=10,
        n_pairs=32,
        gamma_cv=0.15,
        seed=0,
        device="cpu",
    )
    X, Xn, y_train = _toy_semi(d=80, seed=11)
    model = build_model(
        "axion",
        epochs=5,
        patience=2,
        batch_size=32,
        seed=11,
        device="cpu",
        edge=False,
        spear=False,
        rare=False,
        true_ensemble=False,
        ensemble_heads=1,
        dataset_hint="FashionMNIST",
        dataloader_num_workers=0,
    )
    model.fit(Xn, y_train)
    model.is_semi_ = True
    model.dataset_hint = "FashionMNIST"
    ext.fit(model, Xn)
    assert ext.fitted_
    delta = ext.score_delta(model, X)
    assert delta.shape == (len(X),)
    assert np.all(np.isfinite(delta))

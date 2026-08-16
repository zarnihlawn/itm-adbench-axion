"""RARE extension unit tests (huge-n classical soft-AP)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.models import build_model
from axion.models.extensions.rare import RareExtension


def _toy_semi(n_n: int = 100, n_a: int = 20, d: int = 8, seed: int = 0):
    rng = np.random.RandomState(seed)
    Xn = rng.randn(n_n, d).astype(np.float32)
    Xa = rng.randn(n_a, d).astype(np.float32) + 3.0
    X = np.vstack([Xn, Xa])
    y_train = np.zeros(n_n, dtype=np.int64)
    return X, Xn, y_train


def test_rare_off_by_default():
    X, Xn, y_train = _toy_semi(seed=1)
    m = build_model(
        "axion",
        epochs=8,
        patience=3,
        batch_size=32,
        seed=2,
        device="cpu",
        rare=False,
        spear=False,
        true_ensemble=False,
        ensemble_heads=1,
    )
    m.fit(Xn, y_train)
    assert m._rare_ext is None
    assert m.get_params().get("rare") is False


def test_rare_skips_mid_n():
    X, Xn, y_train = _toy_semi(n_n=200, seed=3)
    m = build_model(
        "axion",
        epochs=6,
        patience=2,
        batch_size=64,
        seed=4,
        device="cpu",
        rare=True,
        rare_huge_n_threshold=10000,
        rare_epochs=5,
        rare_pairs=32,
        spear=False,
        true_ensemble=False,
        ensemble_heads=1,
        dataloader_num_workers=0,
    )
    m.fit(Xn, y_train)
    assert m._rare_ext is not None
    assert not m._rare_ext.fitted_
    assert m._rare_ext.resolved_.get("reason") == "not_huge_n"
    assert not m._rare_ext.active(m)


def test_rare_fits_huge_n_and_fuses():
    rng = np.random.RandomState(0)
    # Threshold lowered so unit test stays fast
    Xn = rng.randn(400, 8).astype(np.float32)
    y_train = np.zeros(len(Xn), dtype=np.int64)
    X = np.vstack([Xn, rng.randn(40, 8).astype(np.float32) + 2.5])
    common = dict(
        epochs=4,
        patience=2,
        batch_size=128,
        seed=5,
        device="cpu",
        spear=False,
        true_ensemble=False,
        ensemble_heads=1,
        rare_huge_n_threshold=300,
        rare_max_fit_n=256,
        rare_epochs=12,
        rare_pairs=48,
        rare_gamma=0.25,
        dataloader_num_workers=0,
        max_train_samples=0,
    )
    base = build_model("axion", rare=False, **common)
    base.fit(Xn, y_train)
    s0 = base.score(X)

    rare_m = build_model("axion", rare=True, **common)
    rare_m.fit(Xn, y_train)
    assert rare_m._rare_ext is not None
    assert rare_m._rare_ext.fitted_
    assert rare_m._rare_ext.active(rare_m)
    s1 = rare_m.score(X)
    assert s1.shape == s0.shape
    assert np.all(np.isfinite(s1))
    assert float(np.max(np.abs(s1 - s0))) > 1e-6


def test_rare_inactive_unsupervised():
    rng = np.random.RandomState(2)
    X = rng.randn(450, 8).astype(np.float32)
    m = build_model(
        "axion",
        epochs=4,
        patience=2,
        batch_size=128,
        seed=3,
        device="cpu",
        rare=True,
        rare_huge_n_threshold=300,
        rare_max_fit_n=256,
        rare_epochs=8,
        rare_pairs=32,
        spear=False,
        true_ensemble=False,
        ensemble_heads=1,
        dataloader_num_workers=0,
    )
    m.fit(X)
    assert not m.is_semi_
    if m._rare_ext is not None:
        assert not m._rare_ext.active(m)


def test_rare_blocks_imdb_hint():
    rng = np.random.RandomState(4)
    Xn = rng.randn(450, 8).astype(np.float32)
    y_train = np.zeros(len(Xn), dtype=np.int64)
    m = build_model(
        "axion",
        epochs=4,
        patience=2,
        batch_size=128,
        seed=6,
        device="cpu",
        rare=True,
        rare_huge_n_threshold=300,
        rare_max_fit_n=256,
        rare_epochs=8,
        rare_pairs=32,
        dataset_hint="Imdb",
        spear=False,
        true_ensemble=False,
        ensemble_heads=1,
        dataloader_num_workers=0,
    )
    m.fit(Xn, y_train)
    assert m._rare_ext is not None
    assert not m._rare_ext.fitted_
    assert m._rare_ext.resolved_.get("reason") == "modality_blocked"


def test_rare_cover_allowlist_uses_gamma_cover():
    """CORE: cover allowlisted overrides skip_cover and uses gamma_cover."""
    rng = np.random.RandomState(11)
    Xn = rng.randn(400, 8).astype(np.float32)
    y_train = np.zeros(len(Xn), dtype=np.int64)
    X = np.vstack([Xn, rng.randn(40, 8).astype(np.float32) + 2.5])
    common = dict(
        epochs=4,
        patience=2,
        batch_size=128,
        seed=12,
        device="cpu",
        spear=False,
        edge=False,
        true_ensemble=False,
        ensemble_heads=1,
        rare=True,
        rare_huge_n_threshold=300,
        rare_max_fit_n=256,
        rare_epochs=10,
        rare_pairs=32,
        rare_gamma=0.20,
        rare_gamma_cover=0.12,
        rare_skip_cover=True,
        rare_allow_datasets=["fraud", "cover"],
        dataset_hint="cover",
        dataloader_num_workers=0,
        max_train_samples=0,
    )
    m = build_model("axion", **common)
    m.fit(Xn, y_train)
    assert m._rare_ext is not None
    assert m._rare_ext.fitted_
    assert m._rare_ext.active(m)
    assert abs(float(m._rare_ext.active_gamma_) - 0.12) < 1e-9
    assert m._rare_ext.resolved_.get("is_cover") is True
    s = m.score(X)
    assert s.shape == (len(X),)
    assert np.all(np.isfinite(s))


def test_rare_fraud_keeps_base_gamma():
    rng = np.random.RandomState(13)
    Xn = rng.randn(400, 8).astype(np.float32)
    y_train = np.zeros(len(Xn), dtype=np.int64)
    common = dict(
        epochs=4,
        patience=2,
        batch_size=128,
        seed=14,
        device="cpu",
        spear=False,
        edge=False,
        true_ensemble=False,
        ensemble_heads=1,
        rare=True,
        rare_huge_n_threshold=300,
        rare_max_fit_n=256,
        rare_epochs=10,
        rare_pairs=32,
        rare_gamma=0.20,
        rare_gamma_cover=0.12,
        rare_skip_cover=True,
        rare_allow_datasets=["fraud", "cover"],
        dataset_hint="fraud",
        dataloader_num_workers=0,
        max_train_samples=0,
    )
    m = build_model("axion", **common)
    m.fit(Xn, y_train)
    assert m._rare_ext is not None
    assert m._rare_ext.fitted_
    assert abs(float(m._rare_ext.active_gamma_) - 0.20) < 1e-9
    assert m._rare_ext.resolved_.get("is_cover") is False


def test_rare_cover_still_skipped_without_allowlist():
    rng = np.random.RandomState(15)
    Xn = rng.randn(400, 8).astype(np.float32)
    y_train = np.zeros(len(Xn), dtype=np.int64)
    m = build_model(
        "axion",
        epochs=4,
        patience=2,
        batch_size=128,
        seed=16,
        device="cpu",
        rare=True,
        rare_huge_n_threshold=300,
        rare_max_fit_n=256,
        rare_epochs=8,
        rare_pairs=32,
        rare_skip_cover=True,
        rare_allow_datasets=["fraud"],
        dataset_hint="cover",
        spear=False,
        true_ensemble=False,
        ensemble_heads=1,
        dataloader_num_workers=0,
    )
    m.fit(Xn, y_train)
    assert m._rare_ext is not None
    assert not m._rare_ext.fitted_
    assert m._rare_ext.resolved_.get("reason") == "skip_cover"


def test_rare_extension_protocol():
    ext = RareExtension(
        enabled=True,
        epochs=8,
        n_pairs=32,
        huge_n_threshold=200,
        max_fit_n=180,
        seed=0,
        device="cpu",
    )
    rng = np.random.RandomState(7)
    Xn = rng.randn(260, 8).astype(np.float32)
    y_train = np.zeros(len(Xn), dtype=np.int64)
    model = build_model(
        "axion",
        epochs=5,
        patience=2,
        batch_size=64,
        seed=7,
        device="cpu",
        rare=False,
        spear=False,
        true_ensemble=False,
        ensemble_heads=1,
        dataloader_num_workers=0,
    )
    model.fit(Xn, y_train)
    model.is_semi_ = True
    model._fit_c_hedge(Xn)
    ext.fit(model, Xn)
    assert ext.fitted_
    assert ext.active(model)


"""SPEAR extension unit tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.models import build_model
from axion.models.extensions.spear import (
    SpearExtension,
    extract_view_matrix,
    synthesize_hard_negatives,
)


def _toy_semi(n_n: int = 100, n_a: int = 20, d: int = 8, seed: int = 0):
    rng = np.random.RandomState(seed)
    Xn = rng.randn(n_n, d).astype(np.float32)
    Xa = rng.randn(n_a, d).astype(np.float32) + 3.0
    X = np.vstack([Xn, Xa])
    y_train = np.zeros(n_n, dtype=np.int64)
    return X, Xn, y_train


def test_spear_off_bit_identical():
    """Default spear=False must match core scores (no extension path)."""
    X, Xn, y_train = _toy_semi(seed=7)
    common = dict(
        epochs=12,
        patience=4,
        batch_size=32,
        seed=11,
        device="cpu",
        true_ensemble=False,
        ensemble_heads=1,
        vis_alpha_semi=0.0,
        disagree_alpha_semi=0.0,
        hpd_alpha_semi=0.0,
        c_hedge_alpha_semi=0.0,
    )
    m0 = build_model("axion", spear=False, **common)
    m0.fit(Xn, y_train)
    s0 = m0.score(X)

    m1 = build_model("axion", spear=False, **common)
    m1.fit(Xn, y_train)
    s1 = m1.score(X)
    np.testing.assert_allclose(s0, s1, rtol=0, atol=0)
    assert m0._spear_ext is None
    assert m1.get_params().get("spear") is False or m1.get_params().get("spear") == False


def test_spear_views_and_synth_harder_than_normals():
    X, Xn, y_train = _toy_semi(seed=3)
    model = build_model(
        "axion",
        epochs=10,
        patience=3,
        batch_size=32,
        seed=5,
        device="cpu",
        spear=False,
        true_ensemble=False,
        ensemble_heads=1,
    )
    model.fit(Xn, y_train)
    # Ensure classical residual PCA for c_hedge view
    model._fit_c_hedge(Xn)

    feats = extract_view_matrix(model, Xn[:40])
    assert feats.shape == (40, 6)
    assert np.all(np.isfinite(feats))

    X_syn = synthesize_hard_negatives(
        model, Xn, n_synth=48, seed=99, kinds=("swap", "mix_recon", "extreme")
    )
    assert X_syn.shape[0] == 48
    assert X_syn.shape[1] == Xn.shape[1]
    mcs_n = model._mcs_scores(Xn[:48])
    mcs_a = model._mcs_scores(X_syn)
    # Synth should be harder on average (allow small slack for swap-only noise)
    assert float(mcs_a.mean()) > float(mcs_n.mean()) - 0.05


def test_spear_fits_semi_classical_and_fuses():
    X, Xn, y_train = _toy_semi(n_n=220, seed=9)
    common = dict(
        epochs=10,
        patience=3,
        batch_size=32,
        seed=13,
        device="cpu",
        true_ensemble=False,
        ensemble_heads=1,
        spear_epochs=20,
        spear_pairs=64,
        spear_gamma=0.5,
    )
    base = build_model("axion", spear=False, **common)
    base.fit(Xn, y_train)
    s_base = base.score(X)

    spear_m = build_model("axion", spear=True, spear_classical_only=True, **common)
    spear_m.fit(Xn, y_train)
    assert spear_m._spear_ext is not None
    assert spear_m._spear_ext.fitted_
    assert spear_m._spear_ext.active(spear_m)
    s_sp = spear_m.score(X)
    assert s_sp.shape == s_base.shape
    assert np.all(np.isfinite(s_sp))
    # Fusion should change scores when gamma>0 and fitted
    assert float(np.max(np.abs(s_sp - s_base))) > 1e-6


def test_spear_inactive_unsupervised():
    rng = np.random.RandomState(1)
    X = rng.randn(100, 8).astype(np.float32)
    model = build_model(
        "axion",
        epochs=8,
        patience=3,
        batch_size=32,
        seed=2,
        device="cpu",
        spear=True,
        spear_epochs=15,
        spear_pairs=48,
        true_ensemble=False,
        ensemble_heads=1,
    )
    model.fit(X)  # unsupervised (y=None)
    assert not model.is_semi_
    if model._spear_ext is not None:
        assert not model._spear_ext.active(model)


def test_spear_inactive_imdb_hint():
    X, Xn, y_train = _toy_semi(n_n=220, seed=4)
    model = build_model(
        "axion",
        epochs=8,
        patience=3,
        batch_size=32,
        seed=6,
        device="cpu",
        spear=True,
        spear_epochs=15,
        spear_pairs=48,
        dataset_hint="Imdb",
        true_ensemble=False,
        ensemble_heads=1,
    )
    model.fit(Xn, y_train)
    assert model._spear_ext is not None
    assert not model._spear_ext.fitted_
    assert model._spear_ext.resolved_.get("reason") == "modality_blocked"
    assert not model._spear_ext.active(model)


def test_spear_skips_huge_n():
    rng = np.random.RandomState(0)
    # Cap below max_train in model by passing already-sized X
    Xn = rng.randn(12001, 8).astype(np.float32)
    y_train = np.zeros(len(Xn), dtype=np.int64)
    model = build_model(
        "axion",
        epochs=3,
        patience=2,
        batch_size=256,
        seed=1,
        device="cpu",
        spear=True,
        spear_skip_huge_n=True,
        spear_huge_n_threshold=10000,
        spear_epochs=5,
        spear_pairs=32,
        true_ensemble=False,
        ensemble_heads=1,
        dataloader_num_workers=0,
        max_train_samples=0,
    )
    model.fit(Xn, y_train)
    assert model._spear_ext is not None
    assert not model._spear_ext.fitted_
    assert model._spear_ext.resolved_.get("reason") == "huge_n"
    assert not model._spear_ext.active(model)


def test_spear_skips_tiny_n():
    rng = np.random.RandomState(0)
    Xn = rng.randn(102, 7).astype(np.float32)
    y_train = np.zeros(len(Xn), dtype=np.int64)
    model = build_model(
        "axion",
        epochs=4,
        patience=2,
        batch_size=32,
        seed=1,
        device="cpu",
        spear=True,
        spear_skip_tiny_n=True,
        spear_tiny_n_threshold=200,
        spear_epochs=5,
        spear_pairs=32,
        true_ensemble=False,
        ensemble_heads=1,
        dataloader_num_workers=0,
        latch_alpha_semi_tiny=0.0,
        vis_alpha_semi=0.2,
    )
    model.fit(Xn, y_train)
    assert model._spear_ext is not None
    assert not model._spear_ext.fitted_
    assert model._spear_ext.resolved_.get("reason") == "tiny_n_skip"
    assert not model._spear_ext.active(model)
    assert abs(model.active_vis_alpha_) < 1e-12


def test_spear_views_from_parts_reuse():
    from axion.models.extensions.spear import views_from_parts

    n = 20
    parts = [np.random.randn(n) for _ in range(6)]
    feats = views_from_parts(*parts)
    assert feats.shape == (n, 6)


def test_axion_spear_alias_enables_flag():
    m = build_model("axion_spear", epochs=5, device="cpu")
    assert m.spear is True


def test_spear_fuse_modes_train_only_finite():
    """Tail / Platt / isotonic fuse stay finite and differ from linear add."""
    X, Xn, y_train = _toy_semi(n_n=220, seed=12)
    common = dict(
        epochs=10,
        patience=3,
        batch_size=32,
        seed=17,
        device="cpu",
        true_ensemble=False,
        ensemble_heads=1,
        spear=True,
        spear_classical_only=True,
        spear_epochs=18,
        spear_pairs=64,
        spear_gamma=0.5,
    )
    add_m = build_model("axion", spear_fuse="add", **common)
    add_m.fit(Xn, y_train)
    s_add = add_m.score(X)
    assert add_m._spear_ext is not None
    assert add_m._spear_ext.fuse_mode == "add"

    for mode in ("tail", "platt", "platt_tail", "isotonic"):
        m = build_model("axion", spear_fuse=mode, spear_tail_q=0.70, **common)
        m.fit(Xn, y_train)
        assert m._spear_ext is not None
        assert m._spear_ext.fitted_
        assert m._spear_ext.fuse_mode == mode
        s = m.score(X)
        assert s.shape == s_add.shape
        assert np.all(np.isfinite(s))
        assert float(np.max(np.abs(s - s_add))) > 1e-8


def test_spear_fuse_nlp_does_not_override_classical():
    """spear_fuse_nlp applies only on NLP/allowlist; classical stays add."""
    X, Xn, y_train = _toy_semi(n_n=220, d=8, seed=21)
    m = build_model(
        "axion",
        epochs=8,
        patience=3,
        batch_size=32,
        seed=21,
        device="cpu",
        spear=True,
        spear_classical_only=True,
        spear_epochs=12,
        spear_pairs=48,
        spear_fuse="add",
        spear_fuse_nlp="platt_tail",
        true_ensemble=False,
        ensemble_heads=1,
    )
    m.fit(Xn, y_train)
    assert m._spear_ext is not None
    assert m._spear_ext.fuse_mode == "add"


def test_spear_unknown_fuse_falls_back_to_add():
    ext = SpearExtension(enabled=True, fuse_mode="not_a_mode", device="cpu")
    assert ext.fuse_mode == "add"


def test_spear_extension_protocol_methods():
    ext = SpearExtension(enabled=True, epochs=5, n_pairs=32, seed=0, device="cpu")
    X, Xn, y_train = _toy_semi(n_n=220, seed=8)
    model = build_model(
        "axion",
        epochs=8,
        patience=3,
        batch_size=32,
        seed=8,
        device="cpu",
        spear=False,
        true_ensemble=False,
        ensemble_heads=1,
    )
    model.fit(Xn, y_train)
    model.is_semi_ = True
    model._fit_c_hedge(Xn)
    ext.fit(model, Xn, X_val=Xn[:20])
    assert ext.fitted_
    delta = ext.score_delta(model, X)
    assert delta.shape == (len(X),)
    assert np.all(np.isfinite(delta))

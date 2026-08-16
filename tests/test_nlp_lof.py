"""Unit tests for NLP LOF fuse view (fair BERT; not cv_knn)."""
from __future__ import annotations

import numpy as np

from axion.models.extensions.nlp_lof import NlpLofView
from axion.models import build_model


def test_nlp_lof_lof_and_knn_whiten():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 48).astype(np.float32)
    v = NlpLofView(enabled=True, alpha=0.35, method="lof", n_neighbors=8, memory_cap=80)
    v.fit(X)
    z = v.z_normed(X[:16])
    assert v.active()
    assert z.shape == (16,)
    assert np.isfinite(z).all()

    v2 = NlpLofView(
        enabled=True, alpha=0.3, method="knn", whiten=True, whiten_dim=12, memory_cap=80
    )
    v2.fit(X)
    z2 = v2.z_normed(X[:16])
    assert v2.active()
    assert z2.shape == (16,)
    assert np.isfinite(z2).all()


def test_nlp_lof_transform_add_is_identity():
    rng = np.random.RandomState(1)
    X = rng.randn(80, 24).astype(np.float32)
    v = NlpLofView(enabled=True, alpha=0.4, fuse_mode="add", n_neighbors=6)
    v.fit(X)
    z = v.z_normed(X[:10])
    mcs = rng.randn(10).astype(np.float64)
    out = v.transform(z, mcs)
    np.testing.assert_allclose(out, z, rtol=0, atol=0)


def test_nlp_lof_new_legal_heads():
    rng = np.random.RandomState(2)
    X = rng.randn(90, 32).astype(np.float32)
    for method in ("sphere", "ecod", "copod", "ocsvm", "dsvdd", "ensemble"):
        v = NlpLofView(
            enabled=True,
            alpha=1.0,
            method=method,
            whiten=True,
            whiten_dim=12,
            n_neighbors=6,
            memory_cap=80,
        )
        v.fit(X)
        z = v.z_normed(X[:12])
        assert v.active(), method
        assert z.shape == (12,), method
        assert np.isfinite(z).all(), method


def test_build_model_cv_lof_keys():
    m = build_model(
        "axion",
        cv_lof=True,
        cv_lof_method="lof",
        cv_lof_mode="replace",
        dataset_hint="CIFAR10",
        epochs=1,
        score_batch_size=512,
    )
    assert m.cv_lof is True
    assert m.cv_lof_allow_datasets == ("cifar10",)
    assert m.cv_knn is False


def test_build_model_nlp_lof_keys():
    m = build_model(
        "axion",
        nlp_lof=True,
        nlp_lof_alpha_semi=0.4,
        nlp_lof_method="lof",
        dataset_hint="imdb",
        epochs=1,
        score_batch_size=512,
    )
    assert m.nlp_lof is True
    assert m.nlp_lof_allow_datasets == ("imdb",)
    assert abs(m.nlp_lof_alpha_semi - 0.4) < 1e-9

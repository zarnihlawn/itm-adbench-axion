"""CvKnnView: leave-one-ish train z-norm must not use self-NN (dist≈0)."""
from __future__ import annotations

import numpy as np

from axion.models.extensions.cv_knn import CvKnnView


def test_cv_knn_train_znorm_drops_self_match():
    rng = np.random.RandomState(111)
    X = rng.randn(64, 16).astype(np.float32)
    view = CvKnnView(enabled=True, alpha=0.8, n_neighbors=1, memory_cap=8000, seed=111)
    view.fit(X)
    assert view._nn is not None
    assert view.resolved_.get("enabled") is True
    assert int(view.resolved_.get("k_fit", 0)) >= 2
    # Self-NN would collapse train raw scores near 0 and std to ~1e-8.
    assert float(view.std_ or 0.0) > 1e-4
    z_tr = view.z_normed(X)
    assert z_tr.shape == (64,)
    assert np.isfinite(z_tr).all()
    assert float(np.std(z_tr)) > 0.1
    # Held-out queries: no self-match; scores should vary.
    X_te = rng.randn(32, 16).astype(np.float32) + 0.5
    z_te = view.z_normed(X_te)
    assert z_te.shape == (32,)
    assert np.isfinite(z_te).all()
    assert float(np.std(z_te)) > 0.1


def test_cv_knn_k3_still_fits():
    rng = np.random.RandomState(222)
    X = rng.randn(40, 8).astype(np.float32)
    view = CvKnnView(enabled=True, alpha=0.5, n_neighbors=3, seed=222)
    view.fit(X)
    assert view.active()
    s = view.raw_scores(rng.randn(10, 8).astype(np.float32))
    assert s.shape == (10,)
    assert np.isfinite(s).all()

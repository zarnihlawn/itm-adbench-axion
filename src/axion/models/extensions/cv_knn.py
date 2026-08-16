"""CV kNN / PatchCore-lite memory view for fair ResNet embeds (semi).

Train-anchored kNN distance to normals memory; z-normed then fused like EDGE.
Disclosed method-addendum path toward stretch >=56.5 (Fashion/CIFAR unpaid).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class CvKnnView:
    """PatchCore-lite kNN distance view (CV embeds only)."""

    name = "cv_knn"

    def __init__(
        self,
        enabled: bool = False,
        alpha: float = 0.35,
        memory_cap: int = 8000,
        n_neighbors: int = 5,
        seed: int = 111,
    ):
        self.enabled = bool(enabled)
        self.alpha = float(alpha)
        self.memory_cap = int(memory_cap)
        self.n_neighbors = int(n_neighbors)
        self.seed = int(seed)
        self._nn = None
        self._k_query = int(n_neighbors)
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None
        self.resolved_: Dict[str, Any] = {"enabled": False}

    def active(self) -> bool:
        return bool(self.enabled) and self._nn is not None and abs(self.alpha) > 1e-12

    def get_params(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "alpha": self.alpha,
            "memory_cap": self.memory_cap,
            "n_neighbors": self.n_neighbors,
            "fitted": self._nn is not None,
            "resolved": dict(self.resolved_),
        }

    def fit(self, X_train: np.ndarray) -> None:
        from sklearn.neighbors import NearestNeighbors

        X = np.asarray(X_train, dtype=np.float32)
        n = int(X.shape[0])
        if n < 8:
            self.resolved_ = {"enabled": False, "reason": "tiny_n", "n": n}
            self._nn = None
            return
        rng = np.random.RandomState(self.seed + 77)
        if n > int(self.memory_cap):
            idx = rng.choice(n, size=int(self.memory_cap), replace=False)
            mem = X[idx]
        else:
            mem = X
        k = min(int(self.n_neighbors), max(1, len(mem)))
        # Fit with k+1 so train calibration can drop the trivial self-match (dist≈0).
        k_fit = min(max(k + 1, k), len(mem))
        nn = NearestNeighbors(n_neighbors=k_fit, algorithm="auto")
        nn.fit(mem)
        self._nn = nn
        self._k_query = int(k)
        # Train z-norm anchors: leave-one-ish (skip self NN when present).
        d_tr, _ = nn.kneighbors(X)
        raw = self._mean_knn_dists(d_tr, drop_self=True)
        self.mean_ = float(raw.mean())
        self.std_ = float(max(float(raw.std()), 1e-8))
        self.resolved_ = {
            "enabled": True,
            "n": n,
            "mem": int(len(mem)),
            "k": int(k),
            "k_fit": int(k_fit),
            "mean": self.mean_,
            "std": self.std_,
        }

    def _mean_knn_dists(self, d: np.ndarray, *, drop_self: bool) -> np.ndarray:
        """Mean of the k query distances; optionally skip col0 self-match."""
        d = np.asarray(d, dtype=np.float64)
        k = int(getattr(self, "_k_query", self.n_neighbors) or self.n_neighbors)
        k = max(1, min(k, d.shape[1]))
        if drop_self and d.shape[1] > 1 and float(np.median(d[:, 0])) < 1e-6:
            take = d[:, 1 : 1 + k]
            if take.shape[1] == 0:
                take = d[:, :k]
        else:
            take = d[:, :k]
        return take.mean(axis=1).astype(np.float64)

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        if self._nn is None:
            return np.zeros(len(X), dtype=np.float64)
        X = np.asarray(X, dtype=np.float32)
        d, _ = self._nn.kneighbors(X)
        # Auto-drop self-match when queries overlap the memory (dist≈0).
        return self._mean_knn_dists(d, drop_self=True)

    def z_normed(self, X: np.ndarray) -> np.ndarray:
        raw = self.raw_scores(X)
        mu = float(self.mean_ if self.mean_ is not None else raw.mean())
        sd = float(self.std_ if self.std_ is not None else max(float(raw.std()), 1e-8))
        return (raw - mu) / sd

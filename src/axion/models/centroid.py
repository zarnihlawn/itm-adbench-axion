"""Harness smoke scorer: Mahalanobis / Euclidean distance to train centroid.

Not a ship method — Phase 1 wiring only. Replaced by AXION in Phase 2.
Distinct from paper PCA/MCD baselines (no eigenspace / robust cov ship claim).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class CentroidDistanceModel:
    """Anomaly score = squared distance to train mean (optional diagonal whitening)."""

    name = "centroid_distance"

    def __init__(self, whiten: bool = True, eps: float = 1e-6):
        self.whiten = whiten
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.inv_var_: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "CentroidDistanceModel":
        X = np.asarray(X_train, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        if self.whiten:
            var = X.var(axis=0)
            self.inv_var_ = 1.0 / np.maximum(var, self.eps)
        else:
            self.inv_var_ = np.ones(X.shape[1], dtype=np.float64)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.inv_var_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float64)
        diff = X - self.mean_
        return np.sum((diff ** 2) * self.inv_var_, axis=1).astype(np.float64)

    def get_params(self) -> Dict[str, Any]:
        return {"name": self.name, "whiten": self.whiten, "eps": self.eps}

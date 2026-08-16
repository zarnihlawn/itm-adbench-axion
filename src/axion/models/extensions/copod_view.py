"""COPOD-inspired empirical-copula outlier view (extreme imbalance classical).

Market ancestor: Li et al., COPOD. Skew-aware left/right tail aggregation.
Higher score = more anomalous.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class CopodView:
    name = "copod"

    def __init__(self, enabled: bool = True, alpha: float = 0.18, max_fit_n: int = 20000):
        self.enabled = bool(enabled)
        self.alpha = float(alpha)
        self.max_fit_n = int(max(256, max_fit_n))
        self.fitted_: bool = False
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None
        self.sorted_: Optional[list] = None
        self.skew_: Optional[np.ndarray] = None
        self.resolved_: Dict[str, Any] = {}

    def get_params(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "alpha": self.alpha,
            "fitted": self.fitted_,
            "resolved": dict(self.resolved_),
        }

    def active(self) -> bool:
        return bool(self.enabled and self.fitted_ and abs(self.alpha) > 1e-12)

    def fit(self, X_train: np.ndarray) -> None:
        self.fitted_ = False
        self.sorted_ = None
        self.skew_ = None
        self.resolved_ = {"skipped": True}
        if not self.enabled:
            return
        X = np.asarray(X_train, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 16:
            self.resolved_["reason"] = "tiny_n"
            return
        rng = np.random.RandomState(1)
        if X.shape[0] > self.max_fit_n:
            idx = rng.choice(X.shape[0], size=self.max_fit_n, replace=False)
            X = X[idx]
        self.sorted_ = [np.sort(X[:, j]) for j in range(X.shape[1])]
        # sample skewness per dim (Fisher-Pearson)
        mu = X.mean(axis=0)
        sd = np.maximum(X.std(axis=0), 1e-8)
        z = (X - mu) / sd
        self.skew_ = np.mean(z**3, axis=0)
        raw = self.score_raw(X)
        self.mean_ = float(raw.mean())
        self.std_ = float(max(float(raw.std()), 1e-8))
        self.fitted_ = True
        self.resolved_ = {
            "skipped": False,
            "n_fit": int(X.shape[0]),
            "d": int(X.shape[1]),
            "alpha": float(self.alpha),
        }

    def score_raw(self, X: np.ndarray) -> np.ndarray:
        assert self.sorted_ is not None and self.skew_ is not None
        X = np.asarray(X, dtype=np.float64)
        n_dims = len(self.sorted_)
        scores = np.zeros(len(X), dtype=np.float64)
        for j in range(n_dims):
            col = self.sorted_[j]
            n = float(len(col))
            left = np.searchsorted(col, X[:, j], side="right") / n
            right = 1.0 - np.searchsorted(col, X[:, j], side="left") / n
            left = np.clip(left, 1.0 / (n + 1.0), 1.0)
            right = np.clip(right, 1.0 / (n + 1.0), 1.0)
            # skew-aware: positive skew → emphasize left tail, else right
            if float(self.skew_[j]) >= 0:
                scores += -np.log(left)
            else:
                scores += -np.log(right)
            # also add opposite mild contribution (COPOD-style bipartite)
            scores += 0.25 * (-np.log(np.minimum(left, right)))
        return scores

    def z_normed(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_ or self.sorted_ is None:
            return np.zeros(len(X), dtype=np.float64)
        raw = self.score_raw(X)
        return (raw - float(self.mean_ or 0.0)) / (float(self.std_ or 1.0) + 1e-8)


__all__ = ["CopodView"]

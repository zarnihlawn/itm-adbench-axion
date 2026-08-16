"""Z-score standardization fit on train only (AnoDDAE paper prep)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Standardizer:
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None
    eps: float = 1e-8

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std < self.eps, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer not fitted")
        X = np.asarray(X, dtype=np.float64)
        return ((X - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Standardizer]:
    """Fit z-score on train; transform train and test."""
    scaler = Standardizer().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), scaler

"""AnoDDAE / Livernoche paper-faithful train/test splits.

Must match ``AnoDDAE/AnoDDAE/src/data.py::split_data`` bitwise when using the
same ``random_state`` (global ``np.random.seed`` + ``np.random.shuffle``).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_setting: str = "unsupervised",
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Paper-faithful split (AnoDDAE official).

    - unsupervised: train and test are the full ``(X, y)``
    - semi-supervised: train = 50% of normals only;
      test = remaining normals + all anomalies
    """
    if random_state is not None:
        np.random.seed(random_state)

    if train_setting == "unsupervised":
        return X, X, y, y

    if train_setting == "semi-supervised":
        anomaly_indices = np.where(y == 1)[0]
        normal_indices = np.where(y == 0)[0]

        np.random.shuffle(normal_indices)
        half_normals = len(normal_indices) // 2

        train_normal_indices = normal_indices[:half_normals]
        test_normal_indices = normal_indices[half_normals:]
        test_indices = np.concatenate([test_normal_indices, anomaly_indices])

        return (
            X[train_normal_indices],
            X[test_indices],
            y[train_normal_indices],
            y[test_indices],
        )

    raise ValueError("train_setting must be 'unsupervised' or 'semi-supervised'")


def carve_val_from_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_fraction: float = 0.2,
    random_state: Optional[int] = None,
    min_val: int = 8,
    min_fit: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrity twin: hold out validation from train only (never from test).

    Returns ``(X_fit, X_val, y_fit, y_val)``. Empty val if too small to carve.
    """
    if val_fraction <= 0.0:
        return X_train, X_train[:0], y_train, y_train[:0]

    n = int(X_train.shape[0])
    if n < (min_val + min_fit):
        return X_train, X_train[:0], y_train, y_train[:0]

    rng = np.random.RandomState(0 if random_state is None else int(random_state))
    n_val = int(round(n * float(val_fraction)))
    n_val = max(min_val, min(n_val, n - min_fit))
    idx = rng.permutation(n)
    val_idx = idx[:n_val]
    fit_idx = idx[n_val:]
    return X_train[fit_idx], X_train[val_idx], y_train[fit_idx], y_train[val_idx]

"""Model protocol for AXION harness."""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AnomalyModel(Protocol):
    """Fit on train normals (or full X); score with higher = more anomalous."""

    name: str

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "AnomalyModel":
        ...

    def score(self, X: np.ndarray) -> np.ndarray:
        ...

    def get_params(self) -> Dict[str, Any]:
        ...

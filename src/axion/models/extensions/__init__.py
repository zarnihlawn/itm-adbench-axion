"""AXION score extensions (semi-only ranking heads, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from axion.models.axion_model import AxionModel

from axion.models.extensions.copod_view import CopodView
from axion.models.extensions.ecod_view import EcodView
from axion.models.extensions.edge import EdgeExtension
from axion.models.extensions.goad_lite import GoadLiteExtension
from axion.models.extensions.nest import NestExtension
from axion.models.extensions.rare import RareExtension
from axion.models.extensions.spear import SpearExtension, views_from_parts

__all__ = [
    "ScoreExtension",
    "SpearExtension",
    "RareExtension",
    "EdgeExtension",
    "NestExtension",
    "EcodView",
    "CopodView",
    "GoadLiteExtension",
    "views_from_parts",
]


@runtime_checkable
class ScoreExtension(Protocol):
    """Pluggable score delta after frozen Axion core fit."""

    def fit(
        self,
        model: "AxionModel",
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
    ) -> None:
        ...

    def score_delta(self, model: "AxionModel", X: np.ndarray) -> np.ndarray:
        """Raw extension score (higher = more anomalous). Caller applies z-norm + gamma."""
        ...

    def active(self, model: "AxionModel") -> bool:
        ...

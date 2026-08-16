"""GOAD-lite: cheap transform-prediction disagreement (mid classical semi).

Market ancestor: GOAD (Bergman & Hoshen). Train a tiny MLP to predict which
of K random affine transforms was applied to normals; anomaly score = CE loss.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class _GoadHead(nn.Module):
    def __init__(self, d: int, n_trans: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_trans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GoadLiteExtension:
    name = "goad"

    def __init__(
        self,
        enabled: bool = True,
        alpha: float = 0.10,
        n_transforms: int = 4,
        epochs: int = 20,
        lr: float = 1e-2,
        max_fit_n: int = 4000,
        seed: int = 111,
        device: Optional[str] = None,
    ):
        self.enabled = bool(enabled)
        self.alpha = float(alpha)
        self.n_transforms = int(max(2, n_transforms))
        self.epochs = int(max(1, epochs))
        self.lr = float(lr)
        self.max_fit_n = int(max(128, max_fit_n))
        self.seed = int(seed)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.head: Optional[_GoadHead] = None
        self.scales_: Optional[np.ndarray] = None
        self.shifts_: Optional[np.ndarray] = None
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None
        self.fitted_: bool = False
        self.resolved_: Dict[str, Any] = {}

    def get_params(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "alpha": self.alpha,
            "fitted": self.fitted_,
            "n_transforms": self.n_transforms,
            "resolved": dict(self.resolved_),
        }

    def active(self) -> bool:
        return bool(self.enabled and self.fitted_ and self.head is not None and abs(self.alpha) > 1e-12)

    def _make_transforms(self, d: int, rng: np.random.RandomState) -> None:
        # diagonal scale + shift per transform (cheap affine)
        self.scales_ = rng.uniform(0.5, 1.5, size=(self.n_transforms, d)).astype(np.float32)
        self.shifts_ = rng.normal(0.0, 0.25, size=(self.n_transforms, d)).astype(np.float32)

    def _apply(self, X: np.ndarray, t: int) -> np.ndarray:
        assert self.scales_ is not None and self.shifts_ is not None
        return X * self.scales_[t] + self.shifts_[t]

    def fit(self, X_train: np.ndarray) -> None:
        self.fitted_ = False
        self.head = None
        self.resolved_ = {"skipped": True}
        if not self.enabled:
            return
        X = np.asarray(X_train, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] < 24:
            self.resolved_["reason"] = "tiny_n"
            return
        rng = np.random.RandomState(self.seed + 77)
        if X.shape[0] > self.max_fit_n:
            idx = rng.choice(X.shape[0], size=self.max_fit_n, replace=False)
            X = X[idx]
        d = int(X.shape[1])
        self._make_transforms(d, rng)
        xs = []
        ys = []
        for t in range(self.n_transforms):
            xt = self._apply(X, t)
            xs.append(xt)
            ys.append(np.full(len(X), t, dtype=np.int64))
        Xb = np.concatenate(xs, axis=0)
        yb = np.concatenate(ys, axis=0)
        # shuffle
        perm = rng.permutation(len(Xb))
        Xb, yb = Xb[perm], yb[perm]

        device = self.device
        head = _GoadHead(d, self.n_transforms).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        xt = torch.from_numpy(Xb).to(device)
        yt = torch.from_numpy(yb).to(device)
        bs = min(256, len(Xb))
        head.train()
        for _ in range(self.epochs):
            for i in range(0, len(Xb), bs):
                xb = xt[i : i + bs]
                yb_ = yt[i : i + bs]
                opt.zero_grad(set_to_none=True)
                logits = head(xb)
                loss = loss_fn(logits, yb_)
                loss.backward()
                opt.step()
        head.eval()
        self.head = head
        with torch.no_grad():
            raw = self.score_raw(X)
        self.mean_ = float(raw.mean())
        self.std_ = float(max(float(raw.std()), 1e-8))
        self.fitted_ = True
        self.resolved_ = {
            "skipped": False,
            "n_fit": int(len(X)),
            "d": d,
            "n_transforms": self.n_transforms,
            "alpha": float(self.alpha),
        }

    def score_raw(self, X: np.ndarray) -> np.ndarray:
        assert self.head is not None and self.scales_ is not None
        X = np.asarray(X, dtype=np.float32)
        device = self.device
        self.head.eval()
        losses = []
        with torch.no_grad():
            for t in range(self.n_transforms):
                xt = torch.from_numpy(self._apply(X, t)).to(device)
                logits = self.head(xt)
                logp = torch.log_softmax(logits, dim=1)
                # CE vs true transform id t
                nll = -logp[:, t]
                losses.append(nll.cpu().numpy())
        return np.mean(np.stack(losses, axis=0), axis=0).astype(np.float64)

    def z_normed(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_ or self.head is None:
            return np.zeros(len(X), dtype=np.float64)
        raw = self.score_raw(X)
        return (raw - float(self.mean_ or 0.0)) / (float(self.std_ or 1.0) + 1e-8)


__all__ = ["GoadLiteExtension"]

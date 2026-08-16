"""NEST: Neighbor Embed Soft Tail (CV / NLP high-d semi).

Train-anchored kNN distance + density ratio fused with MCS/recon via a tiny ranker.
Imdb always off. Classical low-d off (leave SPEAR / RARE). Distinct from EDGE
(Mahalanobis/cos density) so it can lift when EDGE is flat at embed ceiling.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from axion.models.extensions.spear import (
    _SpearRanker,
    pairwise_logistic_loss,
    soft_ap_loss,
    synthesize_hard_negatives,
)
from axion.util.progress import ProgressBar

if TYPE_CHECKING:
    from axion.models.axion_model import AxionModel

NEST_VIEW_NAMES: Tuple[str, ...] = (
    "mcs",
    "knn_mean",
    "density_ratio",
    "recon",
)


def _nest_gamma_for_model(model: "AxionModel", gamma_cv: float, gamma_nlp: float) -> float:
    hint = (model.dataset_hint or "").lower()
    if "imdb" in hint:
        return 0.0
    raw_d = int(getattr(model, "raw_d_", 0) or 0)
    if model._is_nlp_modality(raw_d) or any(
        k in hint for k in ("agnews", "amazon", "yelp", "20news")
    ):
        return float(gamma_nlp)
    return float(gamma_cv)


class NestExtension:
    """kNN-neighbor soft-tail ranker for high-d / PCA / CV+NLP semi."""

    name = "nest"

    def __init__(
        self,
        enabled: bool = True,
        gamma_cv: float = 0.10,
        gamma_nlp: float = 0.08,
        n_pairs: int = 256,
        epochs: int = 30,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        hidden: int = 0,
        knn_k: int = 20,
        min_train_n: int = 24,
        max_fit_n: int = 6000,
        skip_imdb: bool = True,
        soft_ap_weight: float = 0.25,
        synth_kinds: Sequence[str] = ("extreme", "mix_recon", "swap"),
        seed: int = 111,
        device: Optional[str] = None,
    ):
        self.enabled = bool(enabled)
        self.gamma_cv = float(gamma_cv)
        self.gamma_nlp = float(gamma_nlp)
        self.n_pairs = int(max(8, n_pairs))
        self.epochs = int(max(1, epochs))
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.hidden = int(hidden)
        self.knn_k = int(max(3, knn_k))
        self.min_train_n = int(min_train_n)
        self.max_fit_n = int(max(256, max_fit_n))
        self.skip_imdb = bool(skip_imdb)
        self.soft_ap_weight = float(soft_ap_weight)
        self.synth_kinds = tuple(synth_kinds)
        self.seed = int(seed)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.ranker: Optional[_SpearRanker] = None
        self.view_mean_: Optional[np.ndarray] = None
        self.view_std_: Optional[np.ndarray] = None
        self.nest_mean_: Optional[float] = None
        self.nest_std_: Optional[float] = None
        self.nn_: Optional[NearestNeighbors] = None
        self.train_med_dist_: float = 1.0
        self.fitted_: bool = False
        self.resolved_: Dict[str, Any] = {}
        self._active_gamma: float = 0.0

    @property
    def gamma(self) -> float:
        return float(self._active_gamma)

    def get_params(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "gamma_cv": self.gamma_cv,
            "gamma_nlp": self.gamma_nlp,
            "gamma": self._active_gamma,
            "knn_k": self.knn_k,
            "n_pairs": self.n_pairs,
            "epochs": self.epochs,
            "skip_imdb": self.skip_imdb,
            "fitted": self.fitted_,
            "resolved": dict(self.resolved_),
        }

    def _is_embed_target(self, model: "AxionModel") -> bool:
        hint = (model.dataset_hint or "").lower()
        if self.skip_imdb and "imdb" in hint:
            return False
        if bool(getattr(model, "used_pca_", False)):
            return True
        raw_d = int(getattr(model, "raw_d_", 0) or 0)
        if raw_d >= 400:
            return True
        if any(
            k in hint
            for k in ("cifar", "fashion", "mnist", "svhn", "agnews", "amazon", "yelp")
        ):
            return True
        if model._is_nlp_modality(raw_d):
            return "imdb" not in hint
        return False

    def active(self, model: "AxionModel") -> bool:
        if not self.enabled or not self.fitted_ or self.ranker is None:
            return False
        if not bool(getattr(model, "is_semi_", False)):
            return False
        g = _nest_gamma_for_model(model, self.gamma_cv, self.gamma_nlp)
        if abs(g) < 1e-12:
            return False
        return self._is_embed_target(model)

    def _fit_knn(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        k = int(min(self.knn_k, max(1, len(X) - 1)))
        self.nn_ = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        self.nn_.fit(X)
        # median self-distance among train (exclude self neighbor if returned)
        dists, _ = self.nn_.kneighbors(X, return_distance=True)
        # first neighbor is usually self at ~0; use mean of neighbors 1:
        if dists.shape[1] > 1:
            row_mean = dists[:, 1:].mean(axis=1)
        else:
            row_mean = dists[:, 0]
        self.train_med_dist_ = float(max(np.median(row_mean), 1e-6))

    def _knn_feats(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self.nn_ is not None
        X = np.asarray(X, dtype=np.float32)
        dists, _ = self.nn_.kneighbors(X, return_distance=True)
        knn_mean = dists.mean(axis=1).astype(np.float64)
        density_ratio = (knn_mean / (self.train_med_dist_ + 1e-8)).astype(np.float64)
        return knn_mean, density_ratio

    def extract_features(
        self, model: "AxionModel", X: np.ndarray, mcs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if mcs is None:
            mcs = model._mcs_scores(X)
        knn_mean, dens = self._knn_feats(X)
        recon = model._recon_error(X)
        return np.stack(
            [
                np.asarray(mcs, dtype=np.float64),
                knn_mean,
                dens,
                np.asarray(recon, dtype=np.float64),
            ],
            axis=1,
        )

    def _normalize_views(self, feats: np.ndarray) -> np.ndarray:
        assert self.view_mean_ is not None and self.view_std_ is not None
        return (feats - self.view_mean_) / (self.view_std_ + 1e-8)

    def fit(
        self,
        model: "AxionModel",
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
    ) -> None:
        self.fitted_ = False
        self.ranker = None
        self.nest_mean_ = None
        self.nest_std_ = None
        self.nn_ = None
        self._active_gamma = 0.0
        self.resolved_ = {"skipped": True}
        if not self.enabled:
            return
        if not bool(getattr(model, "is_semi_", False)):
            self.resolved_["reason"] = "not_semi"
            return
        if not self._is_embed_target(model):
            self.resolved_["reason"] = "not_embed_target"
            return
        X_train = np.asarray(X_train, dtype=np.float32)
        n = int(X_train.shape[0])
        if n < int(self.min_train_n):
            self.resolved_["reason"] = "tiny_n"
            self.resolved_["n"] = n
            return
        if model.net is None:
            self.resolved_["reason"] = "no_net"
            return

        self._active_gamma = _nest_gamma_for_model(model, self.gamma_cv, self.gamma_nlp)
        if abs(self._active_gamma) < 1e-12:
            self.resolved_["reason"] = "gamma_zero"
            return

        rng = np.random.RandomState(self.seed + 411)
        X_fit = X_train
        if n > int(self.max_fit_n):
            idx = rng.choice(n, size=int(self.max_fit_n), replace=False)
            X_fit = X_train[idx]

        self._fit_knn(X_fit)
        n_synth = int(min(self.n_pairs, max(32, len(X_fit) // 3)))
        X_syn = synthesize_hard_negatives(
            model,
            X_fit,
            n_synth=n_synth,
            seed=self.seed + 421,
            kinds=self.synth_kinds,
        )
        if len(X_syn) < 8:
            self.resolved_["reason"] = "synth_empty"
            return

        feats_n = self.extract_features(model, X_fit)
        feats_a = self.extract_features(model, X_syn)
        self.view_mean_ = feats_n.mean(axis=0)
        self.view_std_ = np.maximum(feats_n.std(axis=0), 1e-6)
        Zn = self._normalize_views(feats_n)
        Za = self._normalize_views(feats_a)

        n_a = len(Za)
        n_hold = max(4, n_a // 5)
        hold_idx = rng.choice(n_a, size=n_hold, replace=False)
        train_a_mask = np.ones(n_a, dtype=bool)
        train_a_mask[hold_idx] = False
        Za_tr, Za_ho = Za[train_a_mask], Za[hold_idx]
        n_n = len(Zn)
        n_hold_n = max(4, min(n_hold, n_n // 5))
        Zn_ho = Zn[-n_hold_n:]
        Zn_tr = Zn[:-n_hold_n] if n_n > n_hold_n else Zn

        device = self.device
        if model.device.type == "cpu":
            device = torch.device("cpu")
        self.device = device

        hidden = 0 if self.hidden <= 0 else int(self.hidden)
        ranker = _SpearRanker(n_views=len(NEST_VIEW_NAMES), hidden=hidden).to(device)
        opt = torch.optim.Adam(ranker.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        tn = torch.from_numpy(Zn_tr.astype(np.float32)).to(device)
        ta = torch.from_numpy(Za_tr.astype(np.float32)).to(device)
        hn = torch.from_numpy(Zn_ho.astype(np.float32)).to(device)
        ha = torch.from_numpy(Za_ho.astype(np.float32)).to(device)

        best_state = None
        best_sep = -1e9
        stale = 0
        patience = max(5, self.epochs // 4)
        ep = 0
        ep_bar = ProgressBar(
            self.epochs, desc="ext:nest", leave=True, unit="ep", position=2
        )
        try:
            for ep in range(self.epochs):
                ranker.train()
                bs_n = min(128, tn.shape[0])
                bs_a = min(128, ta.shape[0])
                ni = torch.randint(0, tn.shape[0], (bs_n,), device=device)
                ai = torch.randint(0, ta.shape[0], (bs_a,), device=device)
                sn = ranker(tn[ni])
                sa = ranker(ta[ai])
                loss = pairwise_logistic_loss(sa, sn)
                if self.soft_ap_weight > 1e-12:
                    scores = torch.cat([sa, sn], dim=0)
                    labels = torch.cat(
                        [
                            torch.ones(sa.shape[0], device=device),
                            torch.zeros(sn.shape[0], device=device),
                        ],
                        dim=0,
                    )
                    loss = loss + float(self.soft_ap_weight) * soft_ap_loss(scores, labels)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                ranker.eval()
                with torch.no_grad():
                    sep = float((ranker(ha).mean() - ranker(hn).mean()).item())
                ep_bar.set_description(f"ext:nest sep={sep:.4f}")
                ep_bar.update(1)
                if sep > best_sep + 1e-4:
                    best_sep = sep
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in ranker.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break
        finally:
            ep_bar.close()

        if best_state is not None:
            ranker.load_state_dict(best_state)
        ranker.eval()
        self.ranker = ranker
        with torch.no_grad():
            all_n = torch.from_numpy(Zn.astype(np.float32)).to(device)
            raw = ranker(all_n).cpu().numpy().astype(np.float64)
        self.nest_mean_ = float(raw.mean())
        self.nest_std_ = float(max(float(raw.std()), 1e-8))
        self.fitted_ = True
        self.resolved_ = {
            "skipped": False,
            "n_train": n,
            "n_fit": int(len(X_fit)),
            "n_synth": int(len(X_syn)),
            "epochs_ran": int(ep + 1),
            "best_sep": float(best_sep),
            "gamma": float(self._active_gamma),
            "knn_k": int(min(self.knn_k, max(1, len(X_fit) - 1))),
        }

    def score_delta(
        self, model: "AxionModel", X: np.ndarray, mcs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if self.ranker is None or self.view_mean_ is None or self.nn_ is None:
            return np.zeros(len(X), dtype=np.float64)
        feats = self.extract_features(model, X, mcs=mcs)
        return self.score_delta_from_feats(feats)

    def score_delta_from_feats(self, feats: np.ndarray) -> np.ndarray:
        if self.ranker is None or self.view_mean_ is None:
            return np.zeros(len(feats), dtype=np.float64)
        Z = self._normalize_views(np.asarray(feats, dtype=np.float64))
        self.ranker.eval()
        with torch.no_grad():
            t = torch.from_numpy(Z.astype(np.float32)).to(self.device)
            out = np.zeros(len(feats), dtype=np.float64)
            bs = 4096
            for i in range(0, len(feats), bs):
                out[i : i + bs] = self.ranker(t[i : i + bs]).cpu().numpy()
        return out

    def z_normed_delta(
        self, model: "AxionModel", X: np.ndarray, mcs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        raw = self.score_delta(model, X, mcs=mcs)
        mu = float(self.nest_mean_ or 0.0)
        sd = float(self.nest_std_ or 1.0)
        return (raw - mu) / (sd + 1e-8)


__all__ = ["NestExtension", "NEST_VIEW_NAMES"]

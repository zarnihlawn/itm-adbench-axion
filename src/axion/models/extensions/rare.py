"""RARE: Rare-event AP Ranking Extension (huge-n classical semi only).

Complement of SPEAR: activates when n > huge_n_threshold (cover/fraud).
Protocol-legal: normals + synthetic hard negatives only.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from axion.models.extensions.spear import (
    VIEW_NAMES,
    _SpearRanker,
    extract_view_matrix,
    pairwise_logistic_loss,
    soft_ap_loss,
    synthesize_hard_negatives,
    views_from_parts,
)
from axion.util.progress import ProgressBar

if TYPE_CHECKING:
    from axion.models.axion_model import AxionModel


class RareExtension:
    """Huge-n rare-tail ranking on MCS-primary multi-view scores."""

    name = "rare"

    def __init__(
        self,
        enabled: bool = True,
        gamma: float = 0.20,
        n_pairs: int = 256,
        epochs: int = 40,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        hidden: int = 0,
        min_train_n: int = 64,
        huge_n_threshold: int = 10000,
        max_fit_n: int = 8000,
        soft_ap_weight: float = 0.50,
        pairwise_weight: float = 0.35,
        synth_kinds: Sequence[str] = ("extreme", "mix_recon"),
        skip_cover: bool = True,
        # If non-empty, RARE only activates for these dataset hints (e.g. fraud).
        # If "cover" is listed, skip_cover is overridden for cover.
        allow_datasets: Sequence[str] = (),
        gamma_cover: float = 0.12,
        seed: int = 111,
        device: Optional[str] = None,
    ):
        self.enabled = bool(enabled)
        self.gamma = float(gamma)
        self.gamma_cover = float(gamma_cover)
        self.n_pairs = int(max(8, n_pairs))
        self.epochs = int(max(1, epochs))
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.hidden = int(hidden)
        self.min_train_n = int(min_train_n)
        self.huge_n_threshold = int(huge_n_threshold)
        self.max_fit_n = int(max(512, max_fit_n))
        self.soft_ap_weight = float(soft_ap_weight)
        self.pairwise_weight = float(pairwise_weight)
        self.synth_kinds = tuple(synth_kinds)
        self.skip_cover = bool(skip_cover)
        self.allow_datasets = tuple(
            str(x).strip().lower() for x in allow_datasets if str(x).strip()
        )
        self.seed = int(seed)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.ranker: Optional[_SpearRanker] = None
        self.view_mean_: Optional[np.ndarray] = None
        self.view_std_: Optional[np.ndarray] = None
        self.rare_mean_: Optional[float] = None
        self.rare_std_: Optional[float] = None
        self.fitted_: bool = False
        self.resolved_: Dict[str, Any] = {}
        self.fit_n_: int = 0
        self.active_gamma_: float = float(gamma)

    def _hint_is_cover(self, hint: str) -> bool:
        return "cover" in str(hint or "").lower()

    def _cover_allowlisted(self) -> bool:
        return "cover" in self.allow_datasets

    def effective_gamma(self, hint: Optional[str] = None) -> float:
        h = str(hint or "").lower()
        if self._hint_is_cover(h):
            return float(self.gamma_cover)
        return float(self.gamma)

    def get_params(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "gamma": self.gamma,
            "gamma_cover": self.gamma_cover,
            "active_gamma": float(self.active_gamma_),
            "n_pairs": self.n_pairs,
            "epochs": self.epochs,
            "lr": self.lr,
            "hidden": self.hidden,
            "huge_n_threshold": self.huge_n_threshold,
            "max_fit_n": self.max_fit_n,
            "soft_ap_weight": self.soft_ap_weight,
            "skip_cover": self.skip_cover,
            "allow_datasets": list(self.allow_datasets),
            "fitted": self.fitted_,
            "resolved": dict(self.resolved_),
        }

    def _modality_allowed(self, model: "AxionModel", n_train: int) -> bool:
        hint = (model.dataset_hint or "").lower()
        if "imdb" in hint:
            return False
        if n_train <= int(self.huge_n_threshold):
            return False
        if bool(getattr(model, "used_pca_", False)):
            return False
        raw_d = int(getattr(model, "raw_d_", 0) or 0)
        if raw_d >= 400:
            return False
        if any(
            k in hint
            for k in (
                "cifar",
                "fashion",
                "mnist",
                "svhn",
                "mvtec",
                "agnews",
                "amazon",
                "yelp",
            )
        ):
            return False
        if model._is_nlp_modality(raw_d):
            return False
        return True

    def active(self, model: "AxionModel") -> bool:
        if not self.enabled or not self.fitted_ or self.ranker is None:
            return False
        if not bool(getattr(model, "is_semi_", False)):
            return False
        if abs(float(self.active_gamma_)) < 1e-12:
            return False
        n = int(self.fit_n_ or 0)
        if n <= int(self.huge_n_threshold):
            return False
        return self._modality_allowed(model, n_train=n)

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
        self.rare_mean_ = None
        self.rare_std_ = None
        self.resolved_ = {"skipped": True}
        if not self.enabled:
            return
        if not bool(getattr(model, "is_semi_", False)):
            self.resolved_["reason"] = "not_semi"
            return
        X_train = np.asarray(X_train, dtype=np.float32)
        n = int(X_train.shape[0])
        self.fit_n_ = n
        if n <= int(self.huge_n_threshold):
            self.resolved_["reason"] = "not_huge_n"
            self.resolved_["n"] = n
            return
        hint = str(getattr(model, "dataset_hint", "") or "").lower()
        # skip_cover unless cover is explicitly allowlisted (CORE path)
        if (
            bool(self.skip_cover)
            and self._hint_is_cover(hint)
            and not self._cover_allowlisted()
        ):
            self.resolved_["reason"] = "skip_cover"
            self.resolved_["n"] = n
            return
        if self.allow_datasets:
            if not any(a in hint for a in self.allow_datasets):
                self.resolved_["reason"] = "not_allowlisted"
                self.resolved_["n"] = n
                self.resolved_["hint"] = hint
                return
        if not self._modality_allowed(model, n_train=n):
            self.resolved_["reason"] = "modality_blocked"
            self.resolved_["n"] = n
            return
        if n < int(self.min_train_n):
            self.resolved_["reason"] = "tiny_n"
            return
        if model.net is None:
            self.resolved_["reason"] = "no_net"
            return

        self.active_gamma_ = float(self.effective_gamma(hint))

        rng = np.random.RandomState(self.seed + 191)
        X_fit = X_train
        if n > int(self.max_fit_n):
            idx = rng.choice(n, size=int(self.max_fit_n), replace=False)
            X_fit = X_train[idx]

        n_synth = int(min(self.n_pairs, max(64, len(X_fit) // 4)))
        X_syn = synthesize_hard_negatives(
            model,
            X_fit,
            n_synth=n_synth,
            seed=self.seed + 201,
            kinds=self.synth_kinds,
        )
        if len(X_syn) < 8:
            self.resolved_["reason"] = "synth_empty"
            return

        feats_n = extract_view_matrix(model, X_fit)
        feats_a = extract_view_matrix(model, X_syn)
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
        ranker = _SpearRanker(n_views=len(VIEW_NAMES), hidden=hidden).to(device)
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
            self.epochs, desc="ext:rare", leave=True, unit="ep", position=2
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
                loss = float(self.pairwise_weight) * pairwise_logistic_loss(sa, sn)
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
                ep_bar.set_description(f"ext:rare sep={sep:.4f}")
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
        self.rare_mean_ = float(raw.mean())
        self.rare_std_ = float(max(float(raw.std()), 1e-8))
        self.fitted_ = True
        self.resolved_ = {
            "skipped": False,
            "n_train": n,
            "n_fit": int(len(X_fit)),
            "n_synth": int(len(X_syn)),
            "epochs_ran": int(ep + 1),
            "best_sep": float(best_sep),
            "gamma": float(self.active_gamma_),
            "gamma_base": float(self.gamma),
            "gamma_cover": float(self.gamma_cover),
            "hint": hint,
            "is_cover": bool(self._hint_is_cover(hint)),
        }

    def score_delta_from_views(self, feats: np.ndarray) -> np.ndarray:
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

    def z_normed_delta_from_views(self, feats: np.ndarray) -> np.ndarray:
        raw = self.score_delta_from_views(feats)
        mu = float(self.rare_mean_ or 0.0)
        sd = float(self.rare_std_ or 1.0)
        return (raw - mu) / (sd + 1e-8)


__all__ = ["RareExtension", "views_from_parts"]

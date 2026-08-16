"""SPEAR: Semi-supervised Precision Extension for Anomaly Ranking.

Protocol-legal: trains only on normals (+ synthetic hard negatives from normals).
Never uses test labels. Activates only in semi classical / low-d by default.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from axion.util.progress import ProgressBar

if TYPE_CHECKING:
    from axion.models.axion_model import AxionModel

VIEW_NAMES: Tuple[str, ...] = (
    "mcs",
    "latch",
    "vis",
    "disagree",
    "hpd",
    "c_hedge",
)


class _SpearRanker(nn.Module):
    """Tiny ranker on multi-view score features (linear or 1-hidden MLP)."""

    def __init__(self, n_views: int, hidden: int = 0):
        super().__init__()
        h = int(hidden)
        if h <= 0:
            self.net = nn.Linear(n_views, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(n_views, h),
                nn.ReLU(inplace=True),
                nn.Linear(h, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def views_from_parts(
    mcs: np.ndarray,
    latch: np.ndarray,
    vis: np.ndarray,
    disagree: np.ndarray,
    hpd: np.ndarray,
    c_hedge: np.ndarray,
) -> np.ndarray:
    """Stack precomputed score views into (n, v) SPEAR features."""
    feats = np.stack(
        [
            np.asarray(mcs, dtype=np.float64),
            np.asarray(latch, dtype=np.float64),
            np.asarray(vis, dtype=np.float64),
            np.asarray(disagree, dtype=np.float64),
            np.asarray(hpd, dtype=np.float64),
            np.asarray(c_hedge, dtype=np.float64),
        ],
        axis=1,
    )
    return feats


def extract_view_matrix(model: "AxionModel", X: np.ndarray) -> np.ndarray:
    """Build (n, v) multi-view score features in model space."""
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    mcs, disagree, hpd = model._mcs_scores(X, return_extra=True)
    latch = model._latch_score(X)
    vis = model._recon_error(X)
    ch = model._c_hedge_score(X)
    feats = views_from_parts(mcs, latch, vis, disagree, hpd, ch)
    assert feats.shape == (n, len(VIEW_NAMES))
    return feats


@torch.no_grad()
def _visible_mu(model: "AxionModel", X: np.ndarray) -> np.ndarray:
    """Fully-visible reconstruction mean μ (mask=0)."""
    assert model.net is not None
    model.net.eval()
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    xb = torch.from_numpy(X).to(model.device)
    out = np.zeros_like(X, dtype=np.float32)
    bs = int(getattr(model, "_score_bs", 512) or 512)
    bs = min(max(512, bs), max(1, n))
    for i in range(0, n, bs):
        chunk = xb[i : i + bs]
        mask = torch.zeros_like(chunk)
        mu, _lv, _z = model.net(chunk, mask)
        out[i : i + chunk.shape[0]] = mu.cpu().numpy()
    return out


def synthesize_hard_negatives(
    model: "AxionModel",
    X: np.ndarray,
    *,
    n_synth: int,
    seed: int,
    kinds: Sequence[str] = ("swap", "mix_recon", "extreme"),
) -> np.ndarray:
    """Build hard pseudo-anomalies from normals only (protocol-legal)."""
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    if n < 2 or n_synth <= 0:
        return np.zeros((0, d), dtype=np.float32)
    rng = np.random.RandomState(int(seed))
    kinds_l = list(kinds) if kinds else ["swap"]
    if "mix_recon" in kinds_l or "extreme" in kinds_l:
        if n > 4000:
            idx = rng.choice(n, size=4000, replace=False)
            X_mu = X[idx]
            mu_sub = _visible_mu(model, X_mu)
            mu_map = {int(i): mu_sub[j] for j, i in enumerate(idx)}
        else:
            mu_full = _visible_mu(model, X)
            mu_map = {i: mu_full[i] for i in range(n)}
    else:
        mu_map = {}

    out: List[np.ndarray] = []
    for t in range(int(n_synth)):
        kind = kinds_l[t % len(kinds_l)]
        i = int(rng.randint(0, n))
        row = X[i].copy()
        if kind == "swap":
            j = int(rng.randint(0, n))
            while j == i and n > 1:
                j = int(rng.randint(0, n))
            width = max(1, d // 4)
            start = int(rng.randint(0, max(1, d - width + 1)))
            row[start : start + width] = X[j, start : start + width]
        elif kind == "mix_recon":
            mui = mu_map.get(i)
            if mui is None:
                mui = _visible_mu(model, X[i : i + 1])[0]
            a = float(rng.uniform(0.35, 0.85))
            row = (1.0 - a) * row + a * np.asarray(mui, dtype=np.float32)
        else:
            mui = mu_map.get(i)
            if mui is None:
                mui = _visible_mu(model, X[i : i + 1])[0]
            resid = row - np.asarray(mui, dtype=np.float32)
            scale = float(rng.uniform(1.5, 3.5))
            noise = rng.randn(d).astype(np.float32) * float(np.std(row) + 1e-3)
            row = row + scale * resid + 0.5 * noise
        out.append(row.astype(np.float32))
    return np.stack(out, axis=0)


def pairwise_logistic_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """Encourage pos (anomaly) scores > neg (normal) scores."""
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    return torch.nn.functional.softplus(-diff).mean()


def soft_ap_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Differentiable soft-AP surrogate (higher score = positive preferred)."""
    s = scores
    y = labels.float()
    pos_mask = y > 0.5
    if not bool(pos_mask.any()):
        return scores.new_zeros(())
    diff = s.unsqueeze(0) - s.unsqueeze(1)
    soft_rank = torch.sigmoid(diff).sum(dim=1) + 1.0
    pos_scores = s[pos_mask]
    pos_diff = pos_scores.unsqueeze(0) - pos_scores.unsqueeze(1)
    soft_pos_above = torch.sigmoid(pos_diff).sum(dim=1)
    soft_prec = soft_pos_above / soft_rank[pos_mask].clamp_min(1.0)
    return 1.0 - soft_prec.mean()


class SpearExtension:
    """Semi-only PR-tail ranking extension on frozen Axion multi-view scores."""

    name = "spear"

    def __init__(
        self,
        enabled: bool = True,
        gamma: float = 0.35,
        n_pairs: int = 256,
        epochs: int = 40,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        hidden: int = 0,
        min_train_n: int = 24,
        classical_only: bool = True,
        skip_huge_n: bool = True,
        huge_n_threshold: int = 10000,
        skip_tiny_n: bool = True,
        tiny_n_threshold: int = 200,
        use_soft_ap: bool = False,
        soft_ap_weight: float = 0.25,
        # If non-empty, these hints bypass classical_only (and imdb skip when listed).
        allow_datasets: Optional[Sequence[str]] = None,
        rank_fuse: bool = False,
        # Fuse vs MCS: add (default) | tail | platt | platt_tail | isotonic.
        # All calibrators fit on train normals + synthetic hard negs only.
        fuse_mode: str = "add",
        tail_q: float = 0.75,
        tail_soft: float = 0.35,
        synth_kinds: Sequence[str] = ("swap", "mix_recon", "extreme"),
        seed: int = 111,
        device: Optional[str] = None,
    ):
        self.enabled = bool(enabled)
        self.gamma = float(gamma)
        self.n_pairs = int(max(8, n_pairs))
        self.epochs = int(max(1, epochs))
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.hidden = int(hidden)
        self.min_train_n = int(min_train_n)
        self.classical_only = bool(classical_only)
        self.skip_huge_n = bool(skip_huge_n)
        self.huge_n_threshold = int(huge_n_threshold)
        self.skip_tiny_n = bool(skip_tiny_n)
        self.tiny_n_threshold = int(tiny_n_threshold)
        self.use_soft_ap = bool(use_soft_ap)
        self.soft_ap_weight = float(soft_ap_weight)
        self.allow_datasets = [
            str(x).strip().lower()
            for x in (allow_datasets or ())
            if str(x).strip()
        ]
        self.rank_fuse = bool(rank_fuse)
        mode = str(fuse_mode or "add").strip().lower()
        if mode not in ("add", "tail", "platt", "platt_tail", "isotonic"):
            mode = "add"
        self.fuse_mode = mode
        self.tail_q = float(np.clip(tail_q, 0.0, 0.99))
        self.tail_soft = float(max(1e-3, tail_soft))
        self._train_raw_sorted: Optional[np.ndarray] = None
        self._mcs_mu: float = 0.0
        self._mcs_sd: float = 1.0
        self._tail_tau: float = 0.0
        self._platt_full: Any = None
        self._platt_mcs: Any = None
        self._platt_mu: float = 0.0
        self._platt_sd: float = 1.0
        self._iso: Any = None
        self._iso_mu: float = 0.0
        self._iso_sd: float = 1.0
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
        self.spear_mean_: Optional[float] = None
        self.spear_std_: Optional[float] = None
        self.fitted_: bool = False
        self.resolved_: Dict[str, Any] = {}

    def get_params(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "gamma": self.gamma,
            "n_pairs": self.n_pairs,
            "epochs": self.epochs,
            "lr": self.lr,
            "hidden": self.hidden,
            "classical_only": self.classical_only,
            "skip_huge_n": self.skip_huge_n,
            "huge_n_threshold": self.huge_n_threshold,
            "skip_tiny_n": self.skip_tiny_n,
            "tiny_n_threshold": self.tiny_n_threshold,
            "use_soft_ap": self.use_soft_ap,
            "allow_datasets": list(self.allow_datasets),
            "rank_fuse": self.rank_fuse,
            "fuse_mode": self.fuse_mode,
            "tail_q": self.tail_q,
            "tail_soft": self.tail_soft,
            "fitted": self.fitted_,
            "resolved": dict(self.resolved_),
        }

    def _modality_allowed(self, model: "AxionModel", n_train: Optional[int] = None) -> bool:
        hint = (model.dataset_hint or "").lower()
        n = int(n_train) if n_train is not None else -1
        if bool(self.skip_huge_n) and n > int(self.huge_n_threshold):
            return False
        # Explicit allowlist: bypass classical_only / default imdb skip when listed.
        if self.allow_datasets and any(a in hint for a in self.allow_datasets):
            return True
        if "imdb" in hint:
            return False
        if not self.classical_only:
            return True
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
        if abs(float(self.gamma)) < 1e-12:
            return False
        return self._modality_allowed(model)

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
        self.spear_mean_ = None
        self.spear_std_ = None
        self._train_raw_sorted = None
        self._platt_full = None
        self._platt_mcs = None
        self._iso = None
        self.resolved_ = {"skipped": True}
        if not self.enabled:
            return
        if not bool(getattr(model, "is_semi_", False)):
            self.resolved_["reason"] = "not_semi"
            return
        X_train = np.asarray(X_train, dtype=np.float32)
        n = int(X_train.shape[0])
        if bool(self.skip_huge_n) and n > int(self.huge_n_threshold):
            self.resolved_["reason"] = "huge_n"
            self.resolved_["n"] = n
            return
        if bool(self.skip_tiny_n) and n < int(self.tiny_n_threshold):
            self.resolved_["reason"] = "tiny_n_skip"
            self.resolved_["n"] = n
            return
        if not self._modality_allowed(model, n_train=n):
            self.resolved_["reason"] = "modality_blocked"
            return
        if n < int(self.min_train_n):
            self.resolved_["reason"] = "tiny_n"
            self.resolved_["n"] = n
            return
        if model.net is None:
            self.resolved_["reason"] = "no_net"
            return

        epochs = self.epochs
        hidden = self.hidden
        wd = self.weight_decay
        if n < 80:
            epochs = max(15, epochs // 2)
            hidden = 0
            wd = max(wd, 1e-3)

        rng = np.random.RandomState(self.seed + 91)
        n_synth = int(min(self.n_pairs, max(32, n)))
        X_syn = synthesize_hard_negatives(
            model,
            X_train,
            n_synth=n_synth,
            seed=self.seed + 101,
            kinds=self.synth_kinds,
        )
        if len(X_syn) < 8:
            self.resolved_["reason"] = "synth_empty"
            return

        X_fit = X_train
        if n > 6000:
            idx = rng.choice(n, size=6000, replace=False)
            X_fit = X_train[idx]

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

        ranker = _SpearRanker(n_views=len(VIEW_NAMES), hidden=hidden).to(device)
        opt = torch.optim.Adam(ranker.parameters(), lr=self.lr, weight_decay=wd)

        tn = torch.from_numpy(Zn_tr.astype(np.float32)).to(device)
        ta = torch.from_numpy(Za_tr.astype(np.float32)).to(device)
        hn = torch.from_numpy(Zn_ho.astype(np.float32)).to(device)
        ha = torch.from_numpy(Za_ho.astype(np.float32)).to(device)

        best_state = None
        best_sep = -1e9
        stale = 0
        patience = max(5, epochs // 4)
        ep = 0

        ep_bar = ProgressBar(epochs, desc="ext:spear", leave=True, unit="ep", position=2)
        try:
            for ep in range(epochs):
                ranker.train()
                bs_n = min(128, tn.shape[0])
                bs_a = min(128, ta.shape[0])
                ni = torch.randint(0, tn.shape[0], (bs_n,), device=device)
                ai = torch.randint(0, ta.shape[0], (bs_a,), device=device)
                sn = ranker(tn[ni])
                sa = ranker(ta[ai])
                loss = pairwise_logistic_loss(sa, sn)
                if self.use_soft_ap:
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
                ep_bar.set_description(f"ext:spear sep={sep:.4f}")
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
        self.spear_mean_ = float(raw.mean())
        self.spear_std_ = float(max(float(raw.std()), 1e-8))
        self._train_raw_sorted = np.sort(raw.astype(np.float64))
        mcs_n = np.asarray(feats_n[:, 0], dtype=np.float64)
        self._mcs_mu = float(mcs_n.mean())
        self._mcs_sd = float(max(float(mcs_n.std()), 1e-8))
        mcs_zn = (mcs_n - self._mcs_mu) / (self._mcs_sd + 1e-8)
        self._tail_tau = float(np.quantile(mcs_zn, self.tail_q))
        spear_zn = (raw - float(self.spear_mean_)) / (float(self.spear_std_) + 1e-8)
        if self.fuse_mode in ("platt", "platt_tail"):
            with torch.no_grad():
                ta = torch.from_numpy(Za.astype(np.float32)).to(device)
                raw_a = ranker(ta).cpu().numpy().astype(np.float64)
            mcs_a = np.asarray(feats_a[:, 0], dtype=np.float64)
            mcs_za = (mcs_a - self._mcs_mu) / (self._mcs_sd + 1e-8)
            spear_za = (raw_a - float(self.spear_mean_)) / (float(self.spear_std_) + 1e-8)
            self._fit_platt(mcs_zn, spear_zn, mcs_za, spear_za)
        if self.fuse_mode == "isotonic":
            with torch.no_grad():
                ta = torch.from_numpy(Za.astype(np.float32)).to(device)
                raw_a = ranker(ta).cpu().numpy().astype(np.float64)
            self._fit_isotonic(raw, raw_a)
        self.fitted_ = True
        self.resolved_ = {
            "skipped": False,
            "n_train": n,
            "n_synth": int(len(X_syn)),
            "epochs_ran": int(ep + 1),
            "best_sep": float(best_sep),
            "hidden": int(hidden),
            "gamma": float(self.gamma),
            "rank_fuse": bool(self.rank_fuse),
            "fuse_mode": str(self.fuse_mode),
            "tail_q": float(self.tail_q),
            "tail_tau": float(self._tail_tau),
        }

        if X_val is not None and len(X_val) >= 8:
            Xv = np.asarray(X_val, dtype=np.float32)
            if len(Xv) > 2000:
                Xv = Xv[rng.choice(len(Xv), size=2000, replace=False)]
            dv = self.score_delta(model, Xv)
            self.resolved_["val_delta_mean"] = float(np.mean(dv))
            self.resolved_["val_delta_std"] = float(np.std(dv))

    def score_delta(self, model: "AxionModel", X: np.ndarray) -> np.ndarray:
        if self.ranker is None or self.view_mean_ is None:
            return np.zeros(len(X), dtype=np.float64)
        X = np.asarray(X, dtype=np.float32)
        feats = extract_view_matrix(model, X)
        return self.score_delta_from_views(feats)

    def score_delta_from_views(self, feats: np.ndarray) -> np.ndarray:
        """Ranker forward from precomputed (n, v) views (avoids second MCS)."""
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

    def _to_delta(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64)
        if self.rank_fuse and self._train_raw_sorted is not None and len(self._train_raw_sorted) > 0:
            n = float(len(self._train_raw_sorted))
            ranks = np.searchsorted(self._train_raw_sorted, raw, side="right") / n
            return (ranks - 0.5) / 0.2886751345948129
        mu = float(self.spear_mean_ or 0.0)
        sd = float(self.spear_std_ or 1.0)
        return (raw - mu) / (sd + 1e-8)

    def _mcs_z_from_views(self, feats: np.ndarray) -> np.ndarray:
        mcs = np.asarray(feats[:, 0], dtype=np.float64)
        return (mcs - float(self._mcs_mu)) / (float(self._mcs_sd) + 1e-8)

    def _tail_gate(self, mcs_z: np.ndarray) -> np.ndarray:
        z = np.asarray(mcs_z, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-(z - float(self._tail_tau)) / float(self.tail_soft)))

    def _fit_platt(
        self,
        mcs_zn: np.ndarray,
        spear_zn: np.ndarray,
        mcs_za: np.ndarray,
        spear_za: np.ndarray,
    ) -> None:
        """Train-only Platt residual: extra log-odds from SPEAR beyond MCS."""
        from sklearn.linear_model import LogisticRegression

        yn = np.zeros(len(mcs_zn), dtype=np.int64)
        ya = np.ones(len(mcs_za), dtype=np.int64)
        y = np.concatenate([yn, ya])
        X_full = np.column_stack(
            [
                np.concatenate([mcs_zn, mcs_za]),
                np.concatenate([spear_zn, spear_za]),
            ]
        ).astype(np.float64)
        X_mcs = X_full[:, :1]
        try:
            full = LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                max_iter=250,
                C=1.0,
            )
            mcs_only = LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                max_iter=250,
                C=1.0,
            )
            full.fit(X_full, y)
            mcs_only.fit(X_mcs, y)
        except Exception:
            self._platt_full = None
            self._platt_mcs = None
            return
        self._platt_full = full
        self._platt_mcs = mcs_only
        resid_n = self._platt_residual(mcs_zn, spear_zn)
        self._platt_mu = float(resid_n.mean())
        self._platt_sd = float(max(float(resid_n.std()), 1e-8))

    def _platt_residual(self, mcs_z: np.ndarray, spear_z: np.ndarray) -> np.ndarray:
        if self._platt_full is None or self._platt_mcs is None:
            return np.asarray(spear_z, dtype=np.float64)
        X_full = np.column_stack([mcs_z, spear_z]).astype(np.float64)
        logit_full = np.asarray(self._platt_full.decision_function(X_full), dtype=np.float64)
        logit_mcs = np.asarray(self._platt_mcs.decision_function(X_full[:, :1]), dtype=np.float64)
        return logit_full - logit_mcs

    def _fit_isotonic(self, raw_n: np.ndarray, raw_a: np.ndarray) -> None:
        from sklearn.isotonic import IsotonicRegression

        x = np.concatenate([raw_n, raw_a]).astype(np.float64)
        y = np.concatenate(
            [np.zeros(len(raw_n), dtype=np.float64), np.ones(len(raw_a), dtype=np.float64)]
        )
        if float(np.std(x)) < 1e-10:
            self._iso = None
            return
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        try:
            iso.fit(x, y)
        except Exception:
            self._iso = None
            return
        self._iso = iso
        p_n = np.asarray(iso.predict(raw_n), dtype=np.float64)
        self._iso_mu = float(p_n.mean())
        self._iso_sd = float(max(float(p_n.std()), 1e-8))

    def _isotonic_delta(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64)
        if self._iso is None:
            return self._to_delta(raw)
        p = np.asarray(self._iso.predict(raw), dtype=np.float64)
        return (p - float(self._iso_mu)) / (float(self._iso_sd) + 1e-8)

    def _apply_fuse(self, feats: np.ndarray, raw: np.ndarray) -> np.ndarray:
        spear_z = self._to_delta(raw)
        mode = str(self.fuse_mode)
        if mode == "add":
            return spear_z
        mcs_z = self._mcs_z_from_views(feats)
        if mode == "isotonic":
            delta = self._isotonic_delta(raw)
        elif mode in ("platt", "platt_tail"):
            resid = self._platt_residual(mcs_z, spear_z)
            delta = (resid - float(self._platt_mu)) / (float(self._platt_sd) + 1e-8)
        else:
            delta = spear_z
        if mode in ("tail", "platt_tail"):
            delta = delta * self._tail_gate(mcs_z)
        return delta

    def z_normed_delta(self, model: "AxionModel", X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        feats = extract_view_matrix(model, X)
        return self._apply_fuse(feats, self.score_delta_from_views(feats))

    def z_normed_delta_from_views(self, feats: np.ndarray) -> np.ndarray:
        feats = np.asarray(feats, dtype=np.float64)
        return self._apply_fuse(feats, self.score_delta_from_views(feats))

"""NLP LOF / kNN density fuse on fair BERT embeds (semi).

Protocol-legal: train-only fit on normals (ADBench semi); optional train-only
PCA whitening (P08). Not cv_knn (CV PatchCore path). Not alt/e5 embeds.
Cite paper_ids: P14 (LOF) / P03 (embed+LOF) / P08 (whiten).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class NlpLofView:
    """Train-anchored LOF or kNN distance view for NLP embeds."""

    name = "nlp_lof"

    def __init__(
        self,
        enabled: bool = False,
        alpha: float = 0.35,
        method: str = "lof",
        # fuse: add alpha*z to AXION; blend: (1-a)*axion + a*lof_z; replace: lof_z only
        mode: str = "fuse",
        n_neighbors: int = 20,
        memory_cap: int = 6000,
        whiten: bool = False,
        whiten_dim: int = 64,
        allow_datasets: Optional[Sequence[str]] = None,
        fuse_mode: str = "add",
        tail_q: float = 0.75,
        tail_soft: float = 0.35,
        seed: int = 111,
    ):
        self.enabled = bool(enabled)
        self.alpha = float(alpha)
        self.method = str(method or "lof").strip().lower()
        mode_s = str(mode or "fuse").strip().lower()
        self.mode = mode_s if mode_s in ("fuse", "blend", "replace") else "fuse"
        self.n_neighbors = int(max(3, n_neighbors))
        self.memory_cap = int(max(64, memory_cap))
        self.whiten = bool(whiten)
        self.whiten_dim = int(max(8, whiten_dim))
        self.allow_datasets: List[str] = [
            str(x).strip().lower()
            for x in (allow_datasets or ("imdb",))
            if str(x).strip()
        ]
        self.seed = int(seed)
        fm = str(fuse_mode or "add").strip().lower()
        if fm not in ("add", "tail", "platt", "platt_tail"):
            fm = "add"
        self.fuse_mode = fm
        self.tail_q = float(np.clip(tail_q, 0.0, 0.99))
        self.tail_soft = float(max(1e-3, tail_soft))
        self._mcs_mu = 0.0
        self._mcs_sd = 1.0
        self._tail_tau = 0.0
        self._platt_full = None
        self._platt_mcs = None
        self._platt_mu = 0.0
        self._platt_sd = 1.0
        self._nn = None
        self._lof = None
        self._k_query = int(self.n_neighbors)
        self._whiten_mean: Optional[np.ndarray] = None
        self._whiten_components: Optional[np.ndarray] = None
        self._whiten_scales: Optional[np.ndarray] = None
        self._center: Optional[np.ndarray] = None
        self._proj: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_comp: Optional[np.ndarray] = None
        self._ecod = None
        self._copod = None
        self._ocsvm = None
        self._ensemble_views: List["NlpLofView"] = []
        self._fitted_method = "lof"
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None
        self.resolved_: Dict[str, Any] = {"enabled": False}

    def active(self) -> bool:
        ready = (
            self._lof is not None
            or self._nn is not None
            or self._center is not None
            or self._ecod is not None
            or self._copod is not None
            or self._ocsvm is not None
            or bool(self._ensemble_views)
        )
        return bool(self.enabled) and ready and abs(self.alpha) > 1e-12

    def get_params(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "alpha": self.alpha,
            "method": self.method,
            "mode": self.mode,
            "n_neighbors": self.n_neighbors,
            "memory_cap": self.memory_cap,
            "whiten": self.whiten,
            "whiten_dim": self.whiten_dim,
            "fuse_mode": self.fuse_mode,
            "tail_q": self.tail_q,
            "allow_datasets": list(self.allow_datasets),
            "fitted": self.active(),
            "resolved": dict(self.resolved_),
        }

    def dataset_allowed(self, hint: str) -> bool:
        h = (hint or "").lower()
        if not self.allow_datasets:
            return False
        return any(a in h for a in self.allow_datasets)

    def _fit_whiten(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        mu = X.mean(axis=0)
        Xc = X - mu
        # economy SVD; keep components with positive singular values
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        keep = int(min(self.whiten_dim, vt.shape[0], max(1, (s > 1e-8).sum())))
        comps = vt[:keep].T  # (d, k)
        scales = np.maximum(s[:keep] / np.sqrt(max(len(X) - 1, 1)), 1e-6)
        self._whiten_mean = mu.astype(np.float32)
        self._whiten_components = comps.astype(np.float32)
        self._whiten_scales = scales.astype(np.float32)
        return self._apply_whiten(X)

    def _apply_whiten(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if (
            self._whiten_mean is None
            or self._whiten_components is None
            or self._whiten_scales is None
        ):
            return X
        Xc = X - self._whiten_mean
        Z = Xc @ self._whiten_components
        return (Z / self._whiten_scales).astype(np.float32)

    _LEGAL_METHODS = (
        "lof",
        "knn",
        "iforest",
        "sphere",
        "sphere_trim",
        "ecod",
        "copod",
        "ocsvm",
        "cosine",
        "cosine_knn",
        "mcd",
        "dsvdd",
        "pca_recon",
        "ensemble",
    )

    def _reset_estimators(self) -> None:
        self._nn = None
        self._lof = None
        self._center = None
        self._proj = None
        self._cov_inv = None
        self._pca_mean = None
        self._pca_comp = None
        self._ecod = None
        self._copod = None
        self._ocsvm = None
        self._ensemble_views = []
        self._whiten_mean = None
        self._whiten_components = None
        self._whiten_scales = None
        self.mean_ = None
        self.std_ = None

    def fit(self, X_train: np.ndarray) -> None:
        self._reset_estimators()
        self.resolved_ = {"enabled": False}
        if not self.enabled:
            return
        X = np.asarray(X_train, dtype=np.float32)
        n = int(X.shape[0])
        if n < 16:
            self.resolved_ = {"enabled": False, "reason": "tiny_n", "n": n}
            return
        rng = np.random.RandomState(self.seed + 91)
        if n > int(self.memory_cap):
            idx = rng.choice(n, size=int(self.memory_cap), replace=False)
            mem = X[idx]
        else:
            mem = X
        if self.whiten:
            mem_w = self._fit_whiten(mem)
            X_w = self._apply_whiten(X)
        else:
            mem_w = mem
            X_w = X

        k = int(min(self.n_neighbors, max(3, len(mem_w) - 1)))
        self._k_query = k
        method = self.method if self.method in self._LEGAL_METHODS else "lof"
        self._fitted_method = method
        raw = self._fit_method(method, mem_w, X_w, k=k)

        self.mean_ = float(raw.mean())
        self.std_ = float(max(float(raw.std()), 1e-8))
        self.resolved_ = {
            "enabled": True,
            "n": n,
            "mem": int(len(mem_w)),
            "k": int(k),
            "method": method,
            "whiten": bool(self.whiten),
            "whiten_dim": int(self.whiten_dim) if self.whiten else 0,
            "mean": self.mean_,
            "std": self.std_,
        }

    def _fit_method(
        self, method: str, mem_w: np.ndarray, X_w: np.ndarray, *, k: int
    ) -> np.ndarray:
        if method == "lof":
            from sklearn.neighbors import LocalOutlierFactor

            lof = LocalOutlierFactor(
                n_neighbors=k,
                novelty=True,
                contamination="auto",
                n_jobs=1,
            )
            lof.fit(mem_w)
            self._lof = lof
            return -lof.decision_function(X_w).astype(np.float64)
        if method == "iforest":
            from sklearn.ensemble import IsolationForest

            iso = IsolationForest(
                n_estimators=200,
                contamination="auto",
                random_state=int(self.seed),
                n_jobs=1,
            )
            iso.fit(mem_w)
            self._lof = iso
            return -iso.score_samples(X_w).astype(np.float64)
        if method == "knn":
            from sklearn.neighbors import NearestNeighbors

            k_fit = min(max(k + 1, k), len(mem_w))
            nn = NearestNeighbors(n_neighbors=k_fit, algorithm="auto", n_jobs=1)
            nn.fit(mem_w)
            self._nn = nn
            d_tr, _ = nn.kneighbors(X_w)
            return self._mean_knn_dists(d_tr, drop_self=True)
        if method == "cosine_knn":
            from sklearn.neighbors import NearestNeighbors

            mem_n = self._l2_rows(mem_w)
            X_n = self._l2_rows(X_w)
            k_fit = min(max(k + 1, k), len(mem_n))
            nn = NearestNeighbors(
                n_neighbors=k_fit, algorithm="auto", metric="cosine", n_jobs=1
            )
            nn.fit(mem_n)
            self._nn = nn
            d_tr, _ = nn.kneighbors(X_n)
            return self._mean_knn_dists(d_tr, drop_self=True)
        if method in ("sphere", "sphere_trim", "cosine"):
            center = self._fit_center(mem_w, trim=(method == "sphere_trim"))
            self._center = center
            return self._sphere_raw(X_w, cosine=(method == "cosine"))
        if method == "dsvdd":
            return self._fit_dsvdd(mem_w, X_w)
        if method == "mcd":
            return self._fit_mcd(mem_w, X_w)
        if method == "pca_recon":
            return self._fit_pca_recon(mem_w, X_w)
        if method == "ecod":
            from axion.models.extensions.ecod_view import EcodView

            view = EcodView(enabled=True, alpha=1.0, max_fit_n=int(self.memory_cap))
            view.fit(mem_w)
            self._ecod = view
            return view.score_raw(X_w).astype(np.float64)
        if method == "copod":
            from axion.models.extensions.copod_view import CopodView

            view = CopodView(enabled=True, alpha=1.0, max_fit_n=int(self.memory_cap))
            view.fit(mem_w)
            self._copod = view
            return view.score_raw(X_w).astype(np.float64)
        if method == "ocsvm":
            from sklearn.svm import OneClassSVM

            nu = 0.05
            svm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
            svm.fit(mem_w)
            self._ocsvm = svm
            return -svm.decision_function(X_w).astype(np.float64)
        if method == "ensemble":
            return self._fit_ensemble(mem_w, X_w, k=k)
        # fallback knn
        from sklearn.neighbors import NearestNeighbors

        k_fit = min(max(k + 1, k), len(mem_w))
        nn = NearestNeighbors(n_neighbors=k_fit, algorithm="auto", n_jobs=1)
        nn.fit(mem_w)
        self._nn = nn
        d_tr, _ = nn.kneighbors(X_w)
        return self._mean_knn_dists(d_tr, drop_self=True)

    @staticmethod
    def _l2_rows(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        return (X / np.maximum(nrm, 1e-8)).astype(np.float32)

    def _fit_center(self, mem_w: np.ndarray, *, trim: bool) -> np.ndarray:
        mem = np.asarray(mem_w, dtype=np.float64)
        c = mem.mean(axis=0)
        if trim and len(mem) >= 32:
            d = np.sum((mem - c) ** 2, axis=1)
            keep = d <= np.quantile(d, 0.975)
            if int(keep.sum()) >= 16:
                c = mem[keep].mean(axis=0)
        return c.astype(np.float32)

    def _sphere_raw(self, X_w: np.ndarray, *, cosine: bool) -> np.ndarray:
        X = np.asarray(X_w, dtype=np.float64)
        c = np.asarray(self._center, dtype=np.float64)
        if cosine:
            xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
            cn = c / max(float(np.linalg.norm(c)), 1e-8)
            return (1.0 - xn @ cn).astype(np.float64)
        if self._proj is not None:
            W = np.asarray(self._proj, dtype=np.float64)
            X = X @ W
            c = c @ W if c.shape[0] == W.shape[0] else c
        if self._cov_inv is not None:
            diff = X - c
            return np.einsum("ij,jk,ik->i", diff, self._cov_inv, diff).astype(np.float64)
        return np.sum((X - c) ** 2, axis=1).astype(np.float64)

    def _fit_dsvdd(self, mem_w: np.ndarray, X_w: np.ndarray) -> np.ndarray:
        # Soft-identity DeepSVDD (P15) on train normals only. No test labels.
        Z = np.asarray(mem_w, dtype=np.float64)
        d = int(Z.shape[1])
        W = np.eye(d, dtype=np.float64)
        c = Z.mean(axis=0)
        n = max(len(Z), 1)
        lr = 0.04
        for _ in range(20):
            pred = Z @ W
            grad = (2.0 / n) * (Z.T @ (pred - c)) + 0.15 * (W - np.eye(d))
            W = W - lr * grad
        self._proj = W.astype(np.float32)
        self._center = c.astype(np.float32)
        return self._sphere_raw(X_w, cosine=False)

    def _fit_mcd(self, mem_w: np.ndarray, X_w: np.ndarray) -> np.ndarray:
        from sklearn.covariance import MinCovDet, EmpiricalCovariance

        Z = np.asarray(mem_w, dtype=np.float64)
        d = int(Z.shape[1])
        # MCD is O(n d^2); cap dim for 15Gi box
        if d > 48:
            _, s, vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
            keep = int(min(32, vt.shape[0], max(8, (s > 1e-8).sum())))
            P = vt[:keep].T
            Zp = Z @ P
            Xp = np.asarray(X_w, dtype=np.float64) @ P
            self._proj = P.astype(np.float32)
        else:
            Zp = Z
            Xp = np.asarray(X_w, dtype=np.float64)
        try:
            cov = MinCovDet(support_fraction=0.9, random_state=int(self.seed))
            cov.fit(Zp)
        except Exception:
            cov = EmpiricalCovariance()
            cov.fit(Zp)
        self._center = np.asarray(cov.location_, dtype=np.float32)
        self._cov_inv = np.asarray(cov.precision_, dtype=np.float64)
        diff = Xp - np.asarray(self._center, dtype=np.float64)
        return np.einsum("ij,jk,ik->i", diff, self._cov_inv, diff).astype(np.float64)

    def _fit_pca_recon(self, mem_w: np.ndarray, X_w: np.ndarray) -> np.ndarray:
        Z = np.asarray(mem_w, dtype=np.float64)
        mu = Z.mean(axis=0)
        _, s, vt = np.linalg.svd(Z - mu, full_matrices=False)
        keep = int(min(32, vt.shape[0], max(8, (s > 1e-8).sum())))
        P = vt[:keep]
        self._pca_mean = mu.astype(np.float32)
        self._pca_comp = P.astype(np.float32)
        self._center = mu.astype(np.float32)
        return self._pca_recon_raw(X_w)

    def _pca_recon_raw(self, X_w: np.ndarray) -> np.ndarray:
        X = np.asarray(X_w, dtype=np.float64)
        mu = np.asarray(self._pca_mean, dtype=np.float64)
        P = np.asarray(self._pca_comp, dtype=np.float64)
        Z = (X - mu) @ P.T
        rec = Z @ P + mu
        return np.sum((X - rec) ** 2, axis=1).astype(np.float64)

    def _fit_ensemble(self, mem_w: np.ndarray, X_w: np.ndarray, *, k: int) -> np.ndarray:
        # Complementary train-only views (P08/P12/P14/P15). Mean of z-scores.
        parts = ("sphere", "ecod", "lof")
        zs = []
        for part in parts:
            sub = NlpLofView(
                enabled=True,
                alpha=1.0,
                method=part,
                mode="replace",
                n_neighbors=int(self.n_neighbors),
                memory_cap=int(self.memory_cap),
                whiten=False,
                seed=int(self.seed),
            )
            # already whitened upstream; fit on mem_w directly
            sub.whiten = False
            sub._reset_estimators()
            sub._fitted_method = part
            raw_part = sub._fit_method(part, mem_w, X_w, k=k)
            sub.mean_ = float(raw_part.mean())
            sub.std_ = float(max(float(raw_part.std()), 1e-8))
            sub.resolved_ = {"enabled": True, "method": part}
            if sub.active():
                self._ensemble_views.append(sub)
                zs.append((raw_part - sub.mean_) / (sub.std_ + 1e-8))
        if not zs:
            return np.zeros(len(X_w), dtype=np.float64)
        return np.mean(np.stack(zs, axis=0), axis=0).astype(np.float64)

    def _mean_knn_dists(self, d: np.ndarray, *, drop_self: bool) -> np.ndarray:
        d = np.asarray(d, dtype=np.float64)
        k = int(getattr(self, "_k_query", self.n_neighbors) or self.n_neighbors)
        k = max(1, min(k, d.shape[1]))
        if drop_self and d.shape[1] > 1 and float(np.median(d[:, 0])) < 1e-6:
            take = d[:, 1 : 1 + k]
            if take.shape[1] == 0:
                take = d[:, :k]
        else:
            take = d[:, :k]
        return take.mean(axis=1).astype(np.float64)

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Xw = self._apply_whiten(X) if self.whiten else X
        method = str(getattr(self, "_fitted_method", self.method) or self.method)
        if self._ensemble_views:
            zs = []
            for sub in self._ensemble_views:
                zs.append(sub.z_normed(Xw))
            return np.mean(np.stack(zs, axis=0), axis=0).astype(np.float64)
        if self._lof is not None:
            # LOF novelty uses decision_function; IsolationForest uses score_samples
            if hasattr(self._lof, "decision_function") and method == "lof":
                return -self._lof.decision_function(Xw).astype(np.float64)
            if hasattr(self._lof, "score_samples"):
                return -self._lof.score_samples(Xw).astype(np.float64)
            return -self._lof.decision_function(Xw).astype(np.float64)
        if self._nn is not None:
            Xin = self._l2_rows(Xw) if method == "cosine_knn" else Xw
            d, _ = self._nn.kneighbors(Xin)
            return self._mean_knn_dists(d, drop_self=True)
        if self._ecod is not None:
            return self._ecod.score_raw(Xw).astype(np.float64)
        if self._copod is not None:
            return self._copod.score_raw(Xw).astype(np.float64)
        if self._ocsvm is not None:
            return -self._ocsvm.decision_function(Xw).astype(np.float64)
        if self._pca_comp is not None and self._pca_mean is not None:
            return self._pca_recon_raw(Xw)
        if self._center is not None:
            return self._sphere_raw(Xw, cosine=(method == "cosine"))
        return np.zeros(len(X), dtype=np.float64)

    def z_normed(self, X: np.ndarray) -> np.ndarray:
        raw = self.raw_scores(X)
        mu = float(self.mean_ if self.mean_ is not None else raw.mean())
        sd = float(self.std_ if self.std_ is not None else max(float(raw.std()), 1e-8))
        return (raw - mu) / (sd + 1e-8)

    def set_mcs_anchor(self, mcs_train: np.ndarray) -> None:
        mcs = np.asarray(mcs_train, dtype=np.float64)
        self._mcs_mu = float(mcs.mean())
        self._mcs_sd = float(max(float(mcs.std()), 1e-8))
        mcs_z = (mcs - self._mcs_mu) / (self._mcs_sd + 1e-8)
        self._tail_tau = float(np.quantile(mcs_z, self.tail_q))

    def fit_platt(
        self,
        mcs_n: np.ndarray,
        lof_n: np.ndarray,
        mcs_a: np.ndarray,
        lof_a: np.ndarray,
    ) -> None:
        from sklearn.linear_model import LogisticRegression

        mcs_zn = (np.asarray(mcs_n, dtype=np.float64) - self._mcs_mu) / (self._mcs_sd + 1e-8)
        mcs_za = (np.asarray(mcs_a, dtype=np.float64) - self._mcs_mu) / (self._mcs_sd + 1e-8)
        y = np.concatenate(
            [np.zeros(len(lof_n), dtype=np.int64), np.ones(len(lof_a), dtype=np.int64)]
        )
        X_full = np.column_stack(
            [
                np.concatenate([mcs_zn, mcs_za]),
                np.concatenate([np.asarray(lof_n, dtype=np.float64), np.asarray(lof_a, dtype=np.float64)]),
            ]
        )
        try:
            full = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=250, C=1.0)
            mcs_only = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=250, C=1.0)
            full.fit(X_full, y)
            mcs_only.fit(X_full[:, :1], y)
        except Exception:
            self._platt_full = None
            self._platt_mcs = None
            return
        self._platt_full = full
        self._platt_mcs = mcs_only
        resid = self._platt_residual(mcs_zn, np.asarray(lof_n, dtype=np.float64))
        self._platt_mu = float(resid.mean())
        self._platt_sd = float(max(float(resid.std()), 1e-8))

    def _platt_residual(self, mcs_z: np.ndarray, lof_z: np.ndarray) -> np.ndarray:
        if self._platt_full is None or self._platt_mcs is None:
            return np.asarray(lof_z, dtype=np.float64)
        X_full = np.column_stack([mcs_z, lof_z]).astype(np.float64)
        logit_full = np.asarray(self._platt_full.decision_function(X_full), dtype=np.float64)
        logit_mcs = np.asarray(self._platt_mcs.decision_function(X_full[:, :1]), dtype=np.float64)
        return logit_full - logit_mcs

    def transform(self, lof_z: np.ndarray, mcs_raw: np.ndarray) -> np.ndarray:
        """Train-only fuse of LOF z vs MCS (add / tail / platt / platt_tail)."""
        delta = np.asarray(lof_z, dtype=np.float64)
        mode = str(self.fuse_mode)
        if mode == "add":
            return delta
        mcs_z = (np.asarray(mcs_raw, dtype=np.float64) - self._mcs_mu) / (self._mcs_sd + 1e-8)
        if mode in ("platt", "platt_tail"):
            resid = self._platt_residual(mcs_z, delta)
            delta = (resid - float(self._platt_mu)) / (float(self._platt_sd) + 1e-8)
        if mode in ("tail", "platt_tail"):
            gate = 1.0 / (1.0 + np.exp(-(mcs_z - float(self._tail_tau)) / float(self.tail_soft)))
            delta = delta * gate
        return delta


__all__ = ["NlpLofView"]

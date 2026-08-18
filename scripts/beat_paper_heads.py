#!/usr/bin/env python3
"""Beat-Paper detector zoo: HBOS, LODA, OCSVM, MCD, PCA recon, DTE, bagging, ensemble, ICL-lite.

Leap methods: SOD-lite, CBLOF-lite, avg-kNN, elliptical, fusion heads, soft-vote, LSCP-lite, INNE-lite.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

FIT_N_CAP = 16000
ZOO_METHODS = (
    "hbos",
    "loda",
    "ocsvm",
    "mcd",
    "pca_recon",
    "dte",
    "bagging",
    "rank_ensemble",
    "icl",
    "knn_cosine",
    "gmm",
    "kde",
    "mahalanobis",
    "abod_lite",
    "goad",
    "rank_max_ensemble",
    "sod_lite",
    "cblof_lite",
    "avg_knn",
    "elliptical",
    "fusion_knn_maha",
    "fusion_hbos_iforest",
    "soft_vote_ensemble",
    "lscp_lite",
    "inne_lite",
    "pca_residual_knn",
    "fusion_ecod_knn",
)
KDE_MAX_DIM = 32
KDE_MAX_N = 4000
MAHA_MAX_DIM = 64
ABOD_MAX_N = 800
ABOD_MAX_DIM = 32
SOD_MAX_DIM = 256
CBLOF_MAX_N = 12000
INNE_MAX_N = 8000


def cap_fit(X: np.ndarray, *, cap: int = FIT_N_CAP, seed: int = 0) -> np.ndarray:
    if len(X) <= cap:
        return X
    rng = np.random.RandomState(int(seed))
    return X[rng.choice(len(X), size=int(cap), replace=False)]


def zscore_fit_query(X_fit: np.ndarray, X_query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X_fit.mean(axis=0, keepdims=True)
    sd = np.maximum(X_fit.std(axis=0, keepdims=True), 1e-6)
    return ((X_fit - mu) / sd).astype(np.float32), ((X_query - mu) / sd).astype(np.float32)


def pca_view(
    X_fit: np.ndarray,
    X_query: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    k = min(int(n_components), X_fit.shape[1], max(1, len(X_fit) - 1))
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
    return pca.fit_transform(X_fit).astype(np.float32), pca.transform(X_query).astype(np.float32)


def rank_norm(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64).ravel()
    order = np.argsort(s)
    r = np.empty(len(s), dtype=np.float64)
    r[order] = np.linspace(0.0, 1.0, len(s), endpoint=True)
    return r


def hbos_scores(X_fit: np.ndarray, X_query: np.ndarray, *, n_bins: int = 10) -> np.ndarray:
    X_fit = cap_fit(X_fit)
    d = X_fit.shape[1]
    scores = np.zeros(len(X_query), dtype=np.float64)
    bins = max(4, int(n_bins))
    for j in range(d):
        col = X_fit[:, j]
        lo, hi = float(np.min(col)), float(np.max(col))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
            continue
        try:
            hist, edges = np.histogram(col, bins=bins, range=(lo, hi), density=True)
        except ValueError:
            continue
        hist = np.maximum(hist, 1e-12)
        q = np.clip(X_query[:, j], lo, hi)
        idx = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, len(hist) - 1)
        scores += -np.log(hist[idx])
    return scores


def loda_scores(
    X_fit: np.ndarray,
    X_query: np.ndarray,
    *,
    n_proj: int = 100,
    n_bins: int = 20,
) -> np.ndarray:
    X_fit = cap_fit(X_fit)
    rng = np.random.RandomState(0)
    d = X_fit.shape[1]
    n_proj = int(min(max(8, n_proj), 128))
    W = rng.randn(d, n_proj).astype(np.float32)
    W /= np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-6)
    P_fit = X_fit @ W
    P_q = X_query @ W
    scores = np.zeros(len(X_query), dtype=np.float64)
    for k in range(n_proj):
        col = P_fit[:, k]
        lo, hi = float(np.min(col)), float(np.max(col))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
            continue
        try:
            hist, edges = np.histogram(col, bins=max(8, n_bins), range=(lo, hi), density=True)
        except ValueError:
            continue
        hist = np.maximum(hist, 1e-12)
        q = np.clip(P_q[:, k], lo, hi)
        idx = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, len(hist) - 1)
        scores += -np.log(hist[idx])
    return scores / float(n_proj)


def ocsvm_scores(X_fit: np.ndarray, X_query: np.ndarray, *, nu: float = 0.05) -> np.ndarray:
    from sklearn.svm import OneClassSVM

    X_use = cap_fit(X_fit, cap=4000)
    clf = OneClassSVM(kernel="rbf", gamma="scale", nu=min(max(float(nu), 0.01), 0.5))
    clf.fit(X_use)
    return -clf.decision_function(X_query)


def mcd_scores(X_fit: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    from sklearn.covariance import EllipticEnvelope

    if X_fit.shape[1] > 40:
        raise ValueError("mcd skip high-d")
    X_use = cap_fit(X_fit, cap=2000)
    clf = EllipticEnvelope(support_fraction=0.9, random_state=0, contamination=0.1)
    clf.fit(X_use)
    return -clf.decision_function(X_query)


def pca_recon_scores(X_fit: np.ndarray, X_query: np.ndarray, *, n_components: int = 8) -> np.ndarray:
    from sklearn.decomposition import PCA

    X_use = cap_fit(X_fit)
    k = min(int(n_components), X_use.shape[1], max(1, len(X_use) - 1))
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
    pca.fit(X_use)
    rec = pca.inverse_transform(pca.transform(X_query))
    return np.sum((X_query - rec) ** 2, axis=1)


def dte_lite_scores(X_fit: np.ndarray, X_query: np.ndarray, *, n_scales: int = 4) -> np.ndarray:
    rng = np.random.RandomState(0)
    X_tr = cap_fit(X_fit)
    center = X_tr.mean(axis=0)
    scores = np.zeros(len(X_query), dtype=np.float64)
    for s in range(1, int(n_scales) + 1):
        sigma = 0.1 * s
        noise = rng.randn(*X_query.shape).astype(X_query.dtype) * sigma
        scores += np.linalg.norm((X_query + noise) - center, axis=1)
    return scores / float(n_scales)


def feature_bagging_scores(
    X_fit: np.ndarray,
    X_query: np.ndarray,
    *,
    n_bags: int = 8,
    n_neighbors: int = 20,
) -> np.ndarray:
    from sklearn.neighbors import LocalOutlierFactor

    X_use = cap_fit(X_fit)
    if not _cheap_enough_for_lof(X_use, X_query):
        return _iforest_scores(X_fit, X_query)
    rng = np.random.RandomState(0)
    d = X_use.shape[1]
    width = max(2, d // 2)
    acc = np.zeros(len(X_query), dtype=np.float64)
    used = 0
    for _ in range(int(n_bags)):
        cols = rng.choice(d, size=min(width, d), replace=False)
        k = min(int(n_neighbors), max(2, len(X_use) - 1))
        clf = LocalOutlierFactor(n_neighbors=k, novelty=True, contamination=0.05)
        try:
            clf.fit(X_use[:, cols])
            raw = -clf.score_samples(X_query[:, cols])
        except Exception:
            continue
        acc += rank_norm(raw)
        used += 1
    if used == 0:
        raise RuntimeError("feature_bagging failed")
    return acc / float(used)


def icl_lite_scores(
    X_fit: np.ndarray,
    X_query: np.ndarray,
    *,
    n_masks: int = 8,
    mask_frac: float = 0.25,
) -> np.ndarray:
    """P21-lite: mask random features, ridge-reconstruct, residual energy."""
    from sklearn.linear_model import Ridge

    X_use = cap_fit(X_fit, cap=8000)
    rng = np.random.RandomState(0)
    d = X_use.shape[1]
    n_mask = max(1, int(round(d * float(mask_frac))))
    n_mask = min(n_mask, d - 1) if d > 1 else 1
    scores = np.zeros(len(X_query), dtype=np.float64)
    used = 0
    for _ in range(int(n_masks)):
        if d == 1:
            scores += np.abs(X_query[:, 0] - float(np.mean(X_use[:, 0])))
            used += 1
            continue
        idx = np.sort(rng.choice(d, size=n_mask, replace=False))
        keep = np.array([i for i in range(d) if i not in set(idx.tolist())], dtype=np.int64)
        if len(keep) == 0:
            continue
        model = Ridge(alpha=1.0)
        model.fit(X_use[:, keep], X_use[:, idx])
        pred = model.predict(X_query[:, keep])
        tgt = X_query[:, idx]
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
            tgt = tgt.reshape(-1, 1)
        scores += np.sum((tgt - pred) ** 2, axis=1)
        used += 1
    if used == 0:
        raise RuntimeError("icl_lite failed")
    return scores / float(used)


def _cheap_enough_for_lof(X_fit: np.ndarray, X_query: np.ndarray) -> bool:
    n_fit = min(len(X_fit), FIT_N_CAP)
    n_q = len(X_query)
    d = int(X_fit.shape[1])
    return n_q <= 60000 and d <= 48 and (float(n_fit) * float(n_q) * float(max(d, 1))) < 2e10


def _iforest_scores(X_fit: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    from sklearn.ensemble import IsolationForest

    X_use = cap_fit(X_fit)
    clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=0, n_jobs=1)
    clf.fit(X_use)
    return -clf.score_samples(X_query)


def _lof_scores(X_fit: np.ndarray, X_query: np.ndarray, *, n_neighbors: int = 20) -> np.ndarray:
    from sklearn.neighbors import LocalOutlierFactor

    if not _cheap_enough_for_lof(X_fit, X_query):
        return _iforest_scores(X_fit, X_query)
    X_use = cap_fit(X_fit)
    k = min(int(n_neighbors), max(2, len(X_use) - 1))
    clf = LocalOutlierFactor(n_neighbors=k, novelty=True, contamination=0.05)
    clf.fit(X_use)
    return -clf.score_samples(X_query)


def _copod_ecod_scores(X_fit: np.ndarray, X_query: np.ndarray, *, kind: str) -> np.ndarray:
    from axion.models.extensions.copod_view import CopodView
    from axion.models.extensions.ecod_view import EcodView

    view = (
        CopodView(enabled=True, alpha=1.0, max_fit_n=20000)
        if kind == "copod"
        else EcodView(enabled=True, alpha=1.0, max_fit_n=20000)
    )
    view.fit(cap_fit(X_fit, cap=20000))
    return np.asarray(view.score_raw(X_query), dtype=np.float64)


def knn_cosine_scores(
    X_fit: np.ndarray,
    X_query: np.ndarray,
    *,
    n_neighbors: int = 20,
) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    X_use = cap_fit(X_fit)
    k = min(int(n_neighbors), max(1, len(X_use)))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(X_use)
    dist, _ = nn.kneighbors(X_query, n_neighbors=k, return_distance=True)
    return np.mean(dist, axis=1).astype(np.float64)


def gmm_scores(
    X_fit: np.ndarray,
    X_query: np.ndarray,
    *,
    n_components: Optional[int] = None,
) -> np.ndarray:
    from sklearn.mixture import GaussianMixture

    X_use = cap_fit(X_fit, cap=8000)
    n = len(X_use)
    # Unset: adaptive mixture (native GMM / ensembles). PCA64 ViT/E5 inject n_components via catalog.
    k = int(n_components) if n_components is not None else min(8, max(2, n // 50))
    k = min(max(1, k), max(1, n))
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=0,
        max_iter=80,
        reg_covar=1e-4,
    )
    try:
        gmm.fit(np.asarray(X_use, dtype=np.float64))
        return (-gmm.score_samples(np.asarray(X_query, dtype=np.float64))).astype(np.float64)
    except Exception as exc:
        raise ValueError(f"gmm skip: {exc}") from exc


def kde_scores(X_fit: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    from sklearn.neighbors import KernelDensity

    d = int(X_fit.shape[1])
    if d > KDE_MAX_DIM:
        raise ValueError(f"kde skip high-d ({d} > {KDE_MAX_DIM})")
    X_use = cap_fit(X_fit, cap=KDE_MAX_N)
    if len(X_use) > KDE_MAX_N:
        raise ValueError("kde skip large-n")
    try:
        kde = KernelDensity(bandwidth="scott", kernel="gaussian")
    except (ValueError, TypeError):
        kde = KernelDensity(bandwidth=0.5, kernel="gaussian")
    kde.fit(X_use)
    return (-kde.score_samples(X_query)).astype(np.float64)


def mahalanobis_scores(X_fit: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    from sklearn.covariance import ShrunkCovariance

    d = int(X_fit.shape[1])
    if d > MAHA_MAX_DIM:
        raise ValueError(f"mahalanobis skip high-d ({d} > {MAHA_MAX_DIM})")
    X_use = cap_fit(X_fit, cap=8000)
    cov = ShrunkCovariance()
    cov.fit(np.asarray(X_use, dtype=np.float64))
    return np.asarray(cov.mahalanobis(np.asarray(X_query, dtype=np.float64)), dtype=np.float64)


def abod_lite_scores(X_fit: np.ndarray, X_query: np.ndarray, *, n_neighbors: int = 20) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    d = int(X_fit.shape[1])
    if d > ABOD_MAX_DIM:
        raise ValueError(f"abod skip high-d ({d} > {ABOD_MAX_DIM})")
    X_use = cap_fit(X_fit, cap=ABOD_MAX_N)
    if len(X_use) < 8:
        raise ValueError("abod skip tiny-n")
    k = min(int(n_neighbors), max(3, len(X_use) - 1))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(X_use)
    idx = nn.kneighbors(X_query, n_neighbors=k, return_distance=False)
    scores = np.zeros(len(X_query), dtype=np.float64)
    for i in range(len(X_query)):
        q = X_query[i]
        neigh = X_use[idx[i]]
        vecs = neigh - q
        norms = np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)
        unit = vecs / norms
        grams = unit @ unit.T
        iu = np.triu_indices(len(unit), k=1)
        if len(iu[0]) == 0:
            scores[i] = 0.0
            continue
        var = float(np.var(grams[iu]))
        scores[i] = 1.0 / (var + 1e-8)
    return scores


def goad_scores(X_fit: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    from axion.models.extensions.goad_lite import GoadLiteExtension

    ext = GoadLiteExtension(
        enabled=True,
        device="cpu",
        epochs=12,
        max_fit_n=2000,
        n_transforms=4,
        seed=111,
    )
    ext.fit(np.asarray(X_fit, dtype=np.float32))
    if not ext.fitted_ or ext.head is None:
        raise ValueError("goad skip: not fitted")
    return np.asarray(ext.score_raw(np.asarray(X_query, dtype=np.float32)), dtype=np.float64)


def _ensemble_parts(X_fit: np.ndarray, X_query: np.ndarray) -> list:
    parts = []
    for fn in (
        lambda: _iforest_scores(X_fit, X_query),
        lambda: _lof_scores(X_fit, X_query),
        lambda: dte_lite_scores(X_fit, X_query),
        lambda: hbos_scores(X_fit, X_query),
        lambda: pca_recon_scores(X_fit, X_query),
        lambda: knn_cosine_scores(X_fit, X_query),
        lambda: gmm_scores(X_fit, X_query),
        lambda: mahalanobis_scores(X_fit, X_query),
    ):
        try:
            parts.append(rank_norm(fn()))
        except Exception:
            continue
    if X_fit.shape[1] <= 256:
        for kind in ("copod", "ecod"):
            try:
                parts.append(rank_norm(_copod_ecod_scores(X_fit, X_query, kind=kind)))
            except Exception:
                continue
    return parts


def rank_ensemble_scores(X_fit: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    parts = _ensemble_parts(X_fit, X_query)
    if not parts:
        raise RuntimeError("rank_ensemble empty")
    return np.mean(np.stack(parts, axis=0), axis=0)


def rank_max_ensemble_scores(X_fit: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    parts = _ensemble_parts(X_fit, X_query)
    if not parts:
        raise RuntimeError("rank_max_ensemble empty")
    return np.max(np.stack(parts, axis=0), axis=0)


def sod_lite_scores(
    X_fit: np.ndarray,
    X_query: np.ndarray,
    *,
    n_subspaces: int = 24,
    subspace_dim: int = 8,
    n_neighbors: int = 15,
) -> np.ndarray:
    """Random-subspace kNN distance ensemble (SOD-lite)."""
    from sklearn.neighbors import NearestNeighbors

    X_fit = cap_fit(X_fit)
    d = X_fit.shape[1]
    if d < 2:
        raise ValueError("sod skip tiny-d")
    if d > SOD_MAX_DIM:
        X_fit, X_query = pca_view(X_fit, X_query, SOD_MAX_DIM)
        d = X_fit.shape[1]
    rng = np.random.RandomState(0)
    sub_d = max(2, min(int(subspace_dim), d))
    k = max(1, min(int(n_neighbors), len(X_fit) - 1))
    scores = np.zeros(len(X_query), dtype=np.float64)
    n_ok = 0
    for _ in range(int(max(4, n_subspaces))):
        cols = rng.choice(d, size=sub_d, replace=False)
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(X_fit[:, cols])
        dist, _ = nn.kneighbors(X_query[:, cols])
        scores += dist.mean(axis=1)
        n_ok += 1
    if n_ok == 0:
        raise RuntimeError("sod empty")
    return scores / float(n_ok)


def cblof_lite_scores(
    X_fit: np.ndarray,
    X_query: np.ndarray,
    *,
    n_clusters: int = 8,
) -> np.ndarray:
    """Cluster-based local outlier factor lite (distance to nearest large cluster)."""
    from sklearn.cluster import MiniBatchKMeans

    X_fit = cap_fit(X_fit, cap=CBLOF_MAX_N)
    n = len(X_fit)
    k = max(2, min(int(n_clusters), max(2, n // 20)))
    km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=min(1024, max(32, n // 4)), n_init=3)
    labels = km.fit_predict(X_fit)
    centers = km.cluster_centers_
    sizes = np.bincount(labels, minlength=k).astype(np.float64)
    large = sizes >= max(2.0, 0.1 * float(n) / float(k))
    if not np.any(large):
        large = sizes >= np.median(sizes)
    large_idx = np.where(large)[0]
    # Score = min distance to a large cluster center, scaled by inverse size
    dmat = np.linalg.norm(X_query[:, None, :] - centers[None, large_idx, :], axis=2)
    inv = 1.0 / np.maximum(sizes[large_idx], 1.0)
    return (dmat * inv[None, :]).min(axis=1)


def avg_knn_scores(X_fit: np.ndarray, X_query: np.ndarray, *, n_neighbors: int = 20) -> np.ndarray:
    """Average Euclidean distance to k nearest fit neighbors."""
    from sklearn.neighbors import NearestNeighbors

    X_fit = cap_fit(X_fit)
    k = max(1, min(int(n_neighbors), len(X_fit) - 1))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(X_fit)
    dist, _ = nn.kneighbors(X_query)
    return dist.mean(axis=1)


def elliptical_scores(X_fit: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """EllipticEnvelope / robust Mahalanobis decision_function (higher = more outlier)."""
    from sklearn.covariance import EllipticEnvelope

    X_fit = cap_fit(X_fit)
    d = X_fit.shape[1]
    if d > MAHA_MAX_DIM:
        X_fit, X_query = pca_view(X_fit, X_query, MAHA_MAX_DIM)
    if len(X_fit) < max(10, X_fit.shape[1] + 2):
        raise ValueError("elliptical skip tiny-n")
    contam = float(np.clip(0.05, 0.01, 0.2))
    ee = EllipticEnvelope(contamination=contam, random_state=0, support_fraction=None)
    ee.fit(X_fit)
    # decision_function: larger = more inlier → invert
    return -np.asarray(ee.decision_function(X_query), dtype=np.float64)


def fusion_knn_maha_scores(X_fit: np.ndarray, X_query: np.ndarray, *, n_neighbors: int = 20) -> np.ndarray:
    parts = []
    try:
        parts.append(rank_norm(knn_cosine_scores(X_fit, X_query, n_neighbors=n_neighbors)))
    except Exception:
        pass
    try:
        parts.append(rank_norm(mahalanobis_scores(X_fit, X_query)))
    except Exception:
        pass
    try:
        parts.append(rank_norm(avg_knn_scores(X_fit, X_query, n_neighbors=n_neighbors)))
    except Exception:
        pass
    if not parts:
        raise RuntimeError("fusion_knn_maha empty")
    return np.mean(np.stack(parts, axis=0), axis=0)


def fusion_hbos_iforest_scores(X_fit: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    parts = []
    try:
        parts.append(rank_norm(hbos_scores(X_fit, X_query)))
    except Exception:
        pass
    try:
        parts.append(rank_norm(_iforest_scores(X_fit, X_query)))
    except Exception:
        pass
    try:
        parts.append(rank_norm(loda_scores(X_fit, X_query)))
    except Exception:
        pass
    if not parts:
        raise RuntimeError("fusion_hbos_iforest empty")
    return np.mean(np.stack(parts, axis=0), axis=0)


def fusion_ecod_knn_scores(X_fit: np.ndarray, X_query: np.ndarray, *, n_neighbors: int = 20) -> np.ndarray:
    """ECOD density + cosine kNN soft rank fusion (tabular / embed tails)."""
    parts = []
    try:
        parts.append(rank_norm(_copod_ecod_scores(X_fit, X_query, kind="ecod")))
    except Exception:
        pass
    try:
        parts.append(rank_norm(knn_cosine_scores(X_fit, X_query, n_neighbors=n_neighbors)))
    except Exception:
        pass
    try:
        parts.append(rank_norm(avg_knn_scores(X_fit, X_query, n_neighbors=n_neighbors)))
    except Exception:
        pass
    if not parts:
        raise RuntimeError("fusion_ecod_knn empty")
    return np.mean(np.stack(parts, axis=0), axis=0)


def soft_vote_ensemble_scores(X_fit: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """Broader soft vote than rank_ensemble (adds SOD / avg-kNN / elliptical when cheap)."""
    parts = _ensemble_parts(X_fit, X_query)
    extras = (
        lambda: sod_lite_scores(X_fit, X_query),
        lambda: avg_knn_scores(X_fit, X_query),
        lambda: elliptical_scores(X_fit, X_query),
        lambda: fusion_hbos_iforest_scores(X_fit, X_query),
    )
    for fn in extras:
        try:
            parts.append(rank_norm(fn()))
        except Exception:
            continue
    if not parts:
        raise RuntimeError("soft_vote empty")
    return np.mean(np.stack(parts, axis=0), axis=0)


def lscp_lite_scores(X_fit: np.ndarray, X_query: np.ndarray, *, n_neighbors: int = 20) -> np.ndarray:
    """Locally select the most agreeing base model (LSCP-lite via local rank variance)."""
    from sklearn.neighbors import NearestNeighbors

    X_fit = cap_fit(X_fit)
    panel = []
    for fn in (
        lambda: _iforest_scores(X_fit, X_query),
        lambda: hbos_scores(X_fit, X_query),
        lambda: knn_cosine_scores(X_fit, X_query),
        lambda: avg_knn_scores(X_fit, X_query),
        lambda: pca_recon_scores(X_fit, X_query),
    ):
        try:
            panel.append(rank_norm(fn()))
        except Exception:
            continue
    if len(panel) < 2:
        raise RuntimeError("lscp panel tiny")
    P = np.stack(panel, axis=0)  # (M, Nq)
    # Local peer agreement on fit set via query self-scores approximate: use global std pick
    # Prefer models with high local contrast: score = selected model rank
    k = max(1, min(int(n_neighbors), len(X_query) - 1)) if len(X_query) > 1 else 1
    if len(X_query) < 3:
        return np.mean(P, axis=0)
    nn = NearestNeighbors(n_neighbors=min(k, len(X_query)), algorithm="auto")
    nn.fit(X_query)
    _, idx = nn.kneighbors(X_query)
    out = np.zeros(len(X_query), dtype=np.float64)
    for i in range(len(X_query)):
        local = P[:, idx[i]]
        # model with highest local mean-minus-std (peaky high scores)
        strength = local.mean(axis=1) - local.std(axis=1)
        m = int(np.argmax(strength))
        out[i] = P[m, i]
    return out


def inne_lite_scores(X_fit: np.ndarray, X_query: np.ndarray, *, n_estimators: int = 64) -> np.ndarray:
    """Isolation-based Nearest-Neighbor Ensemble lite (hypersphere isolation counts)."""
    X_fit = cap_fit(X_fit, cap=INNE_MAX_N)
    rng = np.random.RandomState(0)
    n = len(X_fit)
    if n < 8:
        raise ValueError("inne skip tiny-n")
    psi = max(4, min(32, n // 4))
    scores = np.zeros(len(X_query), dtype=np.float64)
    for _ in range(int(max(16, n_estimators))):
        centers_idx = rng.choice(n, size=psi, replace=False)
        centers = X_fit[centers_idx]
        # radius = min distance to other centers
        from sklearn.metrics import pairwise_distances

        dcc = pairwise_distances(centers)
        np.fill_diagonal(dcc, np.inf)
        radii = dcc.min(axis=1)
        radii = np.maximum(radii, 1e-6)
        dq = pairwise_distances(X_query, centers)
        # isolated if outside all hyperspheres → high score; else min normalized margin
        inside = dq <= radii[None, :]
        # count how many spheres contain the point (fewer = more outlier)
        scores += 1.0 - inside.mean(axis=1)
    return scores / float(max(16, n_estimators))


def pca_residual_knn_scores(
    X_fit: np.ndarray,
    X_query: np.ndarray,
    *,
    n_components: int = 16,
    n_neighbors: int = 15,
) -> np.ndarray:
    """kNN distance in PCA residual space (orthogonal complement of top components)."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    X_fit = cap_fit(X_fit)
    d = X_fit.shape[1]
    k_pca = max(1, min(int(n_components), d - 1, max(1, len(X_fit) - 1)))
    pca = PCA(n_components=k_pca, svd_solver="randomized", random_state=0)
    Zf = pca.fit_transform(X_fit)
    Zq = pca.transform(X_query)
    Rf = X_fit - pca.inverse_transform(Zf)
    Rq = X_query - pca.inverse_transform(Zq)
    k = max(1, min(int(n_neighbors), len(Rf) - 1))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(Rf)
    dist, _ = nn.kneighbors(Rq)
    return dist.mean(axis=1)


def score_zoo(
    method: str,
    X_fit: np.ndarray,
    X_query: np.ndarray,
    *,
    pca_dim: Optional[int] = None,
    zscore: bool = True,
    n_neighbors: Optional[int] = None,
    n_components: Optional[int] = None,
) -> np.ndarray:
    X_tr, X_te = X_fit, X_query
    if zscore:
        X_tr, X_te = zscore_fit_query(X_tr, X_te)
    if pca_dim and int(pca_dim) > 0:
        X_tr, X_te = pca_view(X_tr, X_te, int(pca_dim))
    name = str(method).lower()
    if name in ("hbos",):
        return hbos_scores(X_tr, X_te)
    if name in ("loda",):
        return loda_scores(X_tr, X_te)
    if name in ("ocsvm",):
        return ocsvm_scores(X_tr, X_te)
    if name in ("mcd",):
        return mcd_scores(X_tr, X_te)
    if name in ("pca_recon", "pca"):
        return pca_recon_scores(X_tr, X_te)
    if name in ("dte", "dte_lite"):
        return dte_lite_scores(X_tr, X_te)
    if name in ("bagging", "feature_bagging"):
        return feature_bagging_scores(X_tr, X_te)
    if name in ("rank_ensemble", "ensemble"):
        return rank_ensemble_scores(X_tr, X_te)
    if name in ("icl", "icl_lite"):
        return icl_lite_scores(X_tr, X_te)
    if name in ("knn_cosine", "cosine_knn"):
        k = 20 if n_neighbors is None else int(n_neighbors)
        return knn_cosine_scores(X_tr, X_te, n_neighbors=k)
    if name in ("gmm",):
        return gmm_scores(X_tr, X_te, n_components=n_components)
    if name in ("kde",):
        return kde_scores(X_tr, X_te)
    if name in ("mahalanobis", "maha"):
        return mahalanobis_scores(X_tr, X_te)
    if name in ("abod", "abod_lite"):
        k = 20 if n_neighbors is None else int(n_neighbors)
        return abod_lite_scores(X_tr, X_te, n_neighbors=k)
    if name in ("goad", "goad_lite"):
        return goad_scores(X_tr, X_te)
    if name in ("rank_max_ensemble", "rank_max"):
        return rank_max_ensemble_scores(X_tr, X_te)
    if name in ("sod", "sod_lite"):
        k = 15 if n_neighbors is None else int(n_neighbors)
        return sod_lite_scores(X_tr, X_te, n_neighbors=k)
    if name in ("cblof", "cblof_lite"):
        return cblof_lite_scores(X_tr, X_te, n_clusters=n_components or 8)
    if name in ("avg_knn", "knn_avg"):
        k = 20 if n_neighbors is None else int(n_neighbors)
        return avg_knn_scores(X_tr, X_te, n_neighbors=k)
    if name in ("elliptical", "elliptic"):
        return elliptical_scores(X_tr, X_te)
    if name in ("fusion_knn_maha", "knn_maha"):
        k = 20 if n_neighbors is None else int(n_neighbors)
        return fusion_knn_maha_scores(X_tr, X_te, n_neighbors=k)
    if name in ("fusion_hbos_iforest", "hbos_iforest"):
        return fusion_hbos_iforest_scores(X_tr, X_te)
    if name in ("soft_vote_ensemble", "soft_vote"):
        return soft_vote_ensemble_scores(X_tr, X_te)
    if name in ("lscp_lite", "lscp"):
        k = 20 if n_neighbors is None else int(n_neighbors)
        return lscp_lite_scores(X_tr, X_te, n_neighbors=k)
    if name in ("inne_lite", "inne"):
        return inne_lite_scores(X_tr, X_te)
    if name in ("pca_residual_knn", "pca_resid_knn"):
        k = 15 if n_neighbors is None else int(n_neighbors)
        return pca_residual_knn_scores(X_tr, X_te, n_neighbors=k, n_components=n_components or 16)
    if name in ("fusion_ecod_knn", "ecod_knn"):
        k = 20 if n_neighbors is None else int(n_neighbors)
        return fusion_ecod_knn_scores(X_tr, X_te, n_neighbors=k)
    raise ValueError(f"unknown zoo method {method}")

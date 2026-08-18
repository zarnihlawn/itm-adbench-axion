#!/usr/bin/env python3
"""Beat-Paper closed candidate catalog: per-dataset (embed, method) pairs.

Same 57 ADBench tasks and AnoDDAE protocol; disclosed routing only.
Not Table-1 fair ResNet+BERT AXION.

  python scripts/beat_paper_catalog.py --assert-catalog
  python scripts/beat_paper_catalog.py --list cover
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.eval.metrics import PAPER_DDAE  # noqa: E402

RUN_ID = "axion_beat_paper"
THESIS_DIR = ROOT / "results" / "axion_beat_paper"
RECIPE_MAP_PATH = THESIS_DIR / "thesis" / "recipe_map_57_beat.json"

PROMOTE_DELTA = 0.5
ROC_FLOOR_DELTA = -0.5

PAPER_TARGETS = {
    "semi-supervised": {"PR-AUC": 68.0, "ROC-AUC": 88.0},
    "unsupervised": {"PR-AUC": 45.0, "ROC-AUC": 88.0},
}

CLAIM_MARGIN_VS_PAPER = 3.0
LOCAL_BEYOND_MARGIN_VS_PAPER = 2.0

AXION_FINAL_FLOORS = {
    "semi-supervised": {"PR-AUC": 52.60, "ROC-AUC": 81.15},
    "unsupervised": {"PR-AUC": 33.42, "ROC-AUC": 75.40},
}

GATES = {
    "G_bp_semi_pr": ("semi-supervised", "PR-AUC", 68.0),
    "G_bp_semi_roc": ("semi-supervised", "ROC-AUC", 88.0),
    "G_bp_unsup_pr": ("unsupervised", "PR-AUC", 45.0),
    "G_bp_unsup_roc": ("unsupervised", "ROC-AUC", 88.0),
    "G_bp_12probe_semi_pr": ("semi-supervised", "PR-AUC", 64.0),
}

AXION_FINAL_COMPARE = ROOT / "results" / "axion_final" / "compare_to_ddae.json"
AXION_HOLD_RECIPE = "axion_edge_rare"

GUARDS: Set[str] = {
    "breastw", "cardio", "backdoor", "glass", "musk", "shuttle", "satimage-2",
}

LOCKS: Dict[str, float] = {
    "breastw": 97.11, "cardio": 76.65, "glass": 38.72, "backdoor": 79.68,
    "musk": 99.0, "shuttle": 99.0, "satimage-2": 97.53, "cover": 43.79,
    "fraud": 47.77, "thyroid": 71.25, "WDBC": 95.0, "Ionosphere": 90.0,
    "CIFAR10": 20.35, "FashionMNIST": 60.08, "Agnews": 23.26, "Imdb": 9.24,
}

ICL_DATASETS: Set[str] = {"speech", "ALOI", "Waveform", "optdigits", "InternetAds"}
AXION_LOCKED: Set[str] = {"cover", "fraud", "backdoor"}

CLASSICAL_GRAVEYARD: Set[str] = {
    "speech", "ALOI", "Waveform", "optdigits", "vertebral", "smtp",
    "Stamps", "yeast", "SpamBase", "satellite", "PageBlocks",
}

WEAK_ROC_DATASETS: Set[str] = {
    "speech", "ALOI", "Waveform", "optdigits", "vertebral", "yeast",
    "SpamBase", "InternetAds", "landsat", "Pima", "WPBC",
}

WEAK_ROC_EXTRA_RECIPES = [
    "fusion_knn_maha_native", "soft_vote_ensemble_native", "fusion_ecod_knn_native",
    "sod_lite_native", "avg_knn_native", "cblof_lite_native", "lscp_lite_native",
    "inne_lite_native", "pca_residual_knn_native", "pca_residual_knn_pca64",
    "elliptical_native", "fusion_hbos_iforest_native",
    "loda_native", "ocsvm_native", "mcd_native", "pca_recon_native",
    "feature_bagging_native", "rank_ensemble_native", "pca_recon_pca64",
    "icl_lite_pca32", "dte_lite_native", "icl_lite_native",
    "knn_cosine_native", "gmm_native", "kde_native",
    "mahalanobis_native", "abod_lite_native", "goad_native",
    "rank_max_ensemble_native", "gmm_pca32", "gmm_pca64",
]

CV_DATASETS: List[str] = ["CIFAR10", "FashionMNIST", "MNIST-C", "MVTec-AD", "SVHN"]
NLP_DATASETS: List[str] = ["Agnews", "Imdb", "Amazon", "Yelp", "20newsgroups"]

SOFT_SET_98: List[str] = [
    "CIFAR10", "FashionMNIST", "MVTec-AD", "MNIST-C",
    "breastw", "satimage-2", "shuttle", "musk", "http",
]

HARD_TAIL_DATASETS: Set[str] = {
    "speech", "ALOI", "Imdb", "Amazon", "Yelp", "cover", "fraud",
    "backdoor", "glass", "SVHN", "yeast", "WPBC",
}

HONEST_CEILING_BAND = {
    "semi-supervised": {"PR-AUC": [68.0, 75.0], "ROC-AUC": [88.0, 92.0]},
    "unsupervised": {"PR-AUC": [45.0, 55.0], "ROC-AUC": [82.0, 90.0]},
}

SOFT_SET_ROC_TARGET = 98.0
SOFT_SET_PR_TARGET = 95.0

CLASSICAL_HUGE: Set[str] = {
    "cover", "fraud", "census", "celeba", "http", "donors", "skin", "backdoor",
}
CLASSICAL_HUGE_CPU: Set[str] = {"census", "celeba", "http", "donors", "skin"}
CLASSICAL_STRONG: Set[str] = GUARDS | {"thyroid", "WDBC", "Ionosphere", "annthyroid", "mammography"}

PROBE12 = [
    "breastw", "cardio", "glass", "thyroid", "satimage-2", "cover", "fraud",
    "backdoor", "CIFAR10", "FashionMNIST", "Agnews", "Imdb",
]

ALL57: List[str] = []


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    kind: str
    embed: str
    config: Optional[str] = None
    zscore: bool = False
    shallow_method: Optional[str] = None
    axion_profile: str = "alt"
    pca_dim: Optional[int] = None
    # Zoo GMM mixture size when map omits n_components (claim dual_lift default).
    n_components: Optional[int] = None
    note: str = ""


RECIPES: Dict[str, RecipeSpec] = {
    "axion_vit": RecipeSpec("axion_vit", "axion", "vit", "gpu_alt_embeds.yaml", axion_profile="alt"),
    "axion_roberta": RecipeSpec("axion_roberta", "axion", "roberta", "gpu_alt_embeds.yaml", axion_profile="alt"),
    "axion_roberta_spear_platt": RecipeSpec(
        "axion_roberta_spear_platt", "axion", "roberta", "gpu_beat_paper.yaml",
        axion_profile="alt", note="Agnews platt_tail from living gpu.yaml",
    ),
    "axion_edge_rare": RecipeSpec(
        "axion_edge_rare", "axion", "native", "gpu_final.yaml",
        axion_profile="edge_rare", note="Classical huge-n + guard lock; cover/fraud RARE",
    ),
    # Legacy map id used by metrics_sync for donors/skin semi (native AXION, not edge_rare).
    "axion": RecipeSpec(
        "axion", "axion", "native", "gpu_final.yaml",
        axion_profile="alt", note="Alias for frozen dual_lift map recipe=axion",
    ),
    "shallow_lof_vit": RecipeSpec("shallow_lof_vit", "shallow", "vit", shallow_method="lof", zscore=False),
    "shallow_knn_vit": RecipeSpec("shallow_knn_vit", "shallow", "vit", shallow_method="knn", zscore=False),
    "shallow_lof_roberta": RecipeSpec("shallow_lof_roberta", "shallow", "roberta", shallow_method="lof", zscore=False),
    "shallow_lof_e5": RecipeSpec("shallow_lof_e5", "shallow", "e5", shallow_method="lof", zscore=False),
    "shallow_knn_e5": RecipeSpec("shallow_knn_e5", "shallow", "e5", shallow_method="knn", zscore=False),
    "shallow_lof_bert": RecipeSpec("shallow_lof_bert", "shallow", "bert", shallow_method="lof", zscore=False),
    "shallow_iforest_vit": RecipeSpec("shallow_iforest_vit", "shallow", "vit", shallow_method="iforest", zscore=False),
    "shallow_iforest_roberta": RecipeSpec("shallow_iforest_roberta", "shallow", "roberta", shallow_method="iforest", zscore=False),
    "shallow_iforest_e5": RecipeSpec("shallow_iforest_e5", "shallow", "e5", shallow_method="iforest", zscore=False),
    "shallow_iforest": RecipeSpec("shallow_iforest", "shallow", "native", shallow_method="iforest", zscore=True),
    "shallow_lof": RecipeSpec("shallow_lof", "shallow", "native", shallow_method="lof", zscore=True),
    "shallow_knn": RecipeSpec("shallow_knn", "shallow", "native", shallow_method="knn", zscore=True),
    "patchcore_lite_vit": RecipeSpec("patchcore_lite_vit", "patchcore", "vit", note="SVHN/CIFAR PatchCore-lite on ViT"),
    "copod_native": RecipeSpec("copod_native", "copod", "native", zscore=True),
    "ecod_native": RecipeSpec("ecod_native", "ecod", "native", zscore=True),
    "hbos_native": RecipeSpec("hbos_native", "zoo", "native", zscore=True, shallow_method="hbos"),
    "loda_native": RecipeSpec("loda_native", "zoo", "native", zscore=True, shallow_method="loda"),
    "ocsvm_native": RecipeSpec("ocsvm_native", "zoo", "native", zscore=True, shallow_method="ocsvm"),
    "mcd_native": RecipeSpec("mcd_native", "zoo", "native", zscore=True, shallow_method="mcd"),
    "pca_recon_native": RecipeSpec("pca_recon_native", "zoo", "native", zscore=True, shallow_method="pca_recon"),
    "dte_lite_native": RecipeSpec("dte_lite_native", "zoo", "native", zscore=True, shallow_method="dte"),
    "feature_bagging_native": RecipeSpec("feature_bagging_native", "zoo", "native", zscore=True, shallow_method="bagging"),
    "rank_ensemble_native": RecipeSpec("rank_ensemble_native", "zoo", "native", zscore=True, shallow_method="rank_ensemble"),
    "icl_lite_native": RecipeSpec("icl_lite_native", "zoo", "native", zscore=True, shallow_method="icl"),
    "icl_lite_pca32": RecipeSpec("icl_lite_pca32", "zoo", "native", zscore=True, shallow_method="icl", pca_dim=32),
    "pca_recon_pca64": RecipeSpec("pca_recon_pca64", "zoo", "native", zscore=True, shallow_method="pca_recon", pca_dim=64),
    "rank_ensemble_vit": RecipeSpec("rank_ensemble_vit", "zoo", "vit", zscore=True, shallow_method="rank_ensemble"),
    "rank_ensemble_e5": RecipeSpec("rank_ensemble_e5", "zoo", "e5", zscore=True, shallow_method="rank_ensemble"),
    "rank_ensemble_roberta": RecipeSpec("rank_ensemble_roberta", "zoo", "roberta", zscore=True, shallow_method="rank_ensemble"),
    "knn_cosine_native": RecipeSpec("knn_cosine_native", "zoo", "native", zscore=True, shallow_method="knn_cosine"),
    "gmm_native": RecipeSpec("gmm_native", "zoo", "native", zscore=True, shallow_method="gmm"),
    "kde_native": RecipeSpec("kde_native", "zoo", "native", zscore=True, shallow_method="kde"),
    "knn_cosine_vit": RecipeSpec("knn_cosine_vit", "zoo", "vit", zscore=False, shallow_method="knn_cosine"),
    "knn_cosine_e5": RecipeSpec("knn_cosine_e5", "zoo", "e5", zscore=False, shallow_method="knn_cosine"),
    "knn_cosine_roberta": RecipeSpec("knn_cosine_roberta", "zoo", "roberta", zscore=False, shallow_method="knn_cosine"),
    "mahalanobis_native": RecipeSpec("mahalanobis_native", "zoo", "native", zscore=True, shallow_method="mahalanobis"),
    "abod_lite_native": RecipeSpec("abod_lite_native", "zoo", "native", zscore=True, shallow_method="abod_lite"),
    "goad_native": RecipeSpec("goad_native", "zoo", "native", zscore=True, shallow_method="goad"),
    "rank_max_ensemble_native": RecipeSpec("rank_max_ensemble_native", "zoo", "native", zscore=True, shallow_method="rank_max_ensemble"),
    "gmm_pca32": RecipeSpec("gmm_pca32", "zoo", "native", zscore=True, shallow_method="gmm", pca_dim=32, n_components=1),
    "gmm_pca64": RecipeSpec("gmm_pca64", "zoo", "native", zscore=True, shallow_method="gmm", pca_dim=64, n_components=1),
    "gmm_pca64_vit": RecipeSpec("gmm_pca64_vit", "zoo", "vit", zscore=False, shallow_method="gmm", pca_dim=64, n_components=1),
    # Claim NLP GMM slots predominantly used 2 components after PCA64.
    "gmm_pca64_e5": RecipeSpec("gmm_pca64_e5", "zoo", "e5", zscore=False, shallow_method="gmm", pca_dim=64, n_components=2),
    "gmm_pca64_roberta": RecipeSpec("gmm_pca64_roberta", "zoo", "roberta", zscore=False, shallow_method="gmm", pca_dim=64, n_components=1),
    # Leap detectors
    "sod_lite_native": RecipeSpec("sod_lite_native", "zoo", "native", zscore=True, shallow_method="sod_lite"),
    "sod_lite_pca32": RecipeSpec("sod_lite_pca32", "zoo", "native", zscore=True, shallow_method="sod_lite", pca_dim=32),
    "cblof_lite_native": RecipeSpec("cblof_lite_native", "zoo", "native", zscore=True, shallow_method="cblof_lite"),
    "avg_knn_native": RecipeSpec("avg_knn_native", "zoo", "native", zscore=True, shallow_method="avg_knn"),
    "elliptical_native": RecipeSpec("elliptical_native", "zoo", "native", zscore=True, shallow_method="elliptical"),
    "fusion_knn_maha_native": RecipeSpec("fusion_knn_maha_native", "zoo", "native", zscore=True, shallow_method="fusion_knn_maha"),
    "fusion_hbos_iforest_native": RecipeSpec("fusion_hbos_iforest_native", "zoo", "native", zscore=True, shallow_method="fusion_hbos_iforest"),
    "soft_vote_ensemble_native": RecipeSpec("soft_vote_ensemble_native", "zoo", "native", zscore=True, shallow_method="soft_vote_ensemble"),
    "lscp_lite_native": RecipeSpec("lscp_lite_native", "zoo", "native", zscore=True, shallow_method="lscp_lite"),
    "inne_lite_native": RecipeSpec("inne_lite_native", "zoo", "native", zscore=True, shallow_method="inne_lite"),
    "pca_residual_knn_native": RecipeSpec("pca_residual_knn_native", "zoo", "native", zscore=True, shallow_method="pca_residual_knn"),
    "fusion_ecod_knn_native": RecipeSpec("fusion_ecod_knn_native", "zoo", "native", zscore=True, shallow_method="fusion_ecod_knn"),
    "fusion_knn_maha_e5": RecipeSpec("fusion_knn_maha_e5", "zoo", "e5", zscore=False, shallow_method="fusion_knn_maha"),
    "soft_vote_ensemble_e5": RecipeSpec("soft_vote_ensemble_e5", "zoo", "e5", zscore=False, shallow_method="soft_vote_ensemble"),
    "sod_lite_vit": RecipeSpec("sod_lite_vit", "zoo", "vit", zscore=False, shallow_method="sod_lite"),
    "avg_knn_vit": RecipeSpec("avg_knn_vit", "zoo", "vit", zscore=False, shallow_method="avg_knn"),
    "fusion_knn_maha_vit": RecipeSpec("fusion_knn_maha_vit", "zoo", "vit", zscore=False, shallow_method="fusion_knn_maha"),
    # Leap embed / PCA variants
    "pca_residual_knn_vit": RecipeSpec("pca_residual_knn_vit", "zoo", "vit", zscore=False, shallow_method="pca_residual_knn"),
    "pca_residual_knn_e5": RecipeSpec("pca_residual_knn_e5", "zoo", "e5", zscore=False, shallow_method="pca_residual_knn"),
    "pca_residual_knn_roberta": RecipeSpec("pca_residual_knn_roberta", "zoo", "roberta", zscore=False, shallow_method="pca_residual_knn"),
    "pca_residual_knn_pca64": RecipeSpec("pca_residual_knn_pca64", "zoo", "native", zscore=True, shallow_method="pca_residual_knn", pca_dim=64),
    "sod_lite_e5": RecipeSpec("sod_lite_e5", "zoo", "e5", zscore=False, shallow_method="sod_lite"),
    "sod_lite_roberta": RecipeSpec("sod_lite_roberta", "zoo", "roberta", zscore=False, shallow_method="sod_lite"),
    "avg_knn_e5": RecipeSpec("avg_knn_e5", "zoo", "e5", zscore=False, shallow_method="avg_knn"),
    "soft_vote_ensemble_vit": RecipeSpec("soft_vote_ensemble_vit", "zoo", "vit", zscore=False, shallow_method="soft_vote_ensemble"),
    "fusion_ecod_knn_e5": RecipeSpec("fusion_ecod_knn_e5", "zoo", "e5", zscore=False, shallow_method="fusion_ecod_knn"),
    "fusion_ecod_knn_vit": RecipeSpec("fusion_ecod_knn_vit", "zoo", "vit", zscore=False, shallow_method="fusion_ecod_knn"),
    "rank_max_ensemble_e5": RecipeSpec("rank_max_ensemble_e5", "zoo", "e5", zscore=False, shallow_method="rank_max_ensemble"),
    "rank_max_ensemble_vit": RecipeSpec("rank_max_ensemble_vit", "zoo", "vit", zscore=False, shallow_method="rank_max_ensemble"),
}

# SVHN first: PatchCore + knn_vit before weak GMM-vit (heavy-lift).
_CV_SEMI = [
    "patchcore_lite_vit", "shallow_knn_vit", "shallow_lof_vit", "knn_cosine_vit",
    "fusion_knn_maha_vit", "avg_knn_vit", "sod_lite_vit", "pca_residual_knn_vit",
    "soft_vote_ensemble_vit", "rank_max_ensemble_vit", "fusion_ecod_knn_vit",
    "rank_ensemble_vit", "gmm_pca64_vit", "axion_vit",
]
_CV_UNSUP = [
    "patchcore_lite_vit", "shallow_knn_vit", "shallow_lof_vit", "knn_cosine_vit",
    "fusion_knn_maha_vit", "avg_knn_vit", "sod_lite_vit", "pca_residual_knn_vit",
    "soft_vote_ensemble_vit", "rank_max_ensemble_vit", "fusion_ecod_knn_vit",
    "rank_ensemble_vit", "shallow_iforest_vit", "gmm_pca64_vit", "axion_vit",
]
CV_CATALOG_SEMI: Dict[str, List[str]] = {ds: list(_CV_SEMI) for ds in CV_DATASETS}
CV_CATALOG_UNSUP: Dict[str, List[str]] = {ds: list(_CV_UNSUP) for ds in CV_DATASETS}
# Override SVHN: ban promoting gmm first; PatchCore/knn lead.
CV_CATALOG_SEMI["SVHN"] = [
    "patchcore_lite_vit", "shallow_knn_vit", "fusion_knn_maha_vit", "avg_knn_vit",
    "sod_lite_vit", "pca_residual_knn_vit", "soft_vote_ensemble_vit",
    "shallow_lof_vit", "knn_cosine_vit", "rank_max_ensemble_vit", "rank_ensemble_vit",
    "axion_vit", "gmm_pca64_vit",
]
CV_CATALOG_UNSUP["SVHN"] = list(CV_CATALOG_SEMI["SVHN"]) + ["shallow_iforest_vit"]

# NLP: AXION/SPEAR/RoBERTa before GMM-e5 (GMM-e5 collapsed Imdb/Amazon/Yelp/Agnews).
_NLP_SEMI_AG = [
    "axion_roberta_spear_platt", "axion_roberta", "rank_ensemble_roberta",
    "knn_cosine_roberta", "shallow_lof_roberta", "gmm_pca64_roberta",
    "pca_residual_knn_e5", "sod_lite_e5", "rank_max_ensemble_e5",
    "fusion_knn_maha_e5", "soft_vote_ensemble_e5", "fusion_ecod_knn_e5",
    "rank_ensemble_e5", "knn_cosine_e5", "avg_knn_e5", "shallow_lof_e5",
    "shallow_knn_e5", "gmm_pca64_e5",
]
_NLP_SEMI = [
    "axion_roberta", "rank_ensemble_roberta", "knn_cosine_roberta",
    "shallow_lof_roberta", "gmm_pca64_roberta", "pca_residual_knn_e5",
    "sod_lite_e5", "rank_max_ensemble_e5", "fusion_knn_maha_e5",
    "soft_vote_ensemble_e5", "fusion_ecod_knn_e5", "rank_ensemble_e5",
    "knn_cosine_e5", "avg_knn_e5", "shallow_lof_e5", "shallow_knn_e5",
    "gmm_pca64_e5",
]
_NLP_SEMI_20 = _NLP_SEMI + ["shallow_lof_bert"]
_NLP_UNSUP = [
    "pca_residual_knn_e5", "sod_lite_e5", "rank_max_ensemble_e5",
    "fusion_knn_maha_e5", "soft_vote_ensemble_e5", "fusion_ecod_knn_e5",
    "avg_knn_e5", "rank_ensemble_roberta", "knn_cosine_roberta",
    "gmm_pca64_roberta", "rank_ensemble_e5", "knn_cosine_e5",
    "shallow_lof_e5", "shallow_knn_e5", "shallow_iforest_e5", "gmm_pca64_e5",
    "axion_roberta",
]

NLP_CATALOG_SEMI: Dict[str, List[str]] = {
    "Agnews": list(_NLP_SEMI_AG),
    "Imdb": list(_NLP_SEMI),
    "Amazon": list(_NLP_SEMI),
    "Yelp": list(_NLP_SEMI),
    "20newsgroups": list(_NLP_SEMI_20),
}
NLP_CATALOG_UNSUP: Dict[str, List[str]] = {ds: list(_NLP_UNSUP) for ds in NLP_DATASETS}

SEMI_GRAVEYARD_RECIPES = [
    "fusion_knn_maha_native", "soft_vote_ensemble_native", "fusion_ecod_knn_native",
    "sod_lite_native", "avg_knn_native", "cblof_lite_native", "lscp_lite_native",
    "inne_lite_native", "pca_residual_knn_native", "pca_residual_knn_pca64",
    "elliptical_native", "fusion_hbos_iforest_native",
    "icl_lite_native", "icl_lite_pca32", "abod_lite_native", "mahalanobis_native",
    "knn_cosine_native", "gmm_native", "gmm_pca64", "gmm_pca32",
    "rank_max_ensemble_native", "rank_ensemble_native", "goad_native",
    "dte_lite_native", "feature_bagging_native", "ocsvm_native", "mcd_native",
    "pca_recon_native", "pca_recon_pca64", "hbos_native", "loda_native",
    "shallow_knn", "shallow_lof", "copod_native", "kde_native",
]
UNSUP_GRAVEYARD_RECIPES = [
    "fusion_knn_maha_native", "soft_vote_ensemble_native", "fusion_ecod_knn_native",
    "sod_lite_native", "avg_knn_native", "cblof_lite_native", "lscp_lite_native",
    "inne_lite_native", "pca_residual_knn_native", "pca_residual_knn_pca64",
    "elliptical_native", "fusion_hbos_iforest_native",
    "icl_lite_native", "icl_lite_pca32", "abod_lite_native", "mahalanobis_native",
    "knn_cosine_native", "gmm_native", "gmm_pca64", "gmm_pca32",
    "rank_max_ensemble_native", "rank_ensemble_native", "goad_native",
    "feature_bagging_native", "ocsvm_native", "mcd_native", "pca_recon_native",
    "pca_recon_pca64", "hbos_native", "loda_native", "dte_lite_native",
    "copod_native", "ecod_native", "shallow_iforest", "shallow_lof", "kde_native",
]
SEMI_DEFAULT_RECIPES = [
    "fusion_knn_maha_native", "soft_vote_ensemble_native", "fusion_ecod_knn_native",
    "sod_lite_native", "avg_knn_native", "cblof_lite_native", "elliptical_native",
    "shallow_lof", "shallow_knn", "knn_cosine_native", "gmm_native",
    "mahalanobis_native", "goad_native", "rank_max_ensemble_native",
    "rank_ensemble_native", "hbos_native", "dte_lite_native", "loda_native",
    "ocsvm_native", "feature_bagging_native", "axion_edge_rare",
]
UNSUP_DEFAULT_RECIPES = [
    "fusion_knn_maha_native", "soft_vote_ensemble_native", "fusion_ecod_knn_native",
    "sod_lite_native", "avg_knn_native", "cblof_lite_native", "inne_lite_native",
    "elliptical_native", "knn_cosine_native", "gmm_native", "mahalanobis_native",
    "goad_native", "rank_max_ensemble_native", "rank_ensemble_native", "hbos_native",
    "loda_native", "ocsvm_native", "feature_bagging_native", "gmm_pca64",
    "shallow_iforest", "copod_native", "ecod_native", "axion_edge_rare",
]
HUGE_CPU_UNSUP_RECIPES = [
    "fusion_hbos_iforest_native", "hbos_native", "avg_knn_native", "inne_lite_native",
    "rank_ensemble_native", "shallow_iforest", "fusion_ecod_knn_native",
    "soft_vote_ensemble_native", "copod_native", "ecod_native", "axion_edge_rare",
]
HUGE_CPU_SEMI_RECIPES = ["axion_edge_rare", "shallow_iforest", "copod_native", "rank_ensemble_native"]
CLASSICAL_STRONG_RECIPES = ["axion_edge_rare"]
CLASSICAL_LOCKED_RECIPES = ["axion_edge_rare"]

CV_CATALOG = CV_CATALOG_SEMI
NLP_CATALOG = NLP_CATALOG_SEMI


def tier_for(dataset: str) -> str:
    if dataset in AXION_LOCKED:
        return "axion_locked"
    if dataset in GUARDS or dataset in CLASSICAL_STRONG:
        return "strong_buffer"
    if dataset in CLASSICAL_HUGE_CPU:
        return "huge_cpu"
    if dataset in CLASSICAL_GRAVEYARD:
        return "graveyard"
    if dataset in CV_CATALOG_SEMI or dataset in CV_CATALOG_UNSUP:
        return "cv"
    if dataset in NLP_CATALOG_SEMI or dataset in NLP_CATALOG_UNSUP:
        return "nlp"
    return "classical_mid"


def candidates_for(dataset: str, setting: str = "semi-supervised") -> List[str]:
    unsup = setting == "unsupervised"
    if dataset in CV_CATALOG_SEMI or dataset in CV_CATALOG_UNSUP:
        src = CV_CATALOG_UNSUP if unsup else CV_CATALOG_SEMI
        return list(src[dataset])
    if dataset in NLP_CATALOG_SEMI or dataset in NLP_CATALOG_UNSUP:
        src = NLP_CATALOG_UNSUP if unsup else NLP_CATALOG_SEMI
        return list(src[dataset])
    t = tier_for(dataset)
    if t in ("strong_buffer", "axion_locked"):
        return list(CLASSICAL_LOCKED_RECIPES)
    if t == "huge_cpu":
        out = list(HUGE_CPU_UNSUP_RECIPES if unsup else HUGE_CPU_SEMI_RECIPES)
    elif t == "graveyard":
        out = list(UNSUP_GRAVEYARD_RECIPES if unsup else SEMI_GRAVEYARD_RECIPES)
    else:
        out = list(UNSUP_DEFAULT_RECIPES if unsup else SEMI_DEFAULT_RECIPES)
    if dataset in ICL_DATASETS:
        for extra in ("icl_lite_native", "icl_lite_pca32", "pca_recon_pca64"):
            if extra not in out:
                out.append(extra)
    if dataset in WEAK_ROC_DATASETS:
        for extra in WEAK_ROC_EXTRA_RECIPES:
            if extra not in out:
                out.append(extra)
    return out


def classical_cpu_datasets(atlas_specs: Optional[Sequence[Any]] = None) -> List[str]:
    if atlas_specs is None:
        from axion.data.registry import build_beat_paper_registry
        atlas_specs = build_beat_paper_registry(atlas_csv=ROOT / "data" / "atlas_57_beat_paper.csv")
    names = [s.name for s in atlas_specs if s.modality == "classical"]
    return [n for n in names if n not in AXION_LOCKED]


def local_campaign_datasets(atlas_specs: Optional[Sequence[Any]] = None) -> List[str]:
    classical = classical_cpu_datasets(atlas_specs)
    seen = set(classical)
    out = list(classical)
    for name in NLP_DATASETS + CV_DATASETS:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def load_axion_final_locks(path: Optional[Path] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    p = Path(path) if path else AXION_FINAL_COMPARE
    out: Dict[str, Dict[str, Dict[str, float]]] = {"semi-supervised": {}, "unsupervised": {}}
    if not p.exists():
        return out
    data = json.loads(p.read_text(encoding="utf-8"))
    per = data.get("per_dataset") or {}
    for setting, rows in per.items():
        bucket = out.setdefault(str(setting), {})
        if not isinstance(rows, list):
            continue
        for row in rows:
            ds = row.get("dataset")
            if not ds:
                continue
            bucket[str(ds)] = {
                "PR-AUC": float(row.get("PR-AUC", float("nan"))),
                "ROC-AUC": float(row.get("ROC-AUC", float("nan"))),
            }
    return out


def beats_lock(
    test_pr: float,
    test_roc: float,
    lock: Optional[Mapping[str, float]],
    *,
    pr_delta: float = PROMOTE_DELTA,
    roc_floor: float = ROC_FLOOR_DELTA,
) -> bool:
    if not lock:
        return False
    if not (math.isfinite(float(test_pr)) and math.isfinite(float(test_roc))):
        return False
    return float(test_pr) >= float(lock["PR-AUC"]) + pr_delta and float(test_roc) >= float(
        lock["ROC-AUC"]
    ) + roc_floor


def _resolve_cfg_path(raw: Any, *, default: Path) -> Path:
    p = Path(raw) if raw else default
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def _pool_has_npz(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file() and path.suffix == ".npz":
        return True
    return any(path.glob("*.npz"))


def embed_folders(cfg: Mapping[str, Any]) -> Dict[str, Path]:
    paths = cfg.get("paths", {})
    adbench = _resolve_cfg_path(paths.get("adbench_root"), default=ROOT.parent.parent / "ADBench" / "adbench" / "datasets")
    alt = _resolve_cfg_path(paths.get("embeds_alt_root"), default=ROOT / "data" / "embeds_alt")
    whitened = _resolve_cfg_path(paths.get("embeds_alt_whitened_root"), default=ROOT / "data" / "embeds_alt_whitened")
    beat = cfg.get("beat_paper", {})
    e5_rel = str(beat.get("nlp_e5_folder", "NLP_by_E5_large/pool=mean"))
    cv_rel = str(beat.get("cv_folder", "CV_by_ViT"))
    return {
        "adbench": adbench,
        "alt": alt,
        "whitened": whitened,
        "cv_vit": adbench / cv_rel,
        "cv_vit_whitened": whitened / cv_rel,
        "nlp_roberta": adbench / str(beat.get("nlp_roberta_folder", "NLP_by_RoBERTa")),
        "nlp_bert": adbench / str(beat.get("nlp_bert_folder", "NLP_by_BERT")),
        "nlp_e5": alt / e5_rel,
        "nlp_e5_whitened": whitened / e5_rel,
        "classical": adbench / "Classical",
    }


def rewrite_relpaths(rel_paths: Sequence[str], embed_folder: str) -> List[str]:
    folder = str(embed_folder).rstrip("/")
    return [f"{folder}/{Path(rel).name}" for rel in rel_paths]


def relpaths_for_recipe(spec: Any, recipe_id: str, cfg: Mapping[str, Any]) -> List[str]:
    rec = RECIPES[recipe_id]
    if rec.embed == "native":
        return list(spec.relative_paths)
    beat = cfg.get("beat_paper", {})
    folder = {
        "vit": str(beat.get("cv_folder", "CV_by_ViT")),
        "roberta": str(beat.get("nlp_roberta_folder", "NLP_by_RoBERTa")),
        "bert": str(beat.get("nlp_bert_folder", "NLP_by_BERT")),
        "e5": str(beat.get("nlp_e5_folder", "NLP_by_E5_large/pool=mean")),
    }[rec.embed]
    return rewrite_relpaths(spec.relative_paths, folder)


def resolve_recipe_npz_paths(spec: Any, recipe_id: str, cfg: Mapping[str, Any]) -> List[Path]:
    folders = embed_folders(cfg)
    rels = relpaths_for_recipe(spec, recipe_id, cfg)
    embed_root = embed_root_for_recipe(recipe_id, cfg)
    roots = [embed_root, folders["alt"], folders["adbench"]]
    found: List[Path] = []
    seen = set()
    for rel in rels:
        hit: Optional[Path] = None
        for root in roots:
            cand = root / rel if root in (folders["adbench"], folders["alt"]) else root / Path(rel).name
            if cand.exists():
                hit = cand
                break
            by_name = root / Path(rel).name
            if by_name.exists():
                hit = by_name
                break
        if hit is None:
            continue
        rp = hit.resolve()
        if rp not in seen:
            seen.add(rp)
            found.append(hit)
    return found


def embed_root_for_recipe(recipe_id: str, cfg: Mapping[str, Any]) -> Path:
    spec = RECIPES[recipe_id]
    folders = embed_folders(cfg)
    use_whiten = spec.kind in ("shallow", "zoo")
    if spec.embed == "native":
        return folders["classical"]
    if spec.embed == "vit":
        if use_whiten and _pool_has_npz(folders["cv_vit_whitened"]):
            return folders["cv_vit_whitened"]
        vit = folders["cv_vit"]
        return vit if vit.exists() else folders["alt"] / "CV_by_ViT"
    if spec.embed == "roberta":
        return folders["nlp_roberta"]
    if spec.embed == "bert":
        return folders["nlp_bert"]
    if spec.embed == "e5":
        if use_whiten and _pool_has_npz(folders["nlp_e5_whitened"]):
            return folders["nlp_e5_whitened"]
        e5 = folders["nlp_e5"]
        if _pool_has_npz(e5):
            return e5
        return folders["nlp_roberta"]
    raise KeyError(f"Unknown embed {spec.embed} for {recipe_id}")


def axion_config_path(recipe_id: str, cfg: Mapping[str, Any]) -> Path:
    spec = RECIPES[recipe_id]
    if spec.config:
        p = ROOT / "configs" / spec.config
        if p.exists():
            return p
    return ROOT / "configs" / str(cfg.get("beat_paper", {}).get("axion_config", "gpu_beat_paper.yaml"))


def default_recipe_map() -> Dict[str, Any]:
    from axion.data.registry import build_beat_paper_registry
    atlas = ROOT / "data" / "atlas_57_beat_paper.csv"
    out: Dict[str, Any] = {}
    for spec in build_beat_paper_registry(atlas_csv=atlas):
        semi = candidates_for(spec.name, "semi-supervised")
        unsup = candidates_for(spec.name, "unsupervised")
        out[spec.name] = {
            "semi-supervised": {"recipe": semi[0], "source": "catalog_default"},
            "unsupervised": {"recipe": unsup[0], "source": "catalog_default"},
        }
    for ds in AXION_LOCKED:
        out[ds] = {
            "semi-supervised": {"recipe": "axion_edge_rare", "source": "axion_locked"},
            "unsupervised": {"recipe": "axion_edge_rare", "source": "axion_locked"},
        }
    return out


def assert_catalog() -> None:
    from axion.data.registry import build_beat_paper_registry
    atlas = ROOT / "data" / "atlas_57_beat_paper.csv"
    specs = build_beat_paper_registry(atlas_csv=atlas)
    global ALL57
    ALL57 = [s.name for s in specs]
    if len(specs) != 57:
        raise AssertionError(f"Expected 57 datasets, got {len(specs)}")
    missing_recipes: List[str] = []
    missing_cands: List[str] = []
    for spec in specs:
        for setting in ("semi-supervised", "unsupervised"):
            cands = candidates_for(spec.name, setting)
            if not cands:
                missing_cands.append(f"{spec.name}/{setting}")
            for rid in cands:
                if rid not in RECIPES:
                    missing_recipes.append(f"{spec.name}/{setting}:{rid}")
    if missing_recipes:
        raise AssertionError(f"Unknown recipes: {missing_recipes[:5]}")
    if missing_cands:
        raise AssertionError(f"No candidates: {missing_cands}")
    if candidates_for("cover", "semi-supervised") != ["axion_edge_rare"]:
        raise AssertionError("cover semi must be axion_edge_rare only")
    if candidates_for("cover", "unsupervised") != ["axion_edge_rare"]:
        raise AssertionError("cover unsup must be axion_edge_rare only")
    print(f"BEAT_PAPER catalog OK: 57 datasets, {len(RECIPES)} recipes")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assert-catalog", action="store_true", help="Validate 57-row catalog")
    p.add_argument("--list", metavar="DATASET", help="Show candidates for one dataset")
    p.add_argument("--write-default-map", action="store_true")
    p.add_argument("--freeze-57-map", action="store_true", help="Merge local CPU winners onto catalog defaults; write recipe_map_57_beat.json")
    args = p.parse_args()
    if args.assert_catalog:
        assert_catalog()
        return
    if args.list:
        for setting in ("semi-supervised", "unsupervised"):
            print(f"[{setting}]")
            for rid in candidates_for(args.list, setting):
                print(rid, RECIPES[rid])
        return
    if args.write_default_map or args.freeze_57_map:
        THESIS_DIR.mkdir(parents=True, exist_ok=True)
        (THESIS_DIR / "thesis").mkdir(parents=True, exist_ok=True)
        m = default_recipe_map()
        if args.freeze_57_map:
            local_map = ROOT / "results" / "axion_beat_paper_local" / "thesis" / "recipe_map.json"
            if local_map.exists():
                local = json.loads(local_map.read_text(encoding="utf-8"))
                for ds, settings in local.items():
                    if isinstance(settings, dict):
                        m.setdefault(ds, {}).update(settings)
        RECIPE_MAP_PATH.write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"Wrote {RECIPE_MAP_PATH} ({len(m)} datasets)")
        return
    p.print_help()


if __name__ == "__main__":
    main()

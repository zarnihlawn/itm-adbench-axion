"""AXION — Adaptive cross-feature Interaction Observation Network.

Components: MCB + FX-Enc + HPD + LATCH + MCS (+ SCALE) + HEDGE.

Phase 5: train-only PCA whitening for raw d≥400; AXION runs in PCA space.
Classical/low-d semi stays MCS-primary; high-d (post-PCA) semi restores mild LATCH.
HEDGE: fuse train-anchored PCA residual energy into high-d scores (CV/NLP).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from axion.models.axion_net import AxionNet, gaussian_nll
from axion.models.extensions.copod_view import CopodView
from axion.models.extensions.cv_knn import CvKnnView
from axion.models.extensions.nlp_lof import NlpLofView
from axion.models.extensions.ecod_view import EcodView
from axion.models.extensions.edge import EdgeExtension
from axion.models.extensions.goad_lite import GoadLiteExtension
from axion.models.extensions.nest import NestExtension
from axion.models.extensions.rare import RareExtension
from axion.models.extensions.spear import SpearExtension, views_from_parts
from axion.models.mcb import sample_masks, scale_hparams
from axion.models.profiles import CategoryProfile, resolve_profile
from axion.util.hardware import build_profile
from axion.util.progress import ProgressBar, progress, stage, step_log, verbose_enabled


class AxionModel:
    """Masked heteroscedastic recon + latent Mahalanobis anomaly scorer."""

    name = "axion"

    def __init__(
        self,
        hidden: Optional[int] = None,
        latent: Optional[int] = None,
        depth: Optional[int] = None,
        mask_rates: Optional[Sequence[float]] = None,
        score_k: Optional[int] = None,
        dropout: Optional[float] = None,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 20,
        val_fraction: float = 0.15,
        latch_alpha: float = 0.4,
        latch_alpha_semi: Optional[float] = 0.25,
        latch_alpha_semi_tiny: Optional[float] = 0.35,
        latch_alpha_semi_huge: Optional[float] = 0.20,
        latch_alpha_semi_huge_mid: Optional[float] = 0.35,
        latch_alpha_semi_high_d: Optional[float] = 0.45,
        latch_alpha_unsup_huge: float = 0.70,
        latch_alpha_unsup_large_mid: float = 0.75,
        latch_alpha_unsup_tiny: float = 0.75,
        mae_weight: float = 0.6,
        nll_weight: float = 0.4,
        mae_weight_semi: Optional[float] = 0.9,
        nll_weight_semi: Optional[float] = 0.1,
        mae_weight_unsup_huge: float = 0.85,
        nll_weight_unsup_huge: float = 0.15,
        semi_epoch_boost: float = 1.75,
        tiny_epoch_boost: float = 2.0,
        pca_dim: int = 128,
        pca_threshold: int = 400,
        pca_variance: float = 0.99,
        pca_k_min: int = 32,
        hedge_alpha: float = 0.45,
        hedge_alpha_semi_high_d: float = 0.75,
        hedge_alpha_nlp: float = 0.35,
        hedge_alpha_semi_nlp: float = 0.0,
        hedge_alpha_semi_nlp_multi: float = 0.25,
        robust_latch_unsup_huge: bool = False,
        x_latch_alpha_unsup_huge: float = 0.60,
        x_latch_max_d: int = 16,
        contam_trim_frac: float = 0.0,
        contam_trim_warmup: int = 15,
        contam_trim_every: int = 5,
        latch_alpha_unsup_low_d: float = 0.85,
        nlp_raw_d_min: int = 700,
        # Semi PR attack score views / fusion
        vis_alpha_semi: float = 0.15,
        disagree_alpha_semi: float = 0.10,
        hpd_alpha_semi: float = 0.05,
        hpd_alpha_semi_huge: float = 0.0,
        c_hedge_alpha_semi: float = 0.20,
        c_hedge_alpha_semi_huge: float = 0.35,
        vis_alpha_semi_huge: float = 0.25,
        disagree_alpha_semi_huge: float = 0.18,
        bank_weight_semi: bool = True,
        learn_bank_weights: bool = True,
        calix_semi: bool = True,
        # Phase 2 training
        contrastive_weight: float = 0.05,
        latch_shrinkage: bool = True,
        semi_loss_curriculum: bool = True,
        block_prob: float = 0.35,
        # Phase 3
        use_interaction: bool = True,
        interaction_rank: int = 16,
        ensemble_heads: int = 1,
        true_ensemble: bool = False,
        # SPEAR extension (semi PR ranking; default off = bit-identical core)
        spear: bool = False,
        spear_gamma: float = 0.35,
        spear_gamma_nlp: Optional[float] = None,
        spear_pairs: int = 256,
        spear_epochs: int = 40,
        spear_lr: float = 1e-2,
        spear_hidden: int = 0,
        spear_classical_only: bool = True,
        spear_soft_ap: bool = False,
        spear_skip_huge_n: bool = True,
        spear_huge_n_threshold: int = 10000,
        spear_skip_tiny_n: bool = True,
        spear_tiny_n_threshold: int = 200,
        spear_allow_datasets: Optional[Sequence[str]] = None,
        spear_rank_fuse: bool = False,
        spear_fuse: str = "add",
        spear_fuse_nlp: Optional[str] = None,
        spear_tail_q: float = 0.75,
        spear_tail_soft: float = 0.35,
        # RARE: huge-n classical soft-AP (cover/fraud)
        rare: bool = False,
        rare_gamma: float = 0.20,
        rare_pairs: int = 256,
        rare_epochs: int = 40,
        rare_lr: float = 1e-2,
        rare_hidden: int = 0,
        rare_huge_n_threshold: int = 10000,
        rare_max_fit_n: int = 8000,
        rare_skip_cover: bool = True,
        rare_allow_datasets: Optional[Sequence[str]] = None,
        rare_gamma_cover: float = 0.12,
        # EDGE: embed density gradient (CIFAR/Agnews; Imdb off)
        edge: bool = False,
        edge_gamma_cv: float = 0.15,
        edge_gamma_nlp: float = 0.10,
        edge_pairs: int = 256,
        edge_epochs: int = 30,
        edge_lr: float = 1e-2,
        edge_hidden: int = 0,
        edge_skip_imdb: bool = True,
        edge_max_fit_n: int = 6000,
        edge_whiten: bool = False,
        edge_whiten_dim: int = 64,
        edge_knn_view: bool = False,
        edge_knn_k: int = 5,
        edge_advanced_allow_datasets: Optional[Sequence[str]] = None,
        edge_skip_fashion_advanced: bool = True,
        # NEST: neighbor embed soft-tail (CIFAR/Agnews; Imdb off)
        nest: bool = False,
        nest_gamma_cv: float = 0.10,
        nest_gamma_nlp: float = 0.08,
        nest_pairs: int = 256,
        nest_epochs: int = 30,
        nest_lr: float = 1e-2,
        nest_hidden: int = 0,
        nest_knn_k: int = 20,
        nest_skip_imdb: bool = True,
        nest_max_fit_n: int = 6000,
        # CV PatchCore-lite kNN fuse (fair ResNet embeds; Fashion/CIFAR stretch)
        cv_knn: bool = False,
        cv_knn_alpha_semi: float = 0.35,
        cv_knn_memory_cap: int = 8000,
        cv_knn_neighbors: int = 5,
        cv_knn_skip_fashion: bool = True,
        cv_knn_allow_datasets: Optional[Sequence[str]] = None,
        # CV LOF replace on fair ResNet (CIFAR allowlist). Not cv_knn / PatchCore.
        cv_lof: bool = False,
        cv_lof_alpha_semi: float = 1.0,
        cv_lof_method: str = "lof",
        cv_lof_mode: str = "replace",
        cv_lof_neighbors: int = 20,
        cv_lof_memory_cap: int = 6000,
        cv_lof_whiten: bool = False,
        cv_lof_whiten_dim: int = 64,
        cv_lof_allow_datasets: Optional[Sequence[str]] = None,
        # NLP LOF/kNN fuse on fair BERT (Imdb allowlist; not cv_knn; not e5)
        nlp_lof: bool = False,
        nlp_lof_alpha_semi: float = 0.35,
        nlp_lof_method: str = "lof",
        nlp_lof_mode: str = "fuse",
        nlp_lof_fuse: str = "add",
        nlp_lof_tail_q: float = 0.75,
        nlp_lof_tail_soft: float = 0.35,
        nlp_lof_neighbors: int = 20,
        nlp_lof_memory_cap: int = 6000,
        nlp_lof_whiten: bool = False,
        nlp_lof_whiten_dim: int = 64,
        nlp_lof_allow_datasets: Optional[Sequence[str]] = None,
        # Category router + market-inspired classical views
        profile_router: bool = False,
        research_embeds: bool = False,
        ecod: bool = False,
        ecod_alpha: float = 0.12,
        copod: bool = False,
        copod_alpha: float = 0.18,
        goad: bool = False,
        goad_alpha: float = 0.10,
        goad_transforms: int = 4,
        goad_epochs: int = 20,
        # Throughput
        score_batch_size: int = 0,
        score_batch_size_oom_fallback: int = 2048,
        dataloader_num_workers: int = -1,
        tiny_n_safe_hw: bool = False,
        tiny_n_safe_threshold: int = 250,
        tiny_n_score_batch_size: int = 512,
        tiny_n_dataloader_num_workers: int = 0,
        # Mid-n claim/smoke: force g5 MCS HW (512/0) between tiny and huge thresholds
        mid_n_safe_hw: bool = False,
        mid_n_score_batch_size: int = 512,
        mid_n_dataloader_num_workers: int = 0,
        # Huge classical (cover/fraud): max-HW workers/score_batch regresses PR
        huge_n_safe_hw: bool = False,
        huge_n_safe_threshold: int = 10000,
        huge_n_score_batch_size: int = 512,
        huge_n_dataloader_num_workers: int = 0,
        mcs_pack_masks: bool = False,
        vectorized_masks: bool = False,
        dataset_hint: Optional[str] = None,
        device: Optional[str] = None,
        seed: int = 111,
        max_train_samples: int = 0,
    ):
        self.hidden = hidden
        self.latent = latent
        self.depth = depth
        self.mask_rates = tuple(mask_rates) if mask_rates is not None else None
        self.score_k = score_k
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.latch_alpha = latch_alpha
        self.latch_alpha_semi = latch_alpha_semi
        self.latch_alpha_semi_tiny = latch_alpha_semi_tiny
        self.latch_alpha_semi_huge = latch_alpha_semi_huge
        self.latch_alpha_semi_huge_mid = latch_alpha_semi_huge_mid
        self.latch_alpha_semi_high_d = latch_alpha_semi_high_d
        self.latch_alpha_unsup_huge = float(latch_alpha_unsup_huge)
        self.latch_alpha_unsup_large_mid = float(latch_alpha_unsup_large_mid)
        self.latch_alpha_unsup_tiny = float(latch_alpha_unsup_tiny)
        self.mae_weight = mae_weight
        self.nll_weight = nll_weight
        self.mae_weight_semi = mae_weight_semi
        self.nll_weight_semi = nll_weight_semi
        self.mae_weight_unsup_huge = float(mae_weight_unsup_huge)
        self.nll_weight_unsup_huge = float(nll_weight_unsup_huge)
        self.semi_epoch_boost = semi_epoch_boost
        self.tiny_epoch_boost = float(tiny_epoch_boost)
        self.pca_dim = int(pca_dim)
        self.pca_threshold = int(pca_threshold)
        self.pca_variance = float(pca_variance)
        self.pca_k_min = int(pca_k_min)
        self.hedge_alpha = float(hedge_alpha)
        self.hedge_alpha_semi_high_d = float(hedge_alpha_semi_high_d)
        self.hedge_alpha_nlp = float(hedge_alpha_nlp)
        self.hedge_alpha_semi_nlp = float(hedge_alpha_semi_nlp)
        self.hedge_alpha_semi_nlp_multi = float(hedge_alpha_semi_nlp_multi)
        self.robust_latch_unsup_huge = bool(robust_latch_unsup_huge)
        self.x_latch_alpha_unsup_huge = float(x_latch_alpha_unsup_huge)
        self.x_latch_max_d = int(x_latch_max_d)
        self.contam_trim_frac = float(contam_trim_frac)
        self.contam_trim_warmup = int(contam_trim_warmup)
        self.contam_trim_every = int(max(1, contam_trim_every))
        self.latch_alpha_unsup_low_d = float(latch_alpha_unsup_low_d)
        self.nlp_raw_d_min = int(nlp_raw_d_min)
        self.vis_alpha_semi = float(vis_alpha_semi)
        self.disagree_alpha_semi = float(disagree_alpha_semi)
        self.hpd_alpha_semi = float(hpd_alpha_semi)
        self.hpd_alpha_semi_huge = float(hpd_alpha_semi_huge)
        self.c_hedge_alpha_semi = float(c_hedge_alpha_semi)
        self.c_hedge_alpha_semi_huge = float(c_hedge_alpha_semi_huge)
        self.vis_alpha_semi_huge = float(vis_alpha_semi_huge)
        self.disagree_alpha_semi_huge = float(disagree_alpha_semi_huge)
        self.bank_weight_semi = bool(bank_weight_semi)
        self.learn_bank_weights = bool(learn_bank_weights)
        self.calix_semi = bool(calix_semi)
        self.contrastive_weight = float(contrastive_weight)
        self.latch_shrinkage = bool(latch_shrinkage)
        self.semi_loss_curriculum = bool(semi_loss_curriculum)
        self.block_prob = float(np.clip(block_prob, 0.0, 1.0))
        self.use_interaction = bool(use_interaction)
        self.interaction_rank = int(interaction_rank)
        self.ensemble_heads = int(max(1, ensemble_heads))
        self.true_ensemble = bool(true_ensemble)
        self.spear = bool(spear)
        self.spear_gamma = float(spear_gamma)
        self.spear_gamma_nlp = (
            None if spear_gamma_nlp is None else float(spear_gamma_nlp)
        )
        self.spear_pairs = int(spear_pairs)
        self.spear_epochs = int(spear_epochs)
        self.spear_lr = float(spear_lr)
        self.spear_hidden = int(spear_hidden)
        self.spear_classical_only = bool(spear_classical_only)
        self.spear_soft_ap = bool(spear_soft_ap)
        self.spear_skip_huge_n = bool(spear_skip_huge_n)
        self.spear_huge_n_threshold = int(spear_huge_n_threshold)
        self.spear_skip_tiny_n = bool(spear_skip_tiny_n)
        self.spear_tiny_n_threshold = int(spear_tiny_n_threshold)
        self.spear_allow_datasets = [
            str(x).strip().lower()
            for x in (spear_allow_datasets or ())
            if str(x).strip()
        ]
        self.spear_rank_fuse = bool(spear_rank_fuse)
        self.spear_fuse = str(spear_fuse or "add").strip().lower()
        self.spear_fuse_nlp = (
            None if spear_fuse_nlp is None else str(spear_fuse_nlp).strip().lower()
        )
        self.spear_tail_q = float(spear_tail_q)
        self.spear_tail_soft = float(spear_tail_soft)
        self.rare = bool(rare)
        self.rare_gamma = float(rare_gamma)
        self.rare_pairs = int(rare_pairs)
        self.rare_epochs = int(rare_epochs)
        self.rare_lr = float(rare_lr)
        self.rare_hidden = int(rare_hidden)
        self.rare_huge_n_threshold = int(rare_huge_n_threshold)
        self.rare_max_fit_n = int(rare_max_fit_n)
        self.rare_skip_cover = bool(rare_skip_cover)
        self.rare_gamma_cover = float(rare_gamma_cover)
        if rare_allow_datasets is None:
            self.rare_allow_datasets: tuple = ()
        else:
            self.rare_allow_datasets = tuple(
                str(x).strip().lower()
                for x in rare_allow_datasets
                if str(x).strip()
            )
        self.edge = bool(edge)
        self.edge_gamma_cv = float(edge_gamma_cv)
        self.edge_gamma_nlp = float(edge_gamma_nlp)
        self.edge_pairs = int(edge_pairs)
        self.edge_epochs = int(edge_epochs)
        self.edge_lr = float(edge_lr)
        self.edge_hidden = int(edge_hidden)
        self.edge_skip_imdb = bool(edge_skip_imdb)
        self.edge_max_fit_n = int(edge_max_fit_n)
        self.edge_whiten = bool(edge_whiten)
        self.edge_whiten_dim = int(edge_whiten_dim)
        self.edge_knn_view = bool(edge_knn_view)
        self.edge_knn_k = int(edge_knn_k)
        if edge_advanced_allow_datasets is None:
            self.edge_advanced_allow_datasets = ("cifar10",) if (
                bool(edge_whiten) or bool(edge_knn_view)
            ) else ()
        else:
            self.edge_advanced_allow_datasets = tuple(
                str(x).strip().lower()
                for x in edge_advanced_allow_datasets
                if str(x).strip()
            )
        self.edge_skip_fashion_advanced = bool(edge_skip_fashion_advanced)
        self.nest = bool(nest)
        self.nest_gamma_cv = float(nest_gamma_cv)
        self.nest_gamma_nlp = float(nest_gamma_nlp)
        self.nest_pairs = int(nest_pairs)
        self.nest_epochs = int(nest_epochs)
        self.nest_lr = float(nest_lr)
        self.nest_hidden = int(nest_hidden)
        self.nest_knn_k = int(nest_knn_k)
        self.nest_skip_imdb = bool(nest_skip_imdb)
        self.nest_max_fit_n = int(nest_max_fit_n)
        self.cv_knn = bool(cv_knn)
        self.cv_knn_alpha_semi = float(cv_knn_alpha_semi)
        self.cv_knn_memory_cap = int(cv_knn_memory_cap)
        self.cv_knn_neighbors = int(cv_knn_neighbors)
        self.cv_knn_skip_fashion = bool(cv_knn_skip_fashion)
        self.cv_knn_allow_datasets = [
            str(x).strip().lower()
            for x in (cv_knn_allow_datasets or ())
            if str(x).strip()
        ]
        self.cv_lof = bool(cv_lof)
        self.cv_lof_alpha_semi = float(cv_lof_alpha_semi)
        self.cv_lof_method = str(cv_lof_method or "lof").strip().lower()
        self.cv_lof_mode = str(cv_lof_mode or "replace").strip().lower()
        self.cv_lof_neighbors = int(cv_lof_neighbors)
        self.cv_lof_memory_cap = int(cv_lof_memory_cap)
        self.cv_lof_whiten = bool(cv_lof_whiten)
        self.cv_lof_whiten_dim = int(cv_lof_whiten_dim)
        if cv_lof_allow_datasets is None:
            self.cv_lof_allow_datasets = ("cifar10",) if bool(cv_lof) else ()
        else:
            self.cv_lof_allow_datasets = tuple(
                str(x).strip().lower()
                for x in cv_lof_allow_datasets
                if str(x).strip()
            )
        self.nlp_lof = bool(nlp_lof)
        self.nlp_lof_alpha_semi = float(nlp_lof_alpha_semi)
        self.nlp_lof_method = str(nlp_lof_method or "lof").strip().lower()
        self.nlp_lof_mode = str(nlp_lof_mode or "fuse").strip().lower()
        self.nlp_lof_fuse = str(nlp_lof_fuse or "add").strip().lower()
        self.nlp_lof_tail_q = float(nlp_lof_tail_q)
        self.nlp_lof_tail_soft = float(nlp_lof_tail_soft)
        self.nlp_lof_neighbors = int(nlp_lof_neighbors)
        self.nlp_lof_memory_cap = int(nlp_lof_memory_cap)
        self.nlp_lof_whiten = bool(nlp_lof_whiten)
        self.nlp_lof_whiten_dim = int(nlp_lof_whiten_dim)
        if nlp_lof_allow_datasets is None:
            self.nlp_lof_allow_datasets = ("imdb",) if bool(nlp_lof) else ()
        else:
            self.nlp_lof_allow_datasets = tuple(
                str(x).strip().lower()
                for x in nlp_lof_allow_datasets
                if str(x).strip()
            )
        self.profile_router = bool(profile_router)
        self.research_embeds = bool(research_embeds)
        self.ecod = bool(ecod)
        self.ecod_alpha = float(ecod_alpha)
        self.copod = bool(copod)
        self.copod_alpha = float(copod_alpha)
        self.goad = bool(goad)
        self.goad_alpha = float(goad_alpha)
        self.goad_transforms = int(goad_transforms)
        self.goad_epochs = int(goad_epochs)
        self.score_batch_size = int(score_batch_size)
        self.score_batch_size_oom_fallback = int(score_batch_size_oom_fallback)
        self.dataloader_num_workers = int(dataloader_num_workers)
        self.tiny_n_safe_hw = bool(tiny_n_safe_hw)
        self.tiny_n_safe_threshold = int(tiny_n_safe_threshold)
        self.tiny_n_score_batch_size = int(tiny_n_score_batch_size)
        self.tiny_n_dataloader_num_workers = int(tiny_n_dataloader_num_workers)
        self.mid_n_safe_hw = bool(mid_n_safe_hw)
        self.mid_n_score_batch_size = int(mid_n_score_batch_size)
        self.mid_n_dataloader_num_workers = int(mid_n_dataloader_num_workers)
        self.huge_n_safe_hw = bool(huge_n_safe_hw)
        self.huge_n_safe_threshold = int(huge_n_safe_threshold)
        self.huge_n_score_batch_size = int(huge_n_score_batch_size)
        self.huge_n_dataloader_num_workers = int(huge_n_dataloader_num_workers)
        self.mcs_pack_masks = bool(mcs_pack_masks)
        self.vectorized_masks = bool(vectorized_masks)
        self.dataset_hint = (str(dataset_hint).strip().lower() if dataset_hint else "")
        self.seed = seed
        self.max_train_samples = max_train_samples

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._spear_ext: Optional[SpearExtension] = None
        self._rare_ext: Optional[RareExtension] = None
        self._edge_ext: Optional[EdgeExtension] = None
        self._nest_ext: Optional[NestExtension] = None
        self._ecod_view: Optional[EcodView] = None
        self._copod_view: Optional[CopodView] = None
        self._goad_ext: Optional[GoadLiteExtension] = None
        self._cv_knn_view: Optional[CvKnnView] = None
        self._cv_lof_view: Optional[NlpLofView] = None
        self._nlp_lof_view: Optional[NlpLofView] = None
        self.category_profile_: Optional[CategoryProfile] = None
        self._hw_profile = None
        self._hw_resolved_with_n = False
        self._score_bs: int = 512
        self._loader_workers: int = 0
        self.net: Optional[AxionNet] = None
        self.z_mean_: Optional[np.ndarray] = None
        self.z_inv_var_: Optional[np.ndarray] = None
        self.x_mean_: Optional[np.ndarray] = None
        self.x_inv_var_: Optional[np.ndarray] = None
        self.resolved_: Dict[str, Any] = {}
        self.best_val_loss_: float = float("inf")
        self.active_latch_alpha_: float = float(latch_alpha)
        self.active_mae_weight_: float = float(mae_weight)
        self.active_nll_weight_: float = float(nll_weight)
        self.active_hedge_alpha_: float = 0.0
        self.active_x_latch_alpha_: float = 0.0
        self.active_vis_alpha_: float = 0.0
        self.active_disagree_alpha_: float = 0.0
        self.active_hpd_alpha_: float = 0.0
        self.active_c_hedge_alpha_: float = 0.0
        self.is_semi_: bool = False
        self.used_pca_: bool = False
        self.raw_d_: int = 0
        self.pca_mean_: Optional[np.ndarray] = None
        self.pca_components_: Optional[np.ndarray] = None  # (k, d_raw)
        self.pca_scales_: Optional[np.ndarray] = None  # whitening 1/sqrt(eig)
        self.mcs_mean_: Optional[float] = None
        self.mcs_std_: Optional[float] = None
        self.latch_score_mean_: Optional[float] = None
        self.latch_score_std_: Optional[float] = None
        self.x_latch_mean_: Optional[float] = None
        self.x_latch_std_: Optional[float] = None
        self.hedge_mean_: Optional[float] = None
        self.hedge_std_: Optional[float] = None
        self.vis_mean_: Optional[float] = None
        self.vis_std_: Optional[float] = None
        self.disagree_mean_: Optional[float] = None
        self.disagree_std_: Optional[float] = None
        self.hpd_mean_: Optional[float] = None
        self.hpd_std_: Optional[float] = None
        self.c_hedge_mean_: Optional[float] = None
        self.c_hedge_std_: Optional[float] = None
        self.bank_weights_: Optional[np.ndarray] = None
        self.calix_scale_: Dict[str, float] = {}
        self._ensemble_nets: List[AxionNet] = []
        self.c_pca_mean_: Optional[np.ndarray] = None
        self.c_pca_components_: Optional[np.ndarray] = None
        self.c_pca_scales_: Optional[np.ndarray] = None
        self._X_raw_train_: Optional[np.ndarray] = None

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "resolved": self.resolved_,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "latch_alpha": self.latch_alpha,
            "latch_alpha_semi": self.latch_alpha_semi,
            "latch_alpha_semi_tiny": self.latch_alpha_semi_tiny,
            "latch_alpha_semi_huge": self.latch_alpha_semi_huge,
            "latch_alpha_semi_huge_mid": self.latch_alpha_semi_huge_mid,
            "latch_alpha_semi_high_d": self.latch_alpha_semi_high_d,
            "latch_alpha_unsup_huge": self.latch_alpha_unsup_huge,
            "latch_alpha_unsup_large_mid": self.latch_alpha_unsup_large_mid,
            "latch_alpha_unsup_tiny": self.latch_alpha_unsup_tiny,
            "robust_latch_unsup_huge": self.robust_latch_unsup_huge,
            "x_latch_alpha_unsup_huge": self.x_latch_alpha_unsup_huge,
            "x_latch_max_d": self.x_latch_max_d,
            "contam_trim_frac": self.contam_trim_frac,
            "latch_alpha_unsup_low_d": self.latch_alpha_unsup_low_d,
            "active_latch_alpha": self.active_latch_alpha_,
            "active_x_latch_alpha": self.active_x_latch_alpha_,
            "active_hedge_alpha": self.active_hedge_alpha_,
            "active_vis_alpha": self.active_vis_alpha_,
            "active_disagree_alpha": self.active_disagree_alpha_,
            "active_hpd_alpha": self.active_hpd_alpha_,
            "active_c_hedge_alpha": self.active_c_hedge_alpha_,
            "mae_weight": self.mae_weight,
            "nll_weight": self.nll_weight,
            "active_mae_weight": self.active_mae_weight_,
            "active_nll_weight": self.active_nll_weight_,
            "is_semi": self.is_semi_,
            "used_pca": self.used_pca_,
            "pca_dim": self.pca_dim,
            "pca_variance": self.pca_variance,
            "pca_k_min": self.pca_k_min,
            "hedge_alpha": self.hedge_alpha,
            "hedge_alpha_semi_high_d": self.hedge_alpha_semi_high_d,
            "hedge_alpha_nlp": self.hedge_alpha_nlp,
            "hedge_alpha_semi_nlp": self.hedge_alpha_semi_nlp,
            "hedge_alpha_semi_nlp_multi": self.hedge_alpha_semi_nlp_multi,
            "vis_alpha_semi": self.vis_alpha_semi,
            "c_hedge_alpha_semi": self.c_hedge_alpha_semi,
            "c_hedge_alpha_semi_huge": self.c_hedge_alpha_semi_huge,
            "learn_bank_weights": self.learn_bank_weights,
            "block_prob": self.block_prob,
            "true_ensemble": self.true_ensemble,
            "ensemble_heads": self.ensemble_heads,
            "tiny_epoch_boost": self.tiny_epoch_boost,
            "semi_epoch_boost": self.semi_epoch_boost,
            "dataset_hint": self.dataset_hint,
            "use_interaction": self.use_interaction,
            "contrastive_weight": self.contrastive_weight,
            "latch_shrinkage": self.latch_shrinkage,
            "spear": self.spear,
            "spear_gamma": self.spear_gamma,
            "spear_gamma_nlp": self.spear_gamma_nlp,
            "spear_fuse": self.spear_fuse,
            "spear_fuse_nlp": self.spear_fuse_nlp,
            "spear_tail_q": self.spear_tail_q,
            "spear_active": bool(
                self._spear_ext is not None and self._spear_ext.active(self)
            ),
            "spear_resolved": (
                dict(self._spear_ext.resolved_) if self._spear_ext is not None else {}
            ),
            "rare": self.rare,
            "rare_gamma": self.rare_gamma,
            "rare_gamma_cover": self.rare_gamma_cover,
            "rare_active": bool(
                self._rare_ext is not None and self._rare_ext.active(self)
            ),
            "rare_resolved": (
                dict(self._rare_ext.resolved_) if self._rare_ext is not None else {}
            ),
            "edge": self.edge,
            "edge_gamma_cv": self.edge_gamma_cv,
            "edge_gamma_nlp": self.edge_gamma_nlp,
            "edge_whiten": self.edge_whiten,
            "edge_knn_view": self.edge_knn_view,
            "edge_active": bool(
                self._edge_ext is not None and self._edge_ext.active(self)
            ),
            "edge_resolved": (
                dict(self._edge_ext.resolved_) if self._edge_ext is not None else {}
            ),
            "nest": self.nest,
            "nest_gamma_cv": self.nest_gamma_cv,
            "nest_gamma_nlp": self.nest_gamma_nlp,
            "nest_active": bool(
                self._nest_ext is not None and self._nest_ext.active(self)
            ),
            "nest_resolved": (
                dict(self._nest_ext.resolved_) if self._nest_ext is not None else {}
            ),
            "profile_router": self.profile_router,
            "research_embeds": self.research_embeds,
            "category_profile": (
                self.category_profile_.to_dict()
                if self.category_profile_ is not None
                else {}
            ),
            "ecod": self.ecod,
            "ecod_alpha": self.ecod_alpha,
            "ecod_active": bool(self._ecod_view is not None and self._ecod_view.active()),
            "copod": self.copod,
            "copod_alpha": self.copod_alpha,
            "copod_active": bool(
                self._copod_view is not None and self._copod_view.active()
            ),
            "goad": self.goad,
            "goad_alpha": self.goad_alpha,
            "goad_active": bool(self._goad_ext is not None and self._goad_ext.active()),
            "cv_knn": self.cv_knn,
            "cv_knn_alpha_semi": self.cv_knn_alpha_semi,
            "cv_knn_active": bool(
                self._cv_knn_view is not None and self._cv_knn_view.active()
            ),
            "cv_lof": self.cv_lof,
            "cv_lof_method": self.cv_lof_method,
            "cv_lof_mode": self.cv_lof_mode,
            "cv_lof_active": bool(
                self._cv_lof_view is not None and self._cv_lof_view.active()
            ),
            "nlp_lof": self.nlp_lof,
            "nlp_lof_alpha_semi": self.nlp_lof_alpha_semi,
            "nlp_lof_method": self.nlp_lof_method,
            "nlp_lof_mode": self.nlp_lof_mode,
            "nlp_lof_whiten": self.nlp_lof_whiten,
            "nlp_lof_active": bool(
                self._nlp_lof_view is not None and self._nlp_lof_view.active()
            ),
            "score_batch_size": self._score_bs,
            "score_batch_size_oom_fallback": self.score_batch_size_oom_fallback,
            "dataloader_num_workers": self._loader_workers,
            "tiny_n_safe_hw": self.tiny_n_safe_hw,
            "mcs_pack_masks": self.mcs_pack_masks,
            "vectorized_masks": self.vectorized_masks,
            "device": str(self.device),
            "best_val_loss": self.best_val_loss_,
            "train_anchored": self.mcs_mean_ is not None,
        }

    def _ensure_hw(self, d: int, n: Optional[int] = None) -> None:
        """Configure threads / TF32 and resolve score batch + loader workers.

        When ``tiny_n_safe_hw`` is set and ``n`` is below the threshold (glass-scale),
        force g5-safe score_batch=512 and workers=0 so max-HW does not drift tiny-n.

        When ``mid_n_safe_hw`` is set and ``tiny_threshold <= n <= huge_threshold``,
        also force g5-safe HW (breastw/thyroid/FashionMNIST claim parity).

        When ``huge_n_safe_hw`` is set and ``n`` exceeds the threshold on classical
        (non-embed) data, also force g5-safe HW (cover/fraud lock; max-HW alone
        dropped cover PR ~42→28 in bisect).

        Safe-HW also clamps train ``batch_size`` to ≤512 (g5 parity). Mid-n used to
        force score_batch only; ship configs with batch_size=2048 then drifted breastw.
        """
        # Score paths call without n after fit: keep fit-time resolution.
        # If a prior call lacked n, allow one re-resolve when n becomes known.
        if self._hw_profile is not None and (
            n is None or getattr(self, "_hw_resolved_with_n", False)
        ):
            return
        sbs = int(self.score_batch_size)
        workers = int(self.dataloader_num_workers)
        tiny_forced = False
        mid_forced = False
        huge_forced = False
        if (
            bool(self.tiny_n_safe_hw)
            and n is not None
            and int(n) < int(self.tiny_n_safe_threshold)
        ):
            sbs = int(self.tiny_n_score_batch_size)
            workers = int(self.tiny_n_dataloader_num_workers)
            tiny_forced = True
        elif (
            bool(self.huge_n_safe_hw)
            and n is not None
            and int(n) > int(self.huge_n_safe_threshold)
            and not self._is_embed_modality_for_hw()
        ):
            sbs = int(self.huge_n_score_batch_size)
            workers = int(self.huge_n_dataloader_num_workers)
            huge_forced = True
        elif (
            bool(self.mid_n_safe_hw)
            and n is not None
            and int(n) >= int(self.tiny_n_safe_threshold)
            and int(n) <= int(self.huge_n_safe_threshold)
        ):
            sbs = int(self.mid_n_score_batch_size)
            workers = int(self.mid_n_dataloader_num_workers)
            mid_forced = True
        prof = build_profile(
            int(d),
            device=self.device,
            score_batch_size=sbs,
            dataloader_num_workers=workers,
        )
        self._hw_profile = prof
        self._score_bs = int(prof.score_batch_size)
        self._loader_workers = int(prof.dataloader_num_workers)
        if n is not None:
            self._hw_resolved_with_n = True
        if tiny_forced or mid_forced or huge_forced:
            # g5 train batch parity for all safe-HW bands (not only huge_n)
            self.batch_size = int(min(int(self.batch_size), 512))
        self.resolved_["hardware"] = {
            "cpu_count": prof.cpu_count,
            "vram_mb": prof.vram_mb,
            "score_batch_size": prof.score_batch_size,
            "dataloader_num_workers": prof.dataloader_num_workers,
            "torch_num_threads": prof.torch_num_threads,
            "train_batch_size": int(self.batch_size),
            "tiny_n_safe_forced": tiny_forced,
            "mid_n_safe_forced": mid_forced,
            "huge_n_safe_forced": huge_forced,
        }

    def _is_embed_modality_for_hw(self) -> bool:
        """True for CV/NLP embed datasets (max-HW OK); classical stays safe when huge."""
        hint = str(self.dataset_hint or "").lower()
        if any(
            k in hint
            for k in (
                "cifar",
                "fashion",
                "mnist",
                "svhn",
                "mvtec",
                "agnews",
                "imdb",
                "amazon",
                "yelp",
            )
        ):
            return True
        raw_d = int(getattr(self, "raw_d_", 0) or 0)
        if raw_d >= int(self.pca_threshold):
            return True
        try:
            return bool(self._is_nlp_modality(raw_d))
        except Exception:
            return False

    def _shrink_score_batch_on_oom(self) -> bool:
        """Halve score batch toward oom_fallback; return True if shrunk."""
        fb = int(max(512, self.score_batch_size_oom_fallback))
        cur = int(self._score_bs)
        if cur <= fb:
            return False
        new_bs = int(max(fb, cur // 2))
        if new_bs >= cur:
            return False
        self._score_bs = new_bs
        if isinstance(self.resolved_.get("hardware"), dict):
            self.resolved_["hardware"]["score_batch_size"] = new_bs
            self.resolved_["hardware"]["oom_fallback_applied"] = True
        return True

    def _loader_kwargs(self) -> Dict[str, Any]:
        nw = int(self._loader_workers)
        kw: Dict[str, Any] = {
            "pin_memory": (self.device.type == "cuda"),
            "num_workers": nw,
        }
        if nw > 0:
            kw["persistent_workers"] = True
            # Prefetch 4 whenever workers are on: 64 GiB boxes can buffer; keeps A4000 fed
            kw["prefetch_factor"] = 4 if nw >= 2 else 2
        return kw

    def _score_chunk_bs(self, n: int) -> int:
        return int(min(max(512, self._score_bs), max(1, n)))

    def _mcs_chunk_bs(self, n: int) -> int:
        """Chunk size for MCS mask loops.

        Legacy (non-vectorized) masks use per-row Python; huge chunks starve the GPU
        and diverge from g5's ~512 scoring batches. Cap MCS only; encode/recon keep
        large ``_score_bs``.
        """
        bs = self._score_chunk_bs(n)
        if not bool(self.vectorized_masks):
            bs = int(min(bs, 512))
        return int(min(bs, max(1, n)))

    def _sample_masks(self, batch_size: int, d: int, **kwargs):
        return sample_masks(
            batch_size,
            d,
            vectorized=bool(self.vectorized_masks),
            **kwargs,
        )

    def _detect_semi(self, y_train: Optional[np.ndarray]) -> bool:
        if y_train is None:
            return False
        y = np.asarray(y_train).reshape(-1)
        return bool(y.size > 0 and np.all(y == 0))

    def _fit_pca(self, X: np.ndarray) -> np.ndarray:
        """Train-only PCA whitening → (n, k). Rank by variance cap then pca_dim."""
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        self.raw_d_ = d
        k_cap = int(min(self.pca_dim, d - 1, n - 1))
        k_cap = max(2, k_cap)
        mean = X.mean(axis=0)
        Xc = X - mean
        X_fit = Xc
        if n > 20000:
            rng = np.random.RandomState(self.seed + 17)
            idx = rng.choice(n, size=20000, replace=False)
            X_fit = Xc[idx]
        _, s, vt = np.linalg.svd(X_fit, full_matrices=False)
        n_keep = int(min(vt.shape[0], np.count_nonzero(s > 1e-8)))
        n_keep = max(2, n_keep)
        n_fit = max(1, X_fit.shape[0] - 1)
        eig_all = (s[:n_keep] ** 2) / float(n_fit)
        # Adaptive rank: retain pca_variance of train energy, capped by pca_dim
        if self.pca_variance > 0 and self.pca_variance < 1.0 and eig_all.sum() > 0:
            cum = np.cumsum(eig_all) / float(eig_all.sum())
            k_var = int(np.searchsorted(cum, float(self.pca_variance)) + 1)
            k = int(min(k_cap, max(2, k_var)))
        else:
            k = k_cap
        k = int(min(k, n_keep))
        k = max(int(self.pca_k_min), k)
        k = int(min(k, k_cap, n_keep))
        k = max(2, k)
        components = vt[:k].astype(np.float64)
        eig = eig_all[:k]
        scales = 1.0 / np.sqrt(np.maximum(eig, 1e-8))
        self.pca_mean_ = mean.astype(np.float32)
        self.pca_components_ = components.astype(np.float32)
        self.pca_scales_ = scales.astype(np.float32)
        self.used_pca_ = True
        return self._transform_pca(X.astype(np.float32))

    def _transform_pca(self, X: np.ndarray) -> np.ndarray:
        assert self.pca_mean_ is not None and self.pca_components_ is not None
        assert self.pca_scales_ is not None
        X = np.asarray(X, dtype=np.float32)
        Xc = X - self.pca_mean_
        z = Xc @ self.pca_components_.T
        return (z * self.pca_scales_.reshape(1, -1)).astype(np.float32)

    def _pca_residual(self, X: np.ndarray) -> np.ndarray:
        """Energy outside the train PCA subspace (HEDGE geometry score)."""
        assert self.pca_mean_ is not None and self.pca_components_ is not None
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.pca_mean_.astype(np.float64)
        comps = self.pca_components_.astype(np.float64)
        proj = (Xc @ comps.T) @ comps
        return ((Xc - proj) ** 2).sum(axis=1).astype(np.float64)

    def _maybe_project(self, X: np.ndarray) -> np.ndarray:
        if self.used_pca_:
            return self._transform_pca(X)
        return np.asarray(X, dtype=np.float32)

    def _resolve_scale(self, n: int, d: int, *, semi: bool, raw_d: int) -> Dict[str, Any]:
        base = scale_hparams(
            n,
            d,
            semi=semi,
            raw_d=raw_d,
            tiny_n_threshold=int(self.tiny_n_safe_threshold),
        )
        return {
            "hidden": int(self.hidden if self.hidden is not None else base["hidden"]),
            "latent": int(self.latent if self.latent is not None else base["latent"]),
            "depth": int(self.depth if self.depth is not None else base["depth"]),
            "mask_rates": tuple(
                self.mask_rates if self.mask_rates is not None else base["mask_rates"]
            ),
            "score_k": int(self.score_k if self.score_k is not None else base["score_k"]),
            "dropout": float(self.dropout if self.dropout is not None else base["dropout"]),
            "high_mask_delta": float(base.get("high_mask_delta", 0.25)),
            "high_mask_cap": float(base.get("high_mask_cap", 0.85)),
            "semi": bool(semi),
            "raw_d": int(raw_d),
            "tiny_n": bool(base.get("tiny_n", False)),
        }

    def _torch_gen(self) -> torch.Generator:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(self.seed))
        return g

    def _rate_banks(self) -> List[Tuple[float, ...]]:
        rates = tuple(self.resolved_.get("mask_rates", (0.2, 0.35, 0.5)))
        delta = float(self.resolved_.get("high_mask_delta", 0.25))
        cap = float(self.resolved_.get("high_mask_cap", 0.85))
        high = float(min(cap, max(rates) + delta))
        banks: List[Tuple[float, ...]] = [rates, (high,)]
        light = float(max(0.05, min(rates) * 0.6))
        if light + 1e-6 < min(rates):
            banks.append((light,))
        return banks

    def _is_nlp_modality(self, raw_d: int) -> bool:
        """NLP vs CV for high-d hedge routing.

        Prefer dataset_hint (ViT is d=1000 and must NOT be treated as NLP).
        Fallback: raw_d ≥ nlp_raw_d_min (BERT/RoBERTa/MiniLM @ 384–768).
        """
        hint = self.dataset_hint or ""
        if hint:
            if any(
                k in hint
                for k in ("cifar", "fashion", "mnist", "svhn", "mvtec", "imagenet")
            ):
                return False
            if any(
                k in hint
                for k in ("agnews", "imdb", "amazon", "yelp", "20news", "newsgroup")
            ):
                return True
        return int(raw_d) >= int(self.nlp_raw_d_min)

    def _is_nlp_multi(self) -> bool:
        """Agnews-scale multi-file NLP (not single-file Imdb)."""
        hint = self.dataset_hint
        if hint:
            if "imdb" in hint:
                return False
            if "agnews" in hint or "ag_news" in hint or "20news" in hint:
                return True
        return False

    def _resolve_setting_knobs(
        self,
        y_train: Optional[np.ndarray],
        *,
        high_d: bool,
        raw_d: int,
        n: int,
    ) -> None:
        """Classical semi latch + high-d HEDGE; huge/mid/tiny unsup latch polish.

        BERT-scale NLP (hint or raw_d ≥ nlp_raw_d_min): soft NLP hedge on unsup; Imdb semi
        hedge off; Agnews (nlp multi) uses hedge_alpha_semi_nlp_multi.
        CV (incl. ViT d=1000) keeps full HEDGE.
        """
        self.is_semi_ = self._detect_semi(y_train)
        self.active_latch_alpha_ = float(self.latch_alpha)
        self.active_mae_weight_ = float(self.mae_weight)
        self.active_nll_weight_ = float(self.nll_weight)
        self.active_x_latch_alpha_ = 0.0
        self.active_vis_alpha_ = 0.0
        self.active_disagree_alpha_ = 0.0
        self.active_hpd_alpha_ = 0.0
        self.active_c_hedge_alpha_ = 0.0
        is_nlp = self._is_nlp_modality(raw_d)
        huge = int(n) > 10000
        # glass n=214 must count as tiny (threshold 250); hardcoded 200 missed it
        tiny = int(n) < int(self.tiny_n_safe_threshold)
        mid_d = (not high_d) and (64 <= int(raw_d) < int(self.pca_threshold))
        low_d = (not high_d) and int(raw_d) < 64
        very_low_d = (not high_d) and int(raw_d) <= int(self.x_latch_max_d)

        if high_d:
            self.active_hedge_alpha_ = (
                float(self.hedge_alpha_nlp) if is_nlp else float(self.hedge_alpha)
            )
        else:
            self.active_hedge_alpha_ = 0.0

        if not self.is_semi_:
            if huge and mid_d:
                self.active_latch_alpha_ = float(self.latch_alpha_unsup_large_mid)
                self.active_mae_weight_ = float(self.mae_weight_unsup_huge)
                self.active_nll_weight_ = float(self.nll_weight_unsup_huge)
            elif huge:
                self.active_latch_alpha_ = float(self.latch_alpha_unsup_huge)
                self.active_mae_weight_ = float(self.mae_weight_unsup_huge)
                self.active_nll_weight_ = float(self.nll_weight_unsup_huge)
                if low_d:
                    self.active_latch_alpha_ = float(
                        max(self.active_latch_alpha_, self.latch_alpha_unsup_low_d)
                    )
                if very_low_d:
                    self.active_x_latch_alpha_ = float(self.x_latch_alpha_unsup_huge)
            elif tiny:
                self.active_latch_alpha_ = float(self.latch_alpha_unsup_tiny)
            w = self.active_mae_weight_ + self.active_nll_weight_
            if w > 1e-8:
                self.active_mae_weight_ /= w
                self.active_nll_weight_ /= w
            return

        # --- Semi-supervised ---
        if high_d and self.latch_alpha_semi_high_d is not None:
            self.active_latch_alpha_ = float(self.latch_alpha_semi_high_d)
        elif tiny and self.latch_alpha_semi_tiny is not None:
            self.active_latch_alpha_ = float(self.latch_alpha_semi_tiny)
        elif huge and (16 < int(raw_d) < 64) and self.latch_alpha_semi_huge_mid is not None:
            # fraud-scale: huge-n mid-low-d (d=29)
            self.active_latch_alpha_ = float(self.latch_alpha_semi_huge_mid)
        elif huge and self.latch_alpha_semi_huge is not None:
            self.active_latch_alpha_ = float(self.latch_alpha_semi_huge)
        elif self.latch_alpha_semi is not None:
            self.active_latch_alpha_ = float(self.latch_alpha_semi)

        if high_d:
            if is_nlp:
                if self._is_nlp_multi():
                    self.active_hedge_alpha_ = float(self.hedge_alpha_semi_nlp_multi)
                else:
                    self.active_hedge_alpha_ = float(self.hedge_alpha_semi_nlp)
            else:
                self.active_hedge_alpha_ = float(self.hedge_alpha_semi_high_d)

        if self.mae_weight_semi is not None:
            self.active_mae_weight_ = float(self.mae_weight_semi)
        if self.nll_weight_semi is not None:
            self.active_nll_weight_ = float(self.nll_weight_semi)
        w = self.active_mae_weight_ + self.active_nll_weight_
        if w > 1e-8:
            self.active_mae_weight_ /= w
            self.active_nll_weight_ /= w

        # Classical / low-d semi score views (not Imdb-fragile)
        # Tiny-n (glass): MCS-primary like g5 / huge-n. Multi-view fusion dilutes sparse PR.
        if not high_d:
            if tiny:
                self.active_vis_alpha_ = 0.0
                self.active_disagree_alpha_ = 0.0
                self.active_hpd_alpha_ = 0.0
                self.active_c_hedge_alpha_ = 0.0
            elif huge:
                self.active_vis_alpha_ = float(self.vis_alpha_semi_huge)
                self.active_disagree_alpha_ = float(self.disagree_alpha_semi_huge)
                self.active_hpd_alpha_ = float(self.hpd_alpha_semi_huge)
                self.active_c_hedge_alpha_ = float(self.c_hedge_alpha_semi_huge)
            else:
                self.active_vis_alpha_ = float(self.vis_alpha_semi)
                self.active_disagree_alpha_ = float(self.disagree_alpha_semi)
                self.active_hpd_alpha_ = float(self.hpd_alpha_semi)
                self.active_c_hedge_alpha_ = float(self.c_hedge_alpha_semi)
        else:
            # Mild VIS only on CV; NLP stays MCS+LATCH+HEDGE
            if not is_nlp:
                self.active_vis_alpha_ = float(self.vis_alpha_semi) * 0.5
                self.active_disagree_alpha_ = float(self.disagree_alpha_semi) * 0.5

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "AxionModel":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        hint = str(getattr(self, "dataset_hint", "") or "axion")
        step_log(f"FIT begin dataset={hint} seed={self.seed} X={tuple(np.asarray(X_train).shape)}")

        X = np.asarray(X_train, dtype=np.float32)
        # Large-n: prefer larger train cap when caller left default 0 and n huge
        # (harness usually passes max_train_samples; keep behavior if set)
        if self.max_train_samples > 0 and X.shape[0] > self.max_train_samples:
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(X.shape[0], size=self.max_train_samples, replace=False)
            X = X[idx]
            if y_train is not None:
                y_train = np.asarray(y_train)[idx]
            step_log(f"FIT subsample → n={X.shape[0]} (max_train_samples={self.max_train_samples})")

        raw_d = int(X.shape[1])
        self.raw_d_ = raw_d
        high_d = raw_d >= int(self.pca_threshold)
        self.used_pca_ = False
        self.pca_mean_ = None
        self.pca_components_ = None
        self.pca_scales_ = None
        self._X_raw_train_ = X.copy() if high_d else None

        if high_d:
            with stage("pca", detail=f"{hint} raw_d={raw_d}"):
                X = self._fit_pca(X)

        n_rows = int(X.shape[0])
        self._resolve_setting_knobs(
            y_train, high_d=high_d, raw_d=raw_d, n=n_rows
        )
        n, d = X.shape
        self._apply_category_profile(n=n_rows, d=raw_d)
        hw_before = dict(self.resolved_.get("hardware") or {})
        self._ensure_hw(d, n=n)
        hw_meta = dict(self.resolved_.get("hardware") or hw_before)
        hp = self._resolve_scale(n, d, semi=self.is_semi_, raw_d=raw_d)
        self.resolved_ = dict(hp)
        # Preserve HW audit (was wiped by resolved_ = dict(hp))
        if hw_meta:
            self.resolved_["hardware"] = hw_meta
        self.resolved_["n"] = n
        self.resolved_["d"] = d
        self.resolved_["raw_d"] = raw_d
        self.resolved_["used_pca"] = self.used_pca_
        self.resolved_["pca_k"] = int(d) if self.used_pca_ else 0
        self.resolved_["active_latch_alpha"] = self.active_latch_alpha_
        self.resolved_["active_x_latch_alpha"] = self.active_x_latch_alpha_
        self.resolved_["active_hedge_alpha"] = self.active_hedge_alpha_
        self.resolved_["active_vis_alpha"] = self.active_vis_alpha_
        self.resolved_["active_disagree_alpha"] = self.active_disagree_alpha_
        self.resolved_["active_hpd_alpha"] = self.active_hpd_alpha_
        self.resolved_["active_c_hedge_alpha"] = self.active_c_hedge_alpha_
        self.resolved_["active_mae_weight"] = self.active_mae_weight_
        self.resolved_["active_nll_weight"] = self.active_nll_weight_
        self.resolved_["dataset_hint"] = self.dataset_hint

        rng = np.random.RandomState(self.seed)
        fit_idx: Optional[np.ndarray] = None
        # Tiny-n: do not hold out val — every row matters (glass n=214)
        use_val = self.val_fraction > 0 and n >= int(self.tiny_n_safe_threshold)
        if use_val and n >= 40:
            n_val = max(8, int(round(n * self.val_fraction)))
            n_val = min(n_val, n // 5 if n >= 50 else max(4, n // 4))
            perm = rng.permutation(n)
            val_idx, fit_idx = perm[:n_val], perm[n_val:]
            X_fit, X_val = X[fit_idx], X[val_idx]
        else:
            X_fit, X_val = X, X[:0]
            val_idx = None

        use_ix = bool(self.use_interaction) and (
            not high_d or (self.is_semi_ and not self._is_nlp_modality(raw_d))
        )
        self.resolved_["use_interaction"] = use_ix
        self.resolved_["ensemble_heads"] = int(self.ensemble_heads)
        self.resolved_["true_ensemble"] = bool(self.true_ensemble)
        self.resolved_["block_prob"] = float(self.block_prob)
        self.net = AxionNet(
            d_in=d,
            hidden=hp["hidden"],
            latent=hp["latent"],
            depth=hp["depth"],
            dropout=hp["dropout"],
            use_interaction=use_ix,
            interaction_rank=self.interaction_rank,
        ).to(self.device)
        self._ensemble_nets = []

        opt = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        fit_ds = TensorDataset(torch.from_numpy(X_fit))
        req_bs = self.batch_size
        if self.device.type == "cuda":
            req_bs = max(req_bs, 512)
        bs = min(req_bs, max(8, len(X_fit)))
        loader = DataLoader(
            fit_ds,
            batch_size=bs,
            shuffle=True,
            drop_last=False,
            **self._loader_kwargs(),
        )

        best_state = None
        best_val = float("inf")
        stale = 0
        gen = self._torch_gen()
        use_amp = self.device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except (TypeError, AttributeError):
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        max_epochs = int(self.epochs)
        patience = int(self.patience)
        if self.is_semi_ and self.semi_epoch_boost > 1.0:
            max_epochs = int(round(max_epochs * float(self.semi_epoch_boost)))
            patience = int(round(patience * float(self.semi_epoch_boost)))
        if n < int(self.tiny_n_safe_threshold) and self.tiny_epoch_boost > 1.0:
            max_epochs = int(round(max_epochs * float(self.tiny_epoch_boost)))
            patience = int(round(patience * float(self.tiny_epoch_boost)))
        self.resolved_["max_epochs"] = max_epochs
        self.resolved_["patience"] = patience

        # Contaminated-train trim: huge unsup only (cover/fraud-scale)
        use_contam = (
            (not self.is_semi_)
            and int(n_rows) > 10000
            and float(self.contam_trim_frac) > 1e-8
        )
        X_fit_active = X_fit
        self.resolved_["contam_trim_frac"] = float(self.contam_trim_frac) if use_contam else 0.0
        self.resolved_["contam_kept"] = int(len(X_fit))

        step_log(
            f"FIT train epochs={max_epochs} patience={patience} "
            f"n_fit={len(X_fit)} n_val={len(X_val)} d={d} semi={self.is_semi_} "
            f"device={self.device}"
        )
        epoch_bar = ProgressBar(
            max_epochs,
            desc=f"epochs:{hint}",
            leave=True,
            unit="ep",
            position=2,
        )
        show_batch = verbose_enabled() or os.environ.get(
            "AXION_EPOCH_BATCH_BAR", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        try:
            for epoch in range(max_epochs):
                if (
                    use_contam
                    and epoch >= int(self.contam_trim_warmup)
                    and (epoch - int(self.contam_trim_warmup)) % int(self.contam_trim_every) == 0
                ):
                    err = self._recon_error(X_fit)
                    q = float(np.clip(1.0 - float(self.contam_trim_frac), 0.5, 0.999))
                    thr = float(np.quantile(err, q))
                    keep = err <= thr
                    # Always keep at least 80% so trim cannot collapse the fit set
                    if keep.mean() < 0.80:
                        thr = float(np.quantile(err, 0.80))
                        keep = err <= thr
                    if int(keep.sum()) >= max(64, int(0.5 * len(X_fit))):
                        X_fit_active = X_fit[keep]
                        self.resolved_["contam_kept"] = int(keep.sum())
                        fit_ds = TensorDataset(torch.from_numpy(X_fit_active))
                        bs = min(req_bs, max(8, len(X_fit_active)))
                        loader = DataLoader(
                            fit_ds,
                            batch_size=bs,
                            shuffle=True,
                            drop_last=False,
                            **self._loader_kwargs(),
                        )
                        step_log(
                            f"  contam trim epoch={epoch} kept={int(keep.sum())}/{len(X_fit)}"
                        )

                self.net.train()
                last_loss = 0.0
                # Semi loss curriculum: MAE-heavier early → more NLL later (via mask hardness)
                if self.is_semi_ and self.semi_loss_curriculum and max_epochs > 1:
                    t = float(epoch) / float(max(1, max_epochs - 1))
                    hard_factor = 0.85 + 0.30 * t  # soft → harder masks
                else:
                    hard_factor = 1.0
                rates_epoch = tuple(
                    float(max(0.05, min(0.7, r * hard_factor))) for r in hp["mask_rates"]
                )
                # Contrastive only on semi classical / CV (not NLP Imdb-fragile)
                use_contrast = (
                    self.is_semi_
                    and float(self.contrastive_weight) > 1e-8
                    and not self._is_nlp_modality(raw_d)
                )
                n_batches = max(1, len(loader))
                for bi, (xb,) in enumerate(loader, start=1):
                    if show_batch and (bi == 1 or bi == n_batches or bi % max(1, n_batches // 5) == 0):
                        epoch_bar.set_postfix(batch=f"{bi}/{n_batches}")
                    xb = xb.to(self.device, non_blocking=True)
                    mask = self._sample_masks(
                        xb.shape[0],
                        d,
                        rates=rates_epoch,
                        generator=gen,
                        device=self.device,
                        block_prob=float(self.block_prob),
                    )
                    opt.zero_grad(set_to_none=True)
                    if use_amp:
                        # Forward under autocast; loss in fp32 outside (g5-faithful)
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            mu, log_var, z1 = self.net(xb, mask)
                        loss = gaussian_nll(xb, mu.float(), log_var.float(), mask).mean()
                        if use_contrast:
                            mask2 = self._sample_masks(
                                xb.shape[0],
                                d,
                                rates=rates_epoch,
                                generator=gen,
                                device=self.device,
                                block_prob=float(self.block_prob),
                            )
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                _mu2, _lv2, z2 = self.net(xb, mask2)
                            z1n = torch.nn.functional.normalize(z1.float(), dim=-1)
                            z2n = torch.nn.functional.normalize(z2.float(), dim=-1)
                            cos = (z1n * z2n).sum(dim=-1)
                            loss = loss + float(self.contrastive_weight) * (1.0 - cos).mean()
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        mu, log_var, z1 = self.net(xb, mask)
                        loss = gaussian_nll(xb, mu, log_var, mask).mean()
                        if use_contrast:
                            mask2 = self._sample_masks(
                                xb.shape[0],
                                d,
                                rates=rates_epoch,
                                generator=gen,
                                device=self.device,
                                block_prob=float(self.block_prob),
                            )
                            _mu2, _lv2, z2 = self.net(xb, mask2)
                            z1n = torch.nn.functional.normalize(z1, dim=-1)
                            z2n = torch.nn.functional.normalize(z2, dim=-1)
                            cos = (z1n * z2n).sum(dim=-1)
                            loss = loss + float(self.contrastive_weight) * (1.0 - cos).mean()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                        opt.step()
                    last_loss = float(loss.detach().cpu().item())

                if len(X_val) > 0:
                    val_loss = self._eval_nll(X_val, hp["mask_rates"], seed_offset=epoch)
                else:
                    val_loss = last_loss

                epoch_bar.set_description(f"epochs:{hint} loss={val_loss:.4f}")
                epoch_bar.update(1)

                if val_loss + 1e-6 < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break
        finally:
            epoch_bar.close()
        step_log(
            f"FIT epochs done best_val={best_val:.6f} stale_stop={stale}>={patience} "
            f"ran≈{epoch_bar.n}/{max_epochs}"
        )

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_loss_ = float(best_val)

        # True multi-net ensemble (classical / low-d semi preferred; always if true_ensemble)
        do_true_ens = (
            bool(self.true_ensemble)
            and int(self.ensemble_heads) > 1
            and (self.is_semi_ and not high_d)
        )
        self.resolved_["true_ensemble_trained"] = bool(do_true_ens)
        if do_true_ens:
            with stage("true_ensemble", detail=hint):
                self._fit_extra_ensemble_nets(
                    X_fit_active if use_contam else X_fit,
                    X_val=X_val,
                    hp=hp,
                    use_ix=use_ix,
                    d=d,
                    raw_d=raw_d,
                    max_epochs=max_epochs,
                    patience=patience,
                    req_bs=req_bs,
                )

        # Latch / anchors on cleaned fit when contam trim ran
        latch_X = X_fit_active if use_contam else X_fit
        post_stages = [
            "latch",
            "x_latch",
            "c_hedge",
            "bank_weights",
            "score_anchors",
            "spear",
            "rare",
            "edge",
            "nest",
            "market_views",
            "cv_knn",
            "cv_lof",
            "nlp_lof",
        ]
        with progress(
            len(post_stages),
            desc=f"postfit:{hint}",
            leave=True,
            unit="stg",
            position=2,
        ) as post_bar:
            post_bar.set_description("postfit:latch")
            with stage("latch", detail=hint):
                self._fit_latch(latch_X)
            post_bar.update(1)

            post_bar.set_description("postfit:x_latch")
            with stage("x_latch", detail=hint):
                self._fit_x_latch(latch_X)
            post_bar.update(1)

            post_bar.set_description("postfit:c_hedge")
            if (
                abs(float(self.active_c_hedge_alpha_)) > 1e-12
                or bool(self.spear)
                or bool(self.rare)
            ):
                with stage("c_hedge", detail=hint):
                    self._fit_c_hedge(latch_X)
            else:
                self.c_pca_mean_ = None
                self.c_pca_components_ = None
                self.c_pca_scales_ = None
                step_log("c_hedge skipped")
            post_bar.update(1)

            raw_fit = None
            if self.used_pca_ and self._X_raw_train_ is not None:
                raw_fit = (
                    self._X_raw_train_[fit_idx]
                    if fit_idx is not None
                    else self._X_raw_train_
                )
            val_for_calix = X_val if len(X_val) > 0 else None

            post_bar.set_description("postfit:bank_weights")
            if (
                bool(self.learn_bank_weights)
                and bool(self.bank_weight_semi)
                and bool(self.is_semi_)
                and val_for_calix is not None
                and len(val_for_calix) >= 8
            ):
                with stage("bank_weights", detail=hint):
                    self._fit_bank_weights(val_for_calix)
            else:
                step_log("bank_weights skipped")
            post_bar.update(1)

            post_bar.set_description("postfit:score_anchors")
            with stage("score_anchors", detail=hint):
                self._fit_score_anchors(latch_X, X_raw=raw_fit, X_val=val_for_calix)
            post_bar.update(1)

            post_bar.set_description("postfit:spear")
            with stage("spear", detail=hint):
                self._fit_spear(latch_X, X_val=val_for_calix)
            post_bar.update(1)

            post_bar.set_description("postfit:rare")
            with stage("rare", detail=hint):
                self._fit_rare(latch_X, X_val=val_for_calix)
            post_bar.update(1)

            post_bar.set_description("postfit:edge")
            with stage("edge", detail=hint):
                self._fit_edge(latch_X, X_val=val_for_calix)
            post_bar.update(1)

            post_bar.set_description("postfit:nest")
            with stage("nest", detail=hint):
                self._fit_nest(latch_X, X_val=val_for_calix)
            post_bar.update(1)

            post_bar.set_description("postfit:market_views")
            with stage("market_views", detail=hint):
                self._fit_market_views(latch_X)
            post_bar.update(1)

            post_bar.set_description("postfit:cv_knn")
            with stage("cv_knn", detail=hint):
                self._fit_cv_knn(raw_fit if raw_fit is not None else latch_X)
            post_bar.update(1)

            post_bar.set_description("postfit:cv_lof")
            with stage("cv_lof", detail=hint):
                self._fit_cv_lof(raw_fit if raw_fit is not None else latch_X)
            post_bar.update(1)

            post_bar.set_description("postfit:nlp_lof")
            with stage("nlp_lof", detail=hint):
                self._fit_nlp_lof(raw_fit if raw_fit is not None else latch_X)
            post_bar.update(1)

        step_log(f"FIT complete dataset={hint} seed={self.seed}")
        self._X_raw_train_ = None  # free memory
        return self

    def _apply_category_profile(self, *, n: int, d: int) -> None:
        """Resolve atlas bucket; optionally enable market views / hard locks."""
        prof = resolve_profile(
            self.dataset_hint,
            n=n,
            d=d,
            is_semi=bool(self.is_semi_),
            research_embeds=bool(self.research_embeds),
        )
        self.category_profile_ = prof
        if not bool(self.profile_router):
            return
        # Enable market classical views from profile
        if prof.use_ecod and abs(prof.ecod_alpha) > 1e-12:
            self.ecod = True
            self.ecod_alpha = float(prof.ecod_alpha)
        if prof.use_copod and abs(prof.copod_alpha) > 1e-12:
            self.copod = True
            self.copod_alpha = float(prof.copod_alpha)
        if prof.use_goad and abs(prof.goad_alpha) > 1e-12 and bool(self.is_semi_):
            self.goad = True
            self.goad_alpha = float(prof.goad_alpha)
        if prof.use_nest and bool(self.research_embeds):
            self.nest = True
        if prof.use_edge:
            self.edge = True
        elif prof.modality == "nlp" and not prof.use_edge:
            self.edge = False
            self.nest = False
        if not prof.use_spear:
            self.spear = False
        elif bool(self.is_semi_) and prof.modality == "classical":
            self.spear = True
        if prof.use_rare and bool(self.is_semi_):
            self.rare = True
        if prof.vis_alpha_semi is not None:
            self.vis_alpha_semi = float(prof.vis_alpha_semi)
        # Hard locks
        if prof.bucket in ("tiny_classical", "tiny_classical_lock"):
            self.spear = False
            self.rare = False
            self.goad = False
            self.ecod = False
            self.copod = False
            self.nest = False
        if prof.bucket == "backdoor_lock":
            self.rare = False
            self.ecod = False
            self.copod = False
            self.goad = False
        if prof.bucket == "high_d_classical" and bool(self.is_semi_):
            self.spear = True
        if prof.prefer_huge_n_safe_hw:
            self.huge_n_safe_hw = True
            self.vectorized_masks = False
            self.mcs_pack_masks = False
        if not prof.prefer_vectorized and prof.prefer_huge_n_safe_hw:
            self.vectorized_masks = False

    def _fit_market_views(self, X_train: np.ndarray) -> None:
        """Fit ECOD / COPOD / GOAD-lite classical views (train-anchored)."""
        self._ecod_view = None
        self._copod_view = None
        self._goad_ext = None
        X = np.asarray(X_train, dtype=np.float32)
        if bool(self.ecod):
            view = EcodView(enabled=True, alpha=float(self.ecod_alpha))
            view.fit(X)
            self._ecod_view = view
            self.resolved_["ecod"] = view.get_params()
        else:
            self.resolved_["ecod"] = {"enabled": False}
        if bool(self.copod):
            view = CopodView(enabled=True, alpha=float(self.copod_alpha))
            view.fit(X)
            self._copod_view = view
            self.resolved_["copod"] = view.get_params()
        else:
            self.resolved_["copod"] = {"enabled": False}
        if bool(self.goad) and bool(self.is_semi_):
            ext = GoadLiteExtension(
                enabled=True,
                alpha=float(self.goad_alpha),
                n_transforms=int(self.goad_transforms),
                epochs=int(self.goad_epochs),
                seed=int(self.seed),
                device=str(self.device),
            )
            ext.fit(X)
            self._goad_ext = ext
            self.resolved_["goad"] = ext.get_params()
        else:
            self.resolved_["goad"] = {"enabled": False}
        if self.category_profile_ is not None:
            self.resolved_["category_profile"] = self.category_profile_.to_dict()

    def _is_cv_modality(self) -> bool:
        hint = str(self.dataset_hint or "").lower()
        if any(
            k in hint
            for k in (
                "cifar",
                "svhn",
                "fashion",
                "mnist",
                "mvtec",
                "resnet",
                "vit",
            )
        ):
            return True
        # High-d embeds after PCA often CV when not NLP
        raw_d = int(getattr(self, "raw_d_", 0) or 0)
        return bool(self.used_pca_) and not self._is_nlp_modality(raw_d)

    def _fit_cv_knn(self, X_train: np.ndarray) -> None:
        """Fit PatchCore-lite kNN memory on CV embeds (semi only)."""
        self._cv_knn_view = None
        if not bool(self.cv_knn) or not bool(self.is_semi_):
            self.resolved_["cv_knn"] = {"enabled": False, "reason": "off_or_not_semi"}
            return
        if not self._is_cv_modality():
            self.resolved_["cv_knn"] = {"enabled": False, "reason": "not_cv"}
            return
        hint = str(self.dataset_hint or "").lower()
        if bool(self.cv_knn_skip_fashion) and "fashion" in hint:
            self.resolved_["cv_knn"] = {"enabled": False, "reason": "skip_fashion"}
            return
        allow = list(self.cv_knn_allow_datasets or [])
        if allow and not any(a in hint for a in allow):
            self.resolved_["cv_knn"] = {"enabled": False, "reason": "not_allowlisted"}
            return
        alpha = float(self.cv_knn_alpha_semi)
        if abs(alpha) < 1e-12:
            self.resolved_["cv_knn"] = {"enabled": False, "reason": "alpha_zero"}
            return
        view = CvKnnView(
            enabled=True,
            alpha=alpha,
            memory_cap=int(self.cv_knn_memory_cap),
            n_neighbors=int(self.cv_knn_neighbors),
            seed=int(self.seed),
        )
        view.fit(np.asarray(X_train, dtype=np.float32))
        self._cv_knn_view = view
        self.resolved_["cv_knn"] = view.get_params()

    def _fit_cv_lof(self, X_train: np.ndarray) -> None:
        """Fit CV LOF/kNN replace on fair ResNet embeds (semi; CIFAR allowlist).

        Distinct from cv_knn (PatchCore memory fuse). Train-only; no test labels.
        """
        self._cv_lof_view = None
        if not bool(self.cv_lof) or not bool(self.is_semi_):
            self.resolved_["cv_lof"] = {"enabled": False, "reason": "off_or_not_semi"}
            return
        if not self._is_cv_modality():
            self.resolved_["cv_lof"] = {"enabled": False, "reason": "not_cv"}
            return
        hint = str(self.dataset_hint or "").lower()
        allow = list(self.cv_lof_allow_datasets or [])
        if allow and not any(a in hint for a in allow):
            self.resolved_["cv_lof"] = {"enabled": False, "reason": "not_allowlisted"}
            return
        alpha = float(self.cv_lof_alpha_semi)
        if abs(alpha) < 1e-12:
            self.resolved_["cv_lof"] = {"enabled": False, "reason": "alpha_zero"}
            return
        view = NlpLofView(
            enabled=True,
            alpha=alpha,
            method=str(self.cv_lof_method),
            mode=str(self.cv_lof_mode),
            n_neighbors=int(self.cv_lof_neighbors),
            memory_cap=int(self.cv_lof_memory_cap),
            whiten=bool(self.cv_lof_whiten),
            whiten_dim=int(self.cv_lof_whiten_dim),
            allow_datasets=list(self.cv_lof_allow_datasets),
            seed=int(self.seed),
        )
        view.fit(np.asarray(X_train, dtype=np.float32))
        self._cv_lof_view = view
        self.resolved_["cv_lof"] = view.get_params()

    def _fit_nlp_lof(self, X_train: np.ndarray) -> None:
        """Fit NLP LOF/kNN fuse on fair BERT embeds (semi; Imdb allowlist)."""
        self._nlp_lof_view = None
        if not bool(self.nlp_lof) or not bool(self.is_semi_):
            self.resolved_["nlp_lof"] = {"enabled": False, "reason": "off_or_not_semi"}
            return
        hint = str(self.dataset_hint or "").lower()
        raw_d = int(getattr(self, "raw_d_", 0) or 0)
        if not self._is_nlp_modality(raw_d):
            self.resolved_["nlp_lof"] = {"enabled": False, "reason": "not_nlp"}
            return
        allow = list(self.nlp_lof_allow_datasets or [])
        if allow and not any(a in hint for a in allow):
            self.resolved_["nlp_lof"] = {"enabled": False, "reason": "not_allowlisted"}
            return
        alpha = float(self.nlp_lof_alpha_semi)
        if abs(alpha) < 1e-12:
            self.resolved_["nlp_lof"] = {"enabled": False, "reason": "alpha_zero"}
            return
        view = NlpLofView(
            enabled=True,
            alpha=alpha,
            method=str(self.nlp_lof_method),
            mode=str(self.nlp_lof_mode),
            fuse_mode=str(self.nlp_lof_fuse),
            tail_q=float(self.nlp_lof_tail_q),
            tail_soft=float(self.nlp_lof_tail_soft),
            n_neighbors=int(self.nlp_lof_neighbors),
            memory_cap=int(self.nlp_lof_memory_cap),
            whiten=bool(self.nlp_lof_whiten),
            whiten_dim=int(self.nlp_lof_whiten_dim),
            allow_datasets=list(self.nlp_lof_allow_datasets),
            seed=int(self.seed),
        )
        view.fit(np.asarray(X_train, dtype=np.float32))
        if view.active() and str(view.fuse_mode) != "add":
            Xt = np.asarray(X_train, dtype=np.float32)
            Xp = self._maybe_project(Xt)
            mcs_n = self._mcs_scores(Xp)
            view.set_mcs_anchor(mcs_n)
            if str(view.fuse_mode) in ("platt", "platt_tail"):
                from axion.models.extensions.spear import synthesize_hard_negatives

                X_syn = synthesize_hard_negatives(
                    self,
                    Xt,
                    n_synth=int(min(256, max(32, len(Xt)))),
                    seed=int(self.seed) + 77,
                    kinds=("swap",),
                )
                if len(X_syn) >= 8:
                    view.fit_platt(
                        mcs_n,
                        view.z_normed(Xt),
                        self._mcs_scores(self._maybe_project(X_syn)),
                        view.z_normed(X_syn),
                    )
        self._nlp_lof_view = view
        self.resolved_["nlp_lof"] = view.get_params()

    def _fit_spear(
        self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None
    ) -> None:
        """Fit SPEAR ranking extension after frozen core + anchors (semi only)."""
        self._spear_ext = None
        if not bool(self.spear):
            self.resolved_["spear"] = {"enabled": False}
            return
        hint = str(self.dataset_hint or "").lower()
        raw_d = int(getattr(self, "raw_d_", 0) or 0)
        gamma = float(self.spear_gamma)
        fuse = str(self.spear_fuse or "add")
        nlp_like = self._is_nlp_modality(raw_d) or any(
            a in hint for a in (self.spear_allow_datasets or ())
        )
        if nlp_like:
            if self.spear_gamma_nlp is not None:
                gamma = float(self.spear_gamma_nlp)
            if self.spear_fuse_nlp:
                fuse = str(self.spear_fuse_nlp)
        ext = SpearExtension(
            enabled=True,
            gamma=gamma,
            n_pairs=int(self.spear_pairs),
            epochs=int(self.spear_epochs),
            lr=float(self.spear_lr),
            hidden=int(self.spear_hidden),
            classical_only=bool(self.spear_classical_only),
            skip_huge_n=bool(self.spear_skip_huge_n),
            huge_n_threshold=int(self.spear_huge_n_threshold),
            skip_tiny_n=bool(self.spear_skip_tiny_n),
            tiny_n_threshold=int(self.spear_tiny_n_threshold),
            use_soft_ap=bool(self.spear_soft_ap),
            allow_datasets=list(self.spear_allow_datasets),
            rank_fuse=bool(self.spear_rank_fuse),
            fuse_mode=str(fuse),
            tail_q=float(self.spear_tail_q),
            tail_soft=float(self.spear_tail_soft),
            seed=int(self.seed),
            device=str(self.device),
        )
        ext.fit(self, X_train, X_val=X_val)
        self._spear_ext = ext
        self.resolved_["spear"] = ext.get_params()

    def _fit_rare(
        self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None
    ) -> None:
        """Fit RARE huge-n soft-AP extension (cover/fraud semi)."""
        self._rare_ext = None
        if not bool(self.rare):
            self.resolved_["rare"] = {"enabled": False}
            return
        ext = RareExtension(
            enabled=True,
            gamma=float(self.rare_gamma),
            gamma_cover=float(self.rare_gamma_cover),
            n_pairs=int(self.rare_pairs),
            epochs=int(self.rare_epochs),
            lr=float(self.rare_lr),
            hidden=int(self.rare_hidden),
            huge_n_threshold=int(self.rare_huge_n_threshold),
            max_fit_n=int(self.rare_max_fit_n),
            skip_cover=bool(self.rare_skip_cover),
            allow_datasets=tuple(self.rare_allow_datasets),
            seed=int(self.seed),
            device=str(self.device),
        )
        ext.fit(self, X_train, X_val=X_val)
        self._rare_ext = ext
        self.resolved_["rare"] = ext.get_params()

    def _fit_edge(
        self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None
    ) -> None:
        """Fit EDGE embed density ranker (CIFAR/Agnews; Imdb off)."""
        self._edge_ext = None
        if not bool(self.edge):
            self.resolved_["edge"] = {"enabled": False}
            return
        ext = EdgeExtension(
            enabled=True,
            gamma_cv=float(self.edge_gamma_cv),
            gamma_nlp=float(self.edge_gamma_nlp),
            n_pairs=int(self.edge_pairs),
            epochs=int(self.edge_epochs),
            lr=float(self.edge_lr),
            hidden=int(self.edge_hidden),
            skip_imdb=bool(self.edge_skip_imdb),
            max_fit_n=int(self.edge_max_fit_n),
            whiten=bool(self.edge_whiten),
            whiten_dim=int(self.edge_whiten_dim),
            knn_view=bool(self.edge_knn_view),
            knn_k=int(self.edge_knn_k),
            advanced_allow_datasets=list(self.edge_advanced_allow_datasets),
            skip_fashion_advanced=bool(self.edge_skip_fashion_advanced),
            seed=int(self.seed),
            device=str(self.device),
        )
        ext.fit(self, X_train, X_val=X_val)
        self._edge_ext = ext
        self.resolved_["edge"] = ext.get_params()

    def _fit_nest(
        self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None
    ) -> None:
        """Fit NEST kNN soft-tail ranker (CIFAR/Agnews; Imdb off)."""
        self._nest_ext = None
        if not bool(self.nest):
            self.resolved_["nest"] = {"enabled": False}
            return
        ext = NestExtension(
            enabled=True,
            gamma_cv=float(self.nest_gamma_cv),
            gamma_nlp=float(self.nest_gamma_nlp),
            n_pairs=int(self.nest_pairs),
            epochs=int(self.nest_epochs),
            lr=float(self.nest_lr),
            hidden=int(self.nest_hidden),
            knn_k=int(self.nest_knn_k),
            skip_imdb=bool(self.nest_skip_imdb),
            max_fit_n=int(self.nest_max_fit_n),
            seed=int(self.seed),
            device=str(self.device),
        )
        ext.fit(self, X_train, X_val=X_val)
        self._nest_ext = ext
        self.resolved_["nest"] = ext.get_params()

    def _fit_extra_ensemble_nets(
        self,
        X_fit: np.ndarray,
        *,
        X_val: np.ndarray,
        hp: Dict[str, Any],
        use_ix: bool,
        d: int,
        raw_d: int,
        max_epochs: int,
        patience: int,
        req_bs: int,
    ) -> None:
        """Train additional nets with seed offsets; store in ``_ensemble_nets``."""
        assert self.net is not None
        self._ensemble_nets = []
        use_contrast = (
            self.is_semi_
            and float(self.contrastive_weight) > 1e-8
            and not self._is_nlp_modality(raw_d)
        )
        use_amp = self.device.type == "cuda"
        for h in range(1, int(self.ensemble_heads)):
            torch.manual_seed(self.seed + 997 * h)
            np.random.seed(self.seed + 997 * h)
            net_h = AxionNet(
                d_in=d,
                hidden=hp["hidden"],
                latent=hp["latent"],
                depth=hp["depth"],
                dropout=hp["dropout"],
                use_interaction=use_ix,
                interaction_rank=self.interaction_rank,
            ).to(self.device)
            opt = torch.optim.AdamW(
                net_h.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            try:
                scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            except (TypeError, AttributeError):
                scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            fit_ds = TensorDataset(torch.from_numpy(np.asarray(X_fit, dtype=np.float32)))
            bs = min(req_bs, max(8, len(X_fit)))
            loader = DataLoader(
                fit_ds,
                batch_size=bs,
                shuffle=True,
                drop_last=False,
                **self._loader_kwargs(),
            )
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed + 997 * h)
            best_state = None
            best_val = float("inf")
            stale = 0
            # Slightly shorter for extra heads to control wall time
            ep_cap = max(20, int(round(max_epochs * 0.85)))
            pat = max(8, int(round(patience * 0.85)))
            for epoch in range(ep_cap):
                net_h.train()
                last_loss = 0.0
                if self.is_semi_ and self.semi_loss_curriculum and ep_cap > 1:
                    t = float(epoch) / float(max(1, ep_cap - 1))
                    hard_factor = 0.85 + 0.30 * t
                else:
                    hard_factor = 1.0
                rates_epoch = tuple(
                    float(max(0.05, min(0.7, r * hard_factor))) for r in hp["mask_rates"]
                )
                for (xb,) in loader:
                    xb = xb.to(self.device, non_blocking=True)
                    mask = self._sample_masks(
                        xb.shape[0],
                        d,
                        rates=rates_epoch,
                        generator=gen,
                        device=self.device,
                        block_prob=float(self.block_prob),
                    )
                    opt.zero_grad(set_to_none=True)
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            mu, log_var, z1 = net_h(xb, mask)
                        loss = gaussian_nll(xb, mu.float(), log_var.float(), mask).mean()
                        if use_contrast:
                            mask2 = self._sample_masks(
                                xb.shape[0],
                                d,
                                rates=rates_epoch,
                                generator=gen,
                                device=self.device,
                                block_prob=float(self.block_prob),
                            )
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                _mu2, _lv2, z2 = net_h(xb, mask2)
                            z1n = torch.nn.functional.normalize(z1.float(), dim=-1)
                            z2n = torch.nn.functional.normalize(z2.float(), dim=-1)
                            cos = (z1n * z2n).sum(dim=-1)
                            loss = loss + float(self.contrastive_weight) * (
                                1.0 - cos
                            ).mean()
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(net_h.parameters(), 5.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        mu, log_var, z1 = net_h(xb, mask)
                        loss = gaussian_nll(xb, mu, log_var, mask).mean()
                        if use_contrast:
                            mask2 = self._sample_masks(
                                xb.shape[0],
                                d,
                                rates=rates_epoch,
                                generator=gen,
                                device=self.device,
                                block_prob=float(self.block_prob),
                            )
                            _mu2, _lv2, z2 = net_h(xb, mask2)
                            z1n = torch.nn.functional.normalize(z1, dim=-1)
                            z2n = torch.nn.functional.normalize(z2, dim=-1)
                            cos = (z1n * z2n).sum(dim=-1)
                            loss = loss + float(self.contrastive_weight) * (
                                1.0 - cos
                            ).mean()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net_h.parameters(), 5.0)
                        opt.step()
                    last_loss = float(loss.detach().cpu().item())
                if len(X_val) > 0:
                    # Quick val with temporary net swap
                    prev = self.net
                    self.net = net_h
                    val_loss = self._eval_nll(X_val, hp["mask_rates"], seed_offset=epoch + 50 * h)
                    self.net = prev
                else:
                    val_loss = last_loss
                if val_loss + 1e-6 < best_val:
                    best_val = val_loss
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in net_h.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1
                    if stale >= pat:
                        break
            if best_state is not None:
                net_h.load_state_dict(best_state)
            net_h.eval()
            self._ensemble_nets.append(net_h)
        self.resolved_["n_ensemble_nets"] = 1 + len(self._ensemble_nets)

    def _fit_bank_weights(self, X_val: np.ndarray) -> None:
        """Learn MCS bank weights on val normals: softmin of bank means."""
        X_val = np.asarray(X_val, dtype=np.float32)
        if len(X_val) > 2000:
            rng = np.random.RandomState(self.seed + 13)
            X_val = X_val[rng.choice(len(X_val), size=2000, replace=False)]
        rate_banks = self._rate_banks()
        means = []
        primary = self.net
        assert primary is not None
        for bi, bank in enumerate(rate_banks):
            # Mean hybrid score on normals (lower = better fit)
            acc = self._mcs_single_bank(X_val, bank=bank, seed_offset=200 + bi)
            means.append(float(np.mean(acc)))
        means_a = np.asarray(means, dtype=np.float64)
        # softmin: prefer banks that score normals lower
        z = -(means_a - means_a.mean()) / (means_a.std() + 1e-8)
        z = z - z.max()
        w = np.exp(z)
        w = w / w.sum()
        self.bank_weights_ = w.astype(np.float64)
        self.resolved_["bank_weights"] = [float(x) for x in self.bank_weights_]

    @torch.no_grad()
    def _mcs_single_bank(
        self, X: np.ndarray, *, bank: Sequence[float], seed_offset: int = 0
    ) -> np.ndarray:
        assert self.net is not None
        self.net.eval()
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        self._ensure_hw(d)
        k = max(2, int(self.resolved_.get("score_k", 8)) // 2)
        xb = torch.from_numpy(X).to(self.device)
        acc = torch.zeros(n, device=self.device, dtype=torch.float32)
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + 12345 + int(seed_offset))
        mae_w = float(self.active_mae_weight_)
        nll_w = float(self.active_nll_weight_)
        bs = self._score_chunk_bs(n)
        for _ in range(k):
            for i in range(0, n, bs):
                chunk = xb[i : i + bs]
                mask = self._sample_masks(
                    chunk.shape[0],
                    d,
                    rates=bank,
                    generator=g,
                    device=self.device,
                    block_prob=float(self.block_prob),
                )
                mu, log_var, _ = self.net(chunk, mask)
                nll = gaussian_nll(chunk, mu, log_var, mask)
                mae = ((chunk - mu).abs() * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(
                    1.0
                )
                hybrid = mae_w * mae + nll_w * nll
                acc[i : i + chunk.shape[0]] += hybrid
        return (acc / float(max(1, k))).detach().cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def _recon_error(self, X: np.ndarray) -> np.ndarray:
        """Visible-path |x-μ| mean (mask=0) for contamination ranking."""
        assert self.net is not None
        self.net.eval()
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        self._ensure_hw(d)
        xb = torch.from_numpy(X).to(self.device)
        out = torch.zeros(n, device=self.device, dtype=torch.float32)
        bs = self._score_chunk_bs(n)
        for i in range(0, n, bs):
            chunk = xb[i : i + bs]
            mask = torch.zeros_like(chunk)
            mu, _log_var, _z = self.net(chunk, mask)
            out[i : i + chunk.shape[0]] = (chunk - mu).abs().mean(dim=-1)
        return out.detach().cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def _eval_nll(self, X: np.ndarray, rates: Sequence[float], seed_offset: int = 0) -> float:
        assert self.net is not None
        self.net.eval()
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        self._ensure_hw(d)
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + 10_000 + seed_offset)
        xb = torch.from_numpy(X).to(self.device)
        losses = []
        bs = self._score_chunk_bs(n)
        for i in range(0, n, bs):
            chunk = xb[i : i + bs]
            mask = self._sample_masks(
                chunk.shape[0],
                d,
                rates=rates,
                generator=g,
                device=self.device,
                block_prob=float(self.block_prob),
            )
            mu, log_var, _ = self.net(chunk, mask)
            losses.append(gaussian_nll(chunk, mu, log_var, mask).mean().item())
        return float(np.mean(losses))

    @torch.no_grad()
    def _encode_visible(self, X: np.ndarray) -> np.ndarray:
        assert self.net is not None
        self.net.eval()
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        self._ensure_hw(d)
        xb = torch.from_numpy(X).to(self.device)
        zs = []
        bs = self._score_chunk_bs(n)
        for i in range(0, n, bs):
            chunk = xb[i : i + bs]
            mask = torch.zeros_like(chunk)
            _, z = self.net.encoder(chunk, mask)
            zs.append(z)
        return torch.cat(zs, dim=0).detach().cpu().numpy()

    def _fit_latch(self, X: np.ndarray) -> None:
        z = self._encode_visible(X)
        # Huge unsup: median / MAD center resists rare contamination in train
        use_robust = (
            (not bool(getattr(self, "is_semi_", False)))
            and int(z.shape[0]) > 10000
            and (
                bool(self.robust_latch_unsup_huge)
                or abs(float(self.active_x_latch_alpha_)) > 1e-12
            )
        )
        if use_robust:
            self.z_mean_ = np.median(z, axis=0).astype(np.float64)
            mad = np.median(np.abs(z - self.z_mean_), axis=0).astype(np.float64)
            scale = np.maximum(1.4826 * mad, 1e-3)
            self.z_inv_var_ = 1.0 / np.maximum(scale ** 2, 1e-6)
        else:
            self.z_mean_ = z.mean(axis=0).astype(np.float64)
            var = z.var(axis=0).astype(np.float64)
            # Ledoit-Wolf-style diagonal shrinkage for low-d semi (glass/cardio)
            if (
                bool(self.latch_shrinkage)
                and bool(self.is_semi_)
                and int(z.shape[1]) <= 128
            ):
                target = float(np.mean(var))
                n_z = float(max(2, z.shape[0]))
                # shrink toward mean variance; more shrink when n small
                shrink = float(min(0.85, max(0.15, 32.0 / n_z)))
                var = (1.0 - shrink) * var + shrink * target
                self.resolved_["latch_shrink"] = shrink
            self.z_inv_var_ = 1.0 / np.maximum(var, 1e-6)

    def _fit_c_hedge(self, X: np.ndarray) -> None:
        """Classical residual energy: low-rank PCA residual in model space (d<400)."""
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        k = int(max(2, min(16, d // 2, n - 1)))
        mean = X.mean(axis=0)
        Xc = X - mean
        try:
            # Economy SVD
            _, s, vt = np.linalg.svd(Xc, full_matrices=False)
            comps = vt[:k]
            scales = 1.0 / np.maximum(s[:k] / np.sqrt(max(1, n - 1)), 1e-6)
        except np.linalg.LinAlgError:
            self.c_pca_mean_ = None
            self.c_pca_components_ = None
            self.c_pca_scales_ = None
            return
        self.c_pca_mean_ = mean.astype(np.float64)
        self.c_pca_components_ = comps.astype(np.float64)
        self.c_pca_scales_ = scales.astype(np.float64)

    def _c_hedge_score(self, X: np.ndarray) -> np.ndarray:
        if self.c_pca_mean_ is None or self.c_pca_components_ is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.c_pca_mean_
        proj = Xc @ self.c_pca_components_.T
        recon = proj @ self.c_pca_components_
        resid = Xc - recon
        return np.sum(resid ** 2, axis=1)

    def _fit_x_latch(self, X: np.ndarray) -> None:
        """Feature-space Mahalanobis (diagonal) for rare low-d huge unsup."""
        self.x_mean_ = None
        self.x_inv_var_ = None
        if abs(float(self.active_x_latch_alpha_)) < 1e-12:
            return
        X = np.asarray(X, dtype=np.float64)
        # Robust center: median/MAD resists rare contamination
        self.x_mean_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.x_mean_), axis=0)
        scale = np.maximum(1.4826 * mad, 1e-3)
        self.x_inv_var_ = 1.0 / np.maximum(scale ** 2, 1e-6)

    def _latch_score(self, X: np.ndarray) -> np.ndarray:
        if self.z_mean_ is None or self.z_inv_var_ is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        z = self._encode_visible(X)
        diff = z - self.z_mean_
        return np.sum((diff ** 2) * self.z_inv_var_, axis=1)

    def _x_latch_score(self, X: np.ndarray) -> np.ndarray:
        if self.x_mean_ is None or self.x_inv_var_ is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        diff = X - self.x_mean_
        return np.sum((diff ** 2) * self.x_inv_var_, axis=1)

    @torch.no_grad()
    def _mcs_scores(
        self,
        X: np.ndarray,
        *,
        return_extra: bool = False,
        seed_offset: int = 0,
    ):
        """Raw MCS: hybrid MAE+NLL over K masks × rate banks (PCA space if used).

        GPU-resident accumulation + optional K-pack. When return_extra=True also
        returns (disagree, hpd_unc) views.
        """
        assert self.net is not None
        self.net.eval()
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        self._ensure_hw(d)
        k = int(self.resolved_.get("score_k", 8))
        rate_banks = self._rate_banks()
        n_banks = len(rate_banks)
        use_bw = bool(self.is_semi_) and bool(self.bank_weight_semi)
        if self.bank_weights_ is not None and len(self.bank_weights_) == n_banks:
            bw = np.asarray(self.bank_weights_, dtype=np.float64)
        elif use_bw:
            bw = np.ones(n_banks, dtype=np.float64)
            bw[0] = 1.2
            if n_banks > 1:
                bw[1] = 1.0
            if n_banks > 2:
                bw[2] = 0.8
            bw = bw / bw.sum()
        else:
            bw = np.ones(n_banks, dtype=np.float64) / float(n_banks)

        xb = torch.from_numpy(X).to(self.device)
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + 12345 + int(seed_offset))

        bs = self._mcs_chunk_bs(n)
        pack_cap = 1
        if bool(self.mcs_pack_masks) and self.device.type == "cuda":
            # Pack multiple masks per forward when VRAM allows
            pack_cap = int(max(1, min(k, max(1, self._score_bs // max(bs, 1)))))
            pack_cap = int(min(8, pack_cap))
        n_passes = 0
        mae_w = float(self.active_mae_weight_)
        nll_w = float(self.active_nll_weight_)

        def _forward_pack(chunk: torch.Tensor, bank: Sequence[float], P: int):
            B = int(chunk.shape[0])
            if P <= 1:
                mask = self._sample_masks(
                    B, d, rates=bank, generator=g, device=self.device,
                    block_prob=float(self.block_prob),
                )
                mu, log_var, _ = self.net(chunk, mask)
                nll = gaussian_nll(chunk, mu, log_var, mask)
                mae = ((chunk - mu).abs() * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
                hybrid = mae_w * mae + nll_w * nll
                lv = torch.clamp(log_var, -8.0, 8.0)
                unc = (torch.exp(lv) * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
                return hybrid, unc
            x_rep = chunk.repeat(P, 1)
            mask = self._sample_masks(
                B * P, d, rates=bank, generator=g, device=self.device,
                block_prob=float(self.block_prob),
            )
            mu, log_var, _ = self.net(x_rep, mask)
            nll = gaussian_nll(x_rep, mu, log_var, mask)
            mae = ((x_rep - mu).abs() * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
            hybrid = (mae_w * mae + nll_w * nll).view(P, B).sum(dim=0)
            lv = torch.clamp(log_var, -8.0, 8.0)
            unc = ((torch.exp(lv) * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)).view(P, B).sum(dim=0)
            return hybrid, unc

        # g5-faithful path: float64 numpy accumulate, K×chunk order (pack off)
        if pack_cap <= 1:
            acc_np = np.zeros(n, dtype=np.float64)
            hpd_np = np.zeros(n, dtype=np.float64)
            bank_hybrids_np: List[np.ndarray] = []
            n_chunks = int((n + bs - 1) // max(1, bs))
            score_total = int(n_banks) * int(k) * max(1, n_chunks)
            score_bar = ProgressBar(
                score_total,
                desc="score:mcs",
                leave=True,
                unit="chk",
                position=3,
            )
            try:
                for bi, bank in enumerate(rate_banks):
                    bank_acc = np.zeros(n, dtype=np.float64)
                    for _ in range(int(k)):
                        n_passes += 1
                        for i in range(0, n, bs):
                            chunk = xb[i : i + bs]
                            hybrid, unc = _forward_pack(chunk, bank, 1)
                            sl = slice(i, i + chunk.shape[0])
                            bank_acc[sl] += hybrid.detach().cpu().numpy().astype(np.float64)
                            hpd_np[sl] += unc.detach().cpu().numpy().astype(np.float64)
                            score_bar.update(1)
                    bank_acc = bank_acc / float(max(1, k))
                    bank_hybrids_np.append(bank_acc)
                    acc_np = acc_np + float(bw[bi]) * bank_acc
            finally:
                score_bar.close()
            if len(bank_hybrids_np) >= 2:
                disagree_acc = np.std(np.stack(bank_hybrids_np, axis=0), axis=0)
            else:
                disagree_acc = np.zeros(n, dtype=np.float64)
            hpd_np = hpd_np / float(max(1, n_passes))
        else:
            acc = torch.zeros(n, device=self.device, dtype=torch.float32)
            hpd_acc = torch.zeros(n, device=self.device, dtype=torch.float32)
            bank_hybrids: List[torch.Tensor] = []
            n_chunks = int((n + bs - 1) // max(1, bs))
            # pack path: approximate progress as banks × chunks (inner K packed)
            score_total = int(n_banks) * max(1, n_chunks)
            score_bar = ProgressBar(
                score_total,
                desc="score:mcs_pack",
                leave=True,
                unit="chk",
                position=3,
            )
            try:
                for bi, bank in enumerate(rate_banks):
                    bank_acc = torch.zeros(n, device=self.device, dtype=torch.float32)
                    # Pack K masks per chunk (speed); may shift RNG vs g5
                    for i in range(0, n, bs):
                        chunk = xb[i : i + bs]
                        left = int(k)
                        while left > 0:
                            P = int(min(pack_cap, left))
                            try:
                                hybrid, unc = _forward_pack(chunk, bank, P)
                            except RuntimeError as e:
                                if "out of memory" in str(e).lower() and self.device.type == "cuda":
                                    torch.cuda.empty_cache()
                                    if self._shrink_score_batch_on_oom():
                                        bs = self._mcs_chunk_bs(n)
                                        pack_cap = int(max(1, min(k, max(1, self._score_bs // max(bs, 1)))))
                                    hybrid, unc = _forward_pack(chunk, bank, 1)
                                    for _extra in range(max(0, P - 1)):
                                        h2, u2 = _forward_pack(chunk, bank, 1)
                                        hybrid = hybrid + h2
                                        unc = unc + u2
                                    pack_cap = 1
                                else:
                                    raise
                            bank_acc[i : i + chunk.shape[0]] += hybrid
                            hpd_acc[i : i + chunk.shape[0]] += unc
                            left -= P
                        score_bar.update(1)
                    n_passes += int(k)
                    bank_acc = bank_acc / float(max(1, k))
                    bank_hybrids.append(bank_acc)
                    acc = acc + float(bw[bi]) * bank_acc
            finally:
                score_bar.close()

            if len(bank_hybrids) >= 2:
                stacked = torch.stack(bank_hybrids, dim=0)
                disagree_t = stacked.std(dim=0)
            else:
                disagree_t = torch.zeros(n, device=self.device, dtype=torch.float32)
            hpd_t = hpd_acc / float(max(1, n_passes))
            acc_np = acc.detach().cpu().numpy().astype(np.float64)
            disagree_acc = disagree_t.detach().cpu().numpy().astype(np.float64)
            hpd_np = hpd_t.detach().cpu().numpy().astype(np.float64)

        # Average MCS across true ensemble nets (primary already computed)
        if self._ensemble_nets and not getattr(self, "_mcs_ensembling_", False):
            self._mcs_ensembling_ = True
            primary = self.net
            try:
                parts_acc = [acc_np]
                parts_dis = [disagree_acc]
                parts_hpd = [hpd_np]
                for net_h in self._ensemble_nets:
                    self.net = net_h
                    if return_extra:
                        a, dlt, h = self._mcs_scores(
                            X, return_extra=True, seed_offset=seed_offset
                        )
                        parts_acc.append(a)
                        parts_dis.append(dlt)
                        parts_hpd.append(h)
                    else:
                        parts_acc.append(
                            self._mcs_scores(X, return_extra=False, seed_offset=seed_offset)
                        )
                acc_np = np.mean(np.stack(parts_acc, axis=0), axis=0)
                if return_extra:
                    disagree_acc = np.mean(np.stack(parts_dis, axis=0), axis=0)
                    hpd_np = np.mean(np.stack(parts_hpd, axis=0), axis=0)
            finally:
                self.net = primary
                self._mcs_ensembling_ = False

        if return_extra:
            return acc_np, disagree_acc, hpd_np
        return acc_np

    def _fit_score_anchors(
        self,
        X: np.ndarray,
        X_raw: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train-only mean/std for MCS, LATCH, HEDGE, and semi score views."""
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        idx = None
        if n > 8000:
            rng = np.random.RandomState(self.seed + 7)
            idx = rng.choice(n, size=8000, replace=False)
            X = X[idx]

        extra = abs(float(self.active_disagree_alpha_)) > 1e-12 or abs(
            float(self.active_hpd_alpha_)
        ) > 1e-12
        if extra:
            mcs, disagree, hpd = self._mcs_scores(X, return_extra=True)
        else:
            mcs = self._mcs_scores(X)
            disagree = np.zeros(len(X), dtype=np.float64)
            hpd = np.zeros(len(X), dtype=np.float64)

        # Multi-seed MCS bagging only when no true ensemble nets
        if int(self.ensemble_heads) > 1 and not self._ensemble_nets:
            parts = [mcs]
            for h in range(1, int(self.ensemble_heads)):
                parts.append(self._mcs_scores(X, seed_offset=1000 * h))
            mcs = np.mean(np.stack(parts, axis=0), axis=0)

        latch = self._latch_score(X)
        self.mcs_mean_ = float(mcs.mean())
        self.mcs_std_ = float(max(float(mcs.std()), 1e-8))
        self.latch_score_mean_ = float(latch.mean())
        self.latch_score_std_ = float(max(float(latch.std()), 1e-8))

        self.x_latch_mean_ = None
        self.x_latch_std_ = None
        if abs(float(self.active_x_latch_alpha_)) > 1e-12:
            xl = self._x_latch_score(X)
            self.x_latch_mean_ = float(xl.mean())
            self.x_latch_std_ = float(max(float(xl.std()), 1e-8))

        self.hedge_mean_ = None
        self.hedge_std_ = None
        if (
            self.used_pca_
            and abs(float(self.active_hedge_alpha_)) > 1e-12
            and X_raw is not None
        ):
            Xr = np.asarray(X_raw, dtype=np.float32)
            if idx is not None:
                Xr = Xr[idx]
            elif Xr.shape[0] > 8000:
                rng = np.random.RandomState(self.seed + 7)
                Xr = Xr[rng.choice(Xr.shape[0], size=8000, replace=False)]
            hedge = self._pca_residual(Xr)
            self.hedge_mean_ = float(hedge.mean())
            self.hedge_std_ = float(max(float(hedge.std()), 1e-8))

        self.vis_mean_ = None
        self.vis_std_ = None
        if abs(float(self.active_vis_alpha_)) > 1e-12:
            vis = self._recon_error(X)
            self.vis_mean_ = float(vis.mean())
            self.vis_std_ = float(max(float(vis.std()), 1e-8))

        self.disagree_mean_ = None
        self.disagree_std_ = None
        if abs(float(self.active_disagree_alpha_)) > 1e-12:
            self.disagree_mean_ = float(disagree.mean())
            self.disagree_std_ = float(max(float(disagree.std()), 1e-8))

        self.hpd_mean_ = None
        self.hpd_std_ = None
        if abs(float(self.active_hpd_alpha_)) > 1e-12:
            self.hpd_mean_ = float(hpd.mean())
            self.hpd_std_ = float(max(float(hpd.std()), 1e-8))

        self.c_hedge_mean_ = None
        self.c_hedge_std_ = None
        if abs(float(self.active_c_hedge_alpha_)) > 1e-12:
            ch = self._c_hedge_score(X)
            self.c_hedge_mean_ = float(ch.mean())
            self.c_hedge_std_ = float(max(float(ch.std()), 1e-8))

        # CALIX-lite: on val normals, downweight aux views that are anti-aligned with MCS
        self.calix_scale_ = {
            "latch": 1.0,
            "vis": 1.0,
            "disagree": 1.0,
            "hpd": 1.0,
            "c_hedge": 1.0,
            "hedge": 1.0,
        }
        if (
            bool(self.calix_semi)
            and bool(self.is_semi_)
            and X_val is not None
            and len(X_val) >= 8
        ):
            Xv = np.asarray(X_val, dtype=np.float32)
            if len(Xv) > 2000:
                rng = np.random.RandomState(self.seed + 11)
                Xv = Xv[rng.choice(len(Xv), size=2000, replace=False)]
            mcs_v = self._mcs_scores(Xv)
            views = {
                "latch": self._latch_score(Xv),
                "vis": self._recon_error(Xv) if abs(self.active_vis_alpha_) > 1e-12 else None,
                "c_hedge": self._c_hedge_score(Xv)
                if abs(self.active_c_hedge_alpha_) > 1e-12
                else None,
            }
            if abs(self.active_disagree_alpha_) > 1e-12 or abs(self.active_hpd_alpha_) > 1e-12:
                _m, d_v, h_v = self._mcs_scores(Xv, return_extra=True)
                views["disagree"] = d_v
                views["hpd"] = h_v
            for name, arr in views.items():
                if arr is None:
                    continue
                if float(np.std(arr)) < 1e-8 or float(np.std(mcs_v)) < 1e-8:
                    self.calix_scale_[name] = 0.5
                    continue
                corr = float(np.corrcoef(mcs_v, arr)[0, 1])
                # Keep positively aligned views; soft-downweight negative
                self.calix_scale_[name] = float(np.clip(0.35 + 0.65 * max(corr, 0.0), 0.2, 1.0))
            self.resolved_["calix_scale"] = dict(self.calix_scale_)

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        """MCS + fused views with train-anchored z-norm."""
        if self.net is None:
            raise RuntimeError("AxionModel not fitted")
        hint = str(getattr(self, "dataset_hint", "") or "axion")
        X_raw = np.asarray(X, dtype=np.float32)
        step_log(f"SCORE begin dataset={hint} n={X_raw.shape[0]} d={X_raw.shape[1]}")
        hedge = None
        beta = float(self.active_hedge_alpha_)
        if self.used_pca_ and abs(beta) > 1e-12:
            hedge = self._pca_residual(X_raw)
        Xp = self._maybe_project(X_raw)

        spear_on = self._spear_ext is not None and self._spear_ext.active(self)
        rare_on = self._rare_ext is not None and self._rare_ext.active(self)
        edge_on = self._edge_ext is not None and self._edge_ext.active(self)
        nest_on = self._nest_ext is not None and self._nest_ext.active(self)
        need_extra = (
            abs(float(self.active_disagree_alpha_)) > 1e-12
            or abs(float(self.active_hpd_alpha_)) > 1e-12
            or spear_on
            or rare_on
        )
        if need_extra:
            mcs, disagree, hpd = self._mcs_scores(Xp, return_extra=True)
        else:
            mcs = self._mcs_scores(Xp)
            disagree = None
            hpd = None
        if int(self.ensemble_heads) > 1 and not self._ensemble_nets:
            parts = [mcs]
            for h in range(1, int(self.ensemble_heads)):
                parts.append(self._mcs_scores(Xp, seed_offset=1000 * h))
            mcs = np.mean(np.stack(parts, axis=0), axis=0)

        latch = self._latch_score(Xp)
        alpha = float(self.active_latch_alpha_) * float(self.calix_scale_.get("latch", 1.0))
        gamma = float(self.active_x_latch_alpha_)
        vis_a = float(self.active_vis_alpha_) * float(self.calix_scale_.get("vis", 1.0))
        dis_a = float(self.active_disagree_alpha_) * float(
            self.calix_scale_.get("disagree", 1.0)
        )
        hpd_a = float(self.active_hpd_alpha_) * float(self.calix_scale_.get("hpd", 1.0))
        c_a = float(self.active_c_hedge_alpha_) * float(
            self.calix_scale_.get("c_hedge", 1.0)
        )
        beta = beta * float(self.calix_scale_.get("hedge", 1.0))

        vis = None
        ch = None
        if abs(vis_a) >= 1e-12 or spear_on or rare_on or edge_on or nest_on:
            vis = self._recon_error(Xp)
        if abs(c_a) >= 1e-12 or spear_on or rare_on:
            ch = self._c_hedge_score(Xp)

        def _fuse_rank_exts(score_arr: np.ndarray) -> np.ndarray:
            out = score_arr
            if spear_on or rare_on:
                dis_v = (
                    disagree
                    if disagree is not None
                    else np.zeros(len(Xp), dtype=np.float64)
                )
                hpd_v = (
                    hpd if hpd is not None else np.zeros(len(Xp), dtype=np.float64)
                )
                vis_v = (
                    vis if vis is not None else np.zeros(len(Xp), dtype=np.float64)
                )
                ch_v = ch if ch is not None else np.zeros(len(Xp), dtype=np.float64)
                feats = views_from_parts(mcs, latch, vis_v, dis_v, hpd_v, ch_v)
                if spear_on:
                    sp = self._spear_ext.z_normed_delta_from_views(feats)
                    out = out + float(self._spear_ext.gamma) * sp
                if rare_on:
                    rp = self._rare_ext.z_normed_delta_from_views(feats)
                    out = out + float(self._rare_ext.active_gamma_) * rp
            if edge_on:
                eg = self._edge_ext.z_normed_delta(self, Xp, mcs=mcs)
                out = out + float(self._edge_ext.gamma) * eg
            if nest_on:
                ng = self._nest_ext.z_normed_delta(self, Xp, mcs=mcs)
                out = out + float(self._nest_ext.gamma) * ng
            if self._ecod_view is not None and self._ecod_view.active():
                out = out + float(self._ecod_view.alpha) * self._ecod_view.z_normed(Xp)
            if self._copod_view is not None and self._copod_view.active():
                out = out + float(self._copod_view.alpha) * self._copod_view.z_normed(Xp)
            if self._goad_ext is not None and self._goad_ext.active():
                out = out + float(self._goad_ext.alpha) * self._goad_ext.z_normed(Xp)
            if self._cv_knn_view is not None and self._cv_knn_view.active():
                # Use raw (pre-PCA) space for memory distances when available
                X_knn = X_raw if self.used_pca_ else Xp
                out = out + float(self._cv_knn_view.alpha) * self._cv_knn_view.z_normed(
                    X_knn
                )
            if self._cv_lof_view is not None and self._cv_lof_view.active():
                X_cv = X_raw if self.used_pca_ else Xp
                cz = self._cv_lof_view.z_normed(X_cv)
                mode = str(getattr(self._cv_lof_view, "mode", "replace") or "replace")
                a = float(self._cv_lof_view.alpha)
                if mode == "replace":
                    out = cz
                elif mode == "blend":
                    w = float(min(1.0, max(0.0, a)))
                    out = (1.0 - w) * out + w * cz
                else:
                    out = out + a * cz
            if self._nlp_lof_view is not None and self._nlp_lof_view.active():
                X_nlp = X_raw if self.used_pca_ else Xp
                lz = self._nlp_lof_view.z_normed(X_nlp)
                if str(getattr(self._nlp_lof_view, "fuse_mode", "add") or "add") != "add":
                    lz = self._nlp_lof_view.transform(lz, mcs)
                mode = str(getattr(self._nlp_lof_view, "mode", "fuse") or "fuse")
                a = float(self._nlp_lof_view.alpha)
                if mode == "replace":
                    out = lz
                elif mode == "blend":
                    w = float(min(1.0, max(0.0, a)))
                    out = (1.0 - w) * out + w * lz
                else:
                    out = out + a * lz
            return out

        if self.mcs_mean_ is not None and self.mcs_std_ is not None:
            mcs_n = (mcs - self.mcs_mean_) / (self.mcs_std_ + 1e-8)
            score = mcs_n
            if abs(alpha) >= 1e-12:
                latch_n = (latch - float(self.latch_score_mean_ or 0.0)) / (
                    float(self.latch_score_std_ or 1.0) + 1e-8
                )
                score = score + alpha * latch_n
            if abs(gamma) >= 1e-12 and self.x_latch_mean_ is not None:
                xl = self._x_latch_score(Xp)
                xl_n = (xl - float(self.x_latch_mean_)) / (
                    float(self.x_latch_std_ or 1.0) + 1e-8
                )
                score = score + gamma * xl_n
            if hedge is not None and self.hedge_mean_ is not None and self.hedge_std_ is not None:
                hedge_n = (hedge - self.hedge_mean_) / (self.hedge_std_ + 1e-8)
                score = score + beta * hedge_n
            if abs(vis_a) >= 1e-12 and self.vis_mean_ is not None and vis is not None:
                vis_n = (vis - float(self.vis_mean_)) / (float(self.vis_std_ or 1.0) + 1e-8)
                score = score + vis_a * vis_n
            if abs(dis_a) >= 1e-12 and disagree is not None and self.disagree_mean_ is not None:
                d_n = (disagree - float(self.disagree_mean_)) / (
                    float(self.disagree_std_ or 1.0) + 1e-8
                )
                score = score + dis_a * d_n
            if abs(hpd_a) >= 1e-12 and hpd is not None and self.hpd_mean_ is not None:
                h_n = (hpd - float(self.hpd_mean_)) / (float(self.hpd_std_ or 1.0) + 1e-8)
                score = score + hpd_a * h_n
            if abs(c_a) >= 1e-12 and self.c_hedge_mean_ is not None and ch is not None:
                ch_n = (ch - float(self.c_hedge_mean_)) / (
                    float(self.c_hedge_std_ or 1.0) + 1e-8
                )
                score = score + c_a * ch_n
            score = _fuse_rank_exts(score)
            step_log(
                f"SCORE done dataset={hint} n={len(score)} "
                f"spear={int(spear_on)} rare={int(rare_on)} edge={int(edge_on)} nest={int(nest_on)}"
            )
            return score.astype(np.float64)

        # Fallback: batch z-norm (should be rare — anchors fitted in fit())
        if latch.std() > 1e-8 and mcs.std() > 1e-8:
            latch_n = (latch - latch.mean()) / (latch.std() + 1e-8)
            mcs_n = (mcs - mcs.mean()) / (mcs.std() + 1e-8)
            score = mcs_n + alpha * latch_n
        else:
            score = mcs + alpha * latch
        if hedge is not None and hedge.std() > 1e-8:
            hedge_n = (hedge - hedge.mean()) / (hedge.std() + 1e-8)
            score = score + beta * hedge_n
        score = _fuse_rank_exts(score)
        step_log(f"SCORE done (fallback) dataset={hint} n={len(score)}")
        return score.astype(np.float64)

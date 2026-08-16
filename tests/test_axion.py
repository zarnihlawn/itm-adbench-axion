"""AXION model unit tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.models import build_model
from axion.models.mcb import sample_masks, scale_hparams, soften_mask_rates
from axion.train.experiment import run_one_array


def test_scale_hparams_branches():
    small = scale_hparams(100, 10)
    big_d = scale_hparams(1000, 512)
    assert small["hidden"] <= big_d["hidden"]
    assert big_d["latent"] >= 64
    assert big_d["score_k"] >= 12
    assert big_d["high_mask_cap"] <= 0.6
    assert big_d["dropout"] <= 0.05


def test_scale_hparams_post_pca_mid_d():
    # After PCA, model d~128 with raw_d=768 should use mid-d path + extra score_k
    mid = scale_hparams(2000, 128, raw_d=768)
    assert mid["hidden"] == 320
    assert mid["score_k"] >= 20


def test_scale_hparams_tiny_n_glass():
    tiny = scale_hparams(150, 7)
    assert tiny["depth"] >= 3
    assert tiny["score_k"] >= 36
    assert tiny["hidden"] >= 128
    assert tiny.get("tiny_n") is True
    # glass full-set n=214 must stay on tiny MCS path (threshold 250)
    glass = scale_hparams(214, 7, tiny_n_threshold=250)
    assert glass.get("tiny_n") is True
    assert glass["score_k"] >= 36
    mid = scale_hparams(214, 7, tiny_n_threshold=200)
    assert mid.get("tiny_n") is False


def test_scale_hparams_huge_unsup():
    huge = scale_hparams(20000, 10, semi=False)
    assert huge["score_k"] >= 40
    assert huge.get("huge_n") is True
    huge_semi = scale_hparams(20000, 10, semi=True)
    # Semi PR attack: huge-n semi score_k floor 40 (then *1.25, cap 48)
    assert huge_semi["score_k"] >= 40


def test_scale_hparams_semi_softer():
    base = scale_hparams(1000, 20, semi=False)
    semi = scale_hparams(1000, 20, semi=True)
    assert max(semi["mask_rates"]) <= max(base["mask_rates"])
    assert semi["score_k"] >= base["score_k"]
    assert soften_mask_rates((0.2, 0.4))[0] < 0.2


def test_sample_masks_shape():
    m = sample_masks(8, 16, rates=(0.2, 0.4))
    assert m.shape == (8, 16)
    assert ((m == 0) | (m == 1)).all()


def test_axion_fit_score_toy():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 8).astype(np.float32)
    y = np.zeros(120, dtype=np.int64)
    y[-15:] = 1
    X[-15:] += 3.0
    model = build_model("axion", epochs=15, patience=5, batch_size=32, seed=0)
    model.fit(X[:100])
    s = model.score(X)
    assert s.shape == (120,)
    assert np.all(np.isfinite(s))


def test_train_anchored_score_and_semi_mcs_primary():
    rng = np.random.RandomState(2)
    Xn = rng.randn(80, 8).astype(np.float32)
    Xa = rng.randn(20, 8).astype(np.float32) + 3.0
    X = np.vstack([Xn, Xa])
    y_all_normal = np.zeros(80, dtype=np.int64)

    model = build_model(
        "axion",
        epochs=12,
        patience=4,
        batch_size=32,
        seed=3,
        latch_alpha=0.4,
        latch_alpha_semi=0.0,
        latch_alpha_semi_tiny=0.0,
        mae_weight_semi=0.8,
        nll_weight_semi=0.2,
        vis_alpha_semi=0.0,
        disagree_alpha_semi=0.0,
        hpd_alpha_semi=0.0,
        c_hedge_alpha_semi=0.0,
        contrastive_weight=0.0,
        use_interaction=False,
        ensemble_heads=1,
    )
    model.fit(Xn, y_all_normal)
    assert model.is_semi_ is True
    assert model.used_pca_ is False
    assert model.mcs_mean_ is not None
    assert model.mcs_std_ is not None and model.mcs_std_ > 0
    assert abs(model.active_latch_alpha_) < 1e-9
    assert abs(model.active_mae_weight_ - 0.8) < 1e-6

    s = model.score(X)
    assert s.shape == (100,)
    assert float(s[80:].mean()) > float(s[:80].mean())

    y_mixed = np.zeros(100, dtype=np.int64)
    y_mixed[80:] = 1
    model2 = build_model(
        "axion",
        epochs=8,
        patience=3,
        batch_size=32,
        seed=4,
        latch_alpha=0.4,
        latch_alpha_semi=0.0,
        use_interaction=False,
        ensemble_heads=1,
    )
    model2.fit(X, y_mixed)
    assert model2.is_semi_ is False
    # n=100 < 200 → tiny-n unsup latch bump
    assert abs(model2.active_latch_alpha_ - 0.75) < 1e-9


def test_semi_latch_and_score_views():
    rng = np.random.RandomState(42)
    # Mid-n (≥200): score views stay on; tiny-n forces MCS-primary (views off)
    Xn = rng.randn(250, 10).astype(np.float32)
    model = build_model(
        "axion",
        epochs=10,
        patience=4,
        batch_size=32,
        seed=1,
        latch_alpha_semi=0.25,
        latch_alpha_semi_tiny=0.35,
        mae_weight_semi=0.9,
        nll_weight_semi=0.1,
        vis_alpha_semi=0.15,
        disagree_alpha_semi=0.10,
        hpd_alpha_semi=0.05,
        c_hedge_alpha_semi=0.20,
        bank_weight_semi=True,
        calix_semi=True,
        contrastive_weight=0.05,
        latch_shrinkage=True,
        use_interaction=True,
        ensemble_heads=2,
    )
    model.fit(Xn, np.zeros(250, dtype=np.int64))
    assert model.is_semi_ is True
    assert abs(model.active_latch_alpha_ - 0.25) < 1e-9  # mid-n uses latch_alpha_semi
    assert model.active_vis_alpha_ > 0
    assert model.active_c_hedge_alpha_ > 0
    assert model.vis_mean_ is not None
    assert model.c_hedge_mean_ is not None
    s = model.score(Xn)
    assert s.shape == (250,)
    assert np.all(np.isfinite(s))


def test_tiny_n_semi_mcs_primary_views_off():
    """Glass-scale: n<200 semi forces view alphas to 0 (g5 MCS-primary)."""
    rng = np.random.RandomState(42)
    Xn = rng.randn(90, 10).astype(np.float32)
    model = build_model(
        "axion",
        epochs=6,
        patience=3,
        batch_size=32,
        seed=1,
        latch_alpha_semi=0.25,
        latch_alpha_semi_tiny=0.0,
        vis_alpha_semi=0.20,
        disagree_alpha_semi=0.15,
        hpd_alpha_semi=0.05,
        c_hedge_alpha_semi=0.30,
        use_interaction=False,
        ensemble_heads=1,
        contrastive_weight=0.0,
    )
    model.fit(Xn, np.zeros(90, dtype=np.int64))
    assert model.is_semi_ is True
    assert abs(model.active_latch_alpha_) < 1e-12
    assert abs(model.active_vis_alpha_) < 1e-12
    assert abs(model.active_disagree_alpha_) < 1e-12
    assert abs(model.active_hpd_alpha_) < 1e-12
    assert abs(model.active_c_hedge_alpha_) < 1e-12
    s = model.score(Xn)
    assert s.shape == (90,)
    assert np.all(np.isfinite(s))


def test_agnews_nlp_hedge_vs_imdb():
    rng = np.random.RandomState(8)
    X = rng.randn(100, 768).astype(np.float32)
    y0 = np.zeros(100, dtype=np.int64)
    common = dict(
        epochs=5,
        patience=2,
        batch_size=32,
        seed=2,
        pca_dim=16,
        pca_threshold=400,
        hedge_alpha_semi_nlp=0.0,
        hedge_alpha_semi_nlp_multi=0.25,
        nlp_raw_d_min=700,
        use_interaction=False,
        ensemble_heads=1,
        contrastive_weight=0.0,
    )
    imdb = build_model("axion", dataset_hint="Imdb", **common)
    imdb.fit(X, y0)
    assert abs(imdb.active_hedge_alpha_) < 1e-12
    ag = build_model("axion", dataset_hint="Agnews", **common)
    ag.fit(X, y0)
    assert abs(ag.active_hedge_alpha_ - 0.25) < 1e-9
    assert ag.hedge_mean_ is not None


def test_fraud_huge_mid_latch_and_true_ensemble():
    rng = np.random.RandomState(11)
    # fraud-like: huge-n mid-low-d
    n, d = 120, 29
    Xn = rng.randn(n, d).astype(np.float32)
    model = build_model(
        "axion",
        epochs=6,
        patience=3,
        batch_size=32,
        seed=3,
        latch_alpha_semi=0.35,
        latch_alpha_semi_huge=0.30,
        latch_alpha_semi_huge_mid=0.35,
        c_hedge_alpha_semi=0.30,
        c_hedge_alpha_semi_huge=0.35,
        vis_alpha_semi_huge=0.25,
        learn_bank_weights=True,
        bank_weight_semi=True,
        true_ensemble=True,
        ensemble_heads=2,
        block_prob=0.35,
        max_train_samples=0,
    )
    # Force huge path via monkeypatch on resolve by using n>10000 is expensive;
    # call _resolve_setting_knobs directly after a tiny fit setup.
    model._resolve_setting_knobs(
        np.zeros(n, dtype=np.int64), high_d=False, raw_d=d, n=12000
    )
    assert abs(model.active_latch_alpha_ - 0.35) < 1e-9
    assert abs(model.active_c_hedge_alpha_ - 0.35) < 1e-9
    assert model.active_vis_alpha_ > 0.2
    # True ensemble trains on classical semi (keep tiny for wall time)
    model2 = build_model(
        "axion",
        epochs=3,
        patience=2,
        batch_size=32,
        seed=4,
        true_ensemble=True,
        ensemble_heads=2,
        learn_bank_weights=True,
        bank_weight_semi=True,
        use_interaction=False,
        contrastive_weight=0.0,
        score_k=4,
    )
    model2.fit(Xn[:60], np.zeros(60, dtype=np.int64))
    assert len(model2._ensemble_nets) == 1
    s = model2.score(Xn[:60])
    assert s.shape == (60,)
    assert np.all(np.isfinite(s))


def test_pca_high_d_path_and_semi_latch():
    rng = np.random.RandomState(7)
    # Simulate embedding dim ≥ pca_threshold
    n, d = 120, 512
    Xn = rng.randn(n, d).astype(np.float32)
    # Plant a low-rank anomaly direction
    Xa = Xn.copy()
    Xa[:20] += 2.5 * rng.randn(20, d).astype(np.float32)
    y_all_normal = np.zeros(100, dtype=np.int64)

    model = build_model(
        "axion",
        epochs=8,
        patience=3,
        batch_size=32,
        seed=9,
        pca_dim=32,
        pca_threshold=400,
        pca_variance=0.99,
        latch_alpha=0.4,
        latch_alpha_semi=0.0,
        latch_alpha_semi_high_d=0.35,
        hedge_alpha=0.35,
        hedge_alpha_semi_high_d=0.65,
    )
    model.fit(Xn[:100], y_all_normal)
    assert model.used_pca_ is True
    assert model.pca_components_ is not None
    assert model.resolved_["d"] <= 32
    assert model.resolved_["raw_d"] == 512
    assert abs(model.active_latch_alpha_ - 0.35) < 1e-9
    assert abs(model.active_hedge_alpha_ - 0.65) < 1e-9
    assert model.hedge_mean_ is not None
    assert model.hedge_std_ is not None and model.hedge_std_ > 0

    s = model.score(np.vstack([Xn[:100], Xa[:20]]))
    assert s.shape == (120,)
    assert np.all(np.isfinite(s))


def test_huge_unsup_latch_and_mae():
    rng = np.random.RandomState(11)
    # Subsample path: n>10000 mid-d → large_mid latch
    X = rng.randn(12000, 100).astype(np.float32)
    model = build_model(
        "axion",
        epochs=3,
        patience=2,
        batch_size=256,
        seed=0,
        max_train_samples=0,
        latch_alpha=0.4,
        latch_alpha_unsup_huge=0.70,
        latch_alpha_unsup_large_mid=0.75,
        mae_weight_unsup_huge=0.85,
        nll_weight_unsup_huge=0.15,
        pca_threshold=400,
    )
    model.fit(X)  # mixed / unsupervised (y None → not semi)
    assert model.is_semi_ is False
    assert abs(model.active_latch_alpha_ - 0.75) < 1e-9
    assert abs(model.active_mae_weight_ - 0.85) < 1e-6
    assert model.resolved_["score_k"] >= 40


def test_x_latch_low_d_huge_unsup():
    rng = np.random.RandomState(13)
    X = rng.randn(12000, 12).astype(np.float32)
    model = build_model(
        "axion",
        epochs=2,
        patience=1,
        batch_size=256,
        seed=0,
        max_train_samples=0,
        x_latch_alpha_unsup_huge=0.60,
        x_latch_max_d=16,
        pca_threshold=400,
    )
    model.fit(X)
    assert model.is_semi_ is False
    assert abs(model.active_x_latch_alpha_ - 0.60) < 1e-9
    assert model.x_mean_ is not None
    assert model.x_latch_mean_ is not None
    s = model.score(X[:100])
    assert s.shape == (100,)
    assert np.all(np.isfinite(s))
    # fraud-scale d=29 must NOT get x_latch (empirically regresses PR)
    Xf = rng.randn(12000, 29).astype(np.float32)
    model_f = build_model(
        "axion",
        epochs=2,
        patience=1,
        batch_size=256,
        seed=1,
        max_train_samples=0,
        x_latch_alpha_unsup_huge=0.90,
        x_latch_max_d=16,
        latch_alpha_unsup_low_d=0.85,
        pca_threshold=400,
    )
    model_f.fit(Xf)
    assert abs(model_f.active_x_latch_alpha_) < 1e-12
    assert model_f.active_latch_alpha_ >= 0.85 - 1e-9
    rng = np.random.RandomState(9)
    X = rng.randn(120, 8).astype(np.float32)
    model = build_model(
        "axion",
        epochs=3,
        patience=2,
        batch_size=32,
        seed=2,
        latch_alpha=0.4,
        latch_alpha_unsup_tiny=0.75,
    )
    model.fit(X)
    assert model.is_semi_ is False
    assert abs(model.active_latch_alpha_ - 0.75) < 1e-9


def test_contam_trim_huge_unsup():
    rng = np.random.RandomState(21)
    X = rng.randn(11000, 10).astype(np.float32)
    # Plant a few outliers
    X[-50:] += 8.0
    model = build_model(
        "axion",
        epochs=20,
        patience=20,
        batch_size=256,
        seed=0,
        max_train_samples=0,
        contam_trim_frac=0.05,
        contam_trim_warmup=2,
        contam_trim_every=2,
        x_latch_max_d=16,
        pca_threshold=400,
    )
    model.fit(X)
    assert model.resolved_.get("contam_trim_frac", 0) > 0
    assert int(model.resolved_.get("contam_kept", 0)) < 11000
    assert int(model.resolved_.get("contam_kept", 0)) >= int(0.5 * 11000)


def test_scale_hparams_cover_like():
    from axion.models.mcb import scale_hparams

    cover = scale_hparams(50000, 10, semi=False, raw_d=10)
    assert cover["score_k"] >= 48
    fraud = scale_hparams(50000, 29, semi=False, raw_d=29)
    assert fraud["score_k"] >= 40
    assert fraud["score_k"] < cover["score_k"] or fraud["score_k"] == 40


def test_tiny_epoch_boost_recorded():
    rng = np.random.RandomState(4)
    X = rng.randn(80, 8).astype(np.float32)
    model = build_model(
        "axion",
        epochs=10,
        patience=4,
        batch_size=32,
        seed=1,
        tiny_epoch_boost=2.0,
        semi_epoch_boost=1.25,
        latch_alpha_semi=0.0,
        latch_alpha_semi_tiny=0.0,
        use_interaction=False,
        ensemble_heads=1,
        contrastive_weight=0.0,
    )
    model.fit(X, np.zeros(80, dtype=np.int64))
    assert model.resolved_["max_epochs"] >= 20  # 10 * 1.25 semi * 2.0 tiny
    assert model.resolved_["patience"] >= 8


def test_nlp_hedge_softened():
    rng = np.random.RandomState(5)
    X = rng.randn(100, 768).astype(np.float32)
    model = build_model(
        "axion",
        epochs=5,
        patience=2,
        batch_size=32,
        seed=2,
        pca_dim=16,
        pca_threshold=400,
        hedge_alpha_semi_high_d=0.55,
        hedge_alpha_semi_nlp=0.0,
        hedge_alpha_semi_nlp_multi=0.25,
        nlp_raw_d_min=700,
        dataset_hint="Imdb",
        use_interaction=False,
        ensemble_heads=1,
        contrastive_weight=0.0,
    )
    model.fit(X, np.zeros(100, dtype=np.int64))
    assert model.used_pca_ is True
    assert abs(model.active_hedge_alpha_) < 1e-12
    assert model.hedge_mean_ is None  # no HEDGE anchors when beta=0


def test_hedge_disabled_on_classical():
    rng = np.random.RandomState(3)
    X = rng.randn(80, 12).astype(np.float32)
    model = build_model(
        "axion",
        epochs=6,
        patience=3,
        batch_size=32,
        seed=1,
        latch_alpha_semi=0.0,
        latch_alpha_semi_tiny=0.0,
        hedge_alpha=0.5,
        hedge_alpha_semi_high_d=0.5,
        pca_threshold=400,
        vis_alpha_semi=0.0,
        disagree_alpha_semi=0.0,
        hpd_alpha_semi=0.0,
        c_hedge_alpha_semi=0.0,
        use_interaction=False,
        ensemble_heads=1,
    )
    model.fit(X, np.zeros(80, dtype=np.int64))
    assert model.used_pca_ is False
    assert abs(model.active_hedge_alpha_) < 1e-12
    assert model.hedge_mean_ is None


def test_registry_alt_vit_roberta():
    from axion.paths import DEFAULT_ADBENCH_DATASETS
    from axion.data.registry import build_registry

    specs = {
        s.name: s
        for s in build_registry(
            DEFAULT_ADBENCH_DATASETS,
            cv_folder="CV_by_ViT",
            nlp_folder="NLP_by_RoBERTa",
        )
    }
    assert len(specs) == 57
    assert specs["CIFAR10"].embedding == "ViT"
    assert "CV_by_ViT" in specs["CIFAR10"].relative_paths[0]
    assert specs["Imdb"].embedding == "RoBERTa"
    assert "NLP_by_RoBERTa" in specs["Imdb"].relative_paths[0]
    # Classical unchanged
    assert specs["breastw"].relative_paths[0].startswith("Classical/")


def test_axion_run_one_array_smoke():
    rng = np.random.RandomState(1)
    X = rng.randn(100, 6).astype(np.float32)
    y = np.zeros(100, dtype=np.int64)
    y[-12:] = 1
    X[-12:] += 2.5
    r = run_one_array(
        X,
        y,
        dataset="toy",
        setting="semi-supervised",
        seed=111,
        model_name="axion",
        model_kwargs={"epochs": 12, "patience": 4, "batch_size": 32},
        protocol="paper",
    )
    assert np.isfinite(r.metrics["PR-AUC"])
    assert r.model == "axion"

"""Unit tests: category profiles + ECOD/COPOD/GOAD-lite."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.models import build_model
from axion.models.extensions.copod_view import CopodView
from axion.models.extensions.ecod_view import EcodView
from axion.models.extensions.goad_lite import GoadLiteExtension
from axion.models.profiles import resolve_profile


def test_profile_buckets():
    g = resolve_profile("glass", n=214, d=7, is_semi=True)
    assert g.bucket.startswith("tiny")
    assert not g.use_rare
    f = resolve_profile("fraud", n=284807, d=29, is_semi=True)
    assert f.bucket == "huge_imbalance_classical"
    assert f.use_copod and f.prefer_huge_n_safe_hw
    b = resolve_profile("backdoor", n=95329, d=196, is_semi=True)
    assert not b.use_rare and not b.use_ecod
    c = resolve_profile("CIFAR10", n=5000, d=512, is_semi=True, research_embeds=True)
    assert c.modality == "cv" and c.use_edge and c.use_nest


def test_high_d_classical_and_large_n_overrides():
    speech = resolve_profile("speech", n=3686, d=400, is_semi=True)
    assert speech.bucket == "high_d_classical"
    assert speech.use_ecod and abs(speech.ecod_alpha - 0.12) < 1e-9
    assert speech.use_goad and abs(speech.goad_alpha - 0.08) < 1e-9
    assert speech.use_spear
    aloi = resolve_profile("ALOI", n=49534, d=27, is_semi=True)
    assert aloi.bucket == "high_d_classical"
    celeba = resolve_profile("celeba", n=202599, d=39, is_semi=True)
    assert celeba.bucket == "large_n_classical"
    assert abs(celeba.ecod_alpha - 0.20) < 1e-9
    assert celeba.prefer_huge_n_safe_hw
    # Guards stay mild standard alphas
    cardio = resolve_profile("cardio", n=1831, d=21, is_semi=True)
    assert cardio.bucket == "standard_classical"
    assert abs(cardio.ecod_alpha - 0.05) < 1e-9


def test_svhn_research_vis_and_imdb_edge_off():
    sv = resolve_profile("SVHN", n=99289, d=512, is_semi=True, research_embeds=True)
    assert sv.use_nest and sv.vis_alpha_semi is not None and sv.vis_alpha_semi >= 0.35
    im = resolve_profile("Imdb", n=10000, d=768, is_semi=True, research_embeds=True)
    assert not im.use_edge and not im.use_nest
    im_off = resolve_profile("Imdb", n=10000, d=768, is_semi=True, research_embeds=False)
    assert not im_off.use_edge


def test_ecod_copod_scores_finite():
    rng = np.random.RandomState(0)
    Xn = rng.randn(200, 8).astype(np.float32)
    Xa = rng.randn(40, 8).astype(np.float32) + 3.0
    ec = EcodView(alpha=0.12)
    ec.fit(Xn)
    assert ec.active()
    s = ec.z_normed(np.vstack([Xn, Xa]))
    assert np.all(np.isfinite(s))
    cp = CopodView(alpha=0.18)
    cp.fit(Xn)
    assert cp.active()
    s2 = cp.z_normed(np.vstack([Xn, Xa]))
    assert np.all(np.isfinite(s2))


def test_goad_and_router_fuse():
    rng = np.random.RandomState(1)
    Xn = rng.randn(120, 12).astype(np.float32)
    X = np.vstack([Xn, rng.randn(24, 12).astype(np.float32) + 2.0])
    y = np.zeros(len(Xn), dtype=np.int64)
    m = build_model(
        "axion",
        epochs=5,
        patience=2,
        batch_size=32,
        seed=3,
        device="cpu",
        profile_router=True,
        research_embeds=False,
        spear=False,
        rare=False,
        edge=False,
        nest=False,
        goad=True,
        goad_alpha=0.1,
        goad_epochs=8,
        goad_transforms=3,
        ecod=True,
        ecod_alpha=0.12,
        dataset_hint="cardio",
        dataloader_num_workers=0,
    )
    m.fit(Xn, y)
    assert m.category_profile_ is not None
    s = m.score(X)
    assert np.all(np.isfinite(s))
    g = GoadLiteExtension(alpha=0.1, n_transforms=3, epochs=5, device="cpu")
    g.fit(Xn)
    assert g.active()

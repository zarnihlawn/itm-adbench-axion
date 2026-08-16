"""Throughput helpers: vectorized masks + hardware profile."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.models.mcb import sample_masks
from axion.util.hardware import (
    build_profile,
    recommend_score_batch_size,
)


def test_sample_masks_vectorized_shape_and_binary():
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    m = sample_masks(
        64, 16, rates=(0.2, 0.4), generator=g, device=torch.device("cpu"), vectorized=True
    )
    assert m.shape == (64, 16)
    assert ((m == 0) | (m == 1)).all()
    s = m.sum(dim=1)
    assert (s >= 1).all()
    assert (s < 16).all()


def test_sample_masks_legacy_seeded_stable():
    def once(seed: int):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        return sample_masks(
            32, 8, rates=(0.25,), generator=g, device=torch.device("cpu"), vectorized=False
        )

    a = once(42)
    b = once(42)
    c = once(43)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_mcs_chunk_bs_caps_legacy_masks():
    from axion.models.axion_model import AxionModel

    m = AxionModel(epochs=1, device="cpu", vectorized_masks=False, score_batch_size=32768)
    m._score_bs = 32768
    assert m._mcs_chunk_bs(100_000) == 512
    m.vectorized_masks = True
    assert m._mcs_chunk_bs(100_000) == 32768


def test_recommend_score_batch_size_scales_with_d():
    low = recommend_score_batch_size(10, device=torch.device("cpu"), override=0)
    high = recommend_score_batch_size(512, device=torch.device("cpu"), override=0)
    assert low >= 1024
    assert high <= low
    assert recommend_score_batch_size(10, override=2048) == 2048


def test_build_profile_cpu():
    p = build_profile(10, device=torch.device("cpu"), score_batch_size=0, dataloader_num_workers=0)
    assert p.score_batch_size >= 512
    assert p.dataloader_num_workers == 0

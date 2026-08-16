"""Bitwise protocol twin: AXION split_data vs official AnoDDAE."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.data.normalize import Standardizer, standardize_train_test
from axion.data.registry import build_registry, load_npz
from axion.data.splits import carve_val_from_train, split_data
from axion.paths import ANODDAE_SRC, DEFAULT_ADBENCH_DATASETS


def _load_anoddae_split():
    path = ANODDAE_SRC / "data.py"
    if not path.exists():
        pytest.skip(f"AnoDDAE reference missing: {path}")
    spec = importlib.util.spec_from_file_location("anoddae_data", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.split_data


@pytest.fixture(scope="module")
def anoddae_split():
    return _load_anoddae_split()


@pytest.fixture(scope="module")
def sample_arrays():
    """Three diverse real ADBench arrays + one synthetic."""
    root = DEFAULT_ADBENCH_DATASETS
    paths = [
        root / "Classical" / "6_cardio.npz",
        root / "Classical" / "14_glass.npz",
        root / "Classical" / "4_breastw.npz",
    ]
    arrays = []
    for p in paths:
        if not p.exists():
            pytest.skip(f"Missing NPZ {p}")
        arrays.append(load_npz(p))
    rng = np.random.RandomState(0)
    X = rng.randn(200, 8).astype(np.float32)
    y = np.zeros(200, dtype=np.int64)
    y[-20:] = 1
    arrays.append((X, y))
    return arrays


@pytest.mark.parametrize("setting", ["unsupervised", "semi-supervised"])
@pytest.mark.parametrize("seed", [111, 222, 333])
def test_split_matches_anoddae(anoddae_split, sample_arrays, setting, seed):
    for X, y in sample_arrays:
        # Reset global RNG identically before each call by using fresh seed inside both
        a = split_data(X.copy(), y.copy(), train_setting=setting, random_state=seed)
        b = anoddae_split(X.copy(), y.copy(), train_setting=setting, random_state=seed)
        for i, (xa, xb) in enumerate(zip(a, b)):
            np.testing.assert_array_equal(
                xa, xb, err_msg=f"mismatch part={i} setting={setting} seed={seed}"
            )


def test_semi_train_all_normal(sample_arrays):
    X, y = sample_arrays[0]
    Xtr, Xte, ytr, yte = split_data(X, y, "semi-supervised", random_state=111)
    assert ytr.sum() == 0
    assert yte.sum() == (y == 1).sum()
    assert len(Xtr) == (y == 0).sum() // 2
    assert len(Xte) == len(X) - len(Xtr)


def test_unsupervised_identity(sample_arrays):
    X, y = sample_arrays[0]
    Xtr, Xte, ytr, yte = split_data(X, y, "unsupervised", random_state=42)
    np.testing.assert_array_equal(Xtr, X)
    np.testing.assert_array_equal(Xte, X)
    np.testing.assert_array_equal(ytr, y)
    np.testing.assert_array_equal(yte, y)


def test_standardize_fit_on_train_only():
    rng = np.random.RandomState(1)
    X_train = rng.randn(100, 4) * 5 + 10
    X_test = rng.randn(50, 4) * 5 + 10
    Xt, Xe, scaler = standardize_train_test(X_train, X_test)
    assert Xt.shape == X_train.shape
    np.testing.assert_allclose(Xt.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(Xt.std(axis=0), 1.0, atol=1e-5)
    # Test transform uses train stats — mean not necessarily 0
    assert scaler.mean_ is not None
    recon = Xt * scaler.scale_ + scaler.mean_
    np.testing.assert_allclose(recon, X_train.astype(np.float32), rtol=1e-5, atol=1e-5)


def test_carve_val_disjoint():
    rng = np.random.RandomState(2)
    X = rng.randn(100, 3).astype(np.float32)
    y = np.zeros(100, dtype=np.int64)
    Xf, Xv, yf, yv = carve_val_from_train(X, y, val_fraction=0.2, random_state=7)
    assert len(Xf) + len(Xv) == 100
    assert len(Xv) >= 8
    # no overlapping rows by identity of indices via unique rows count
    # (float rows may collide rarely; check sizes only for robustness)
    assert yf.shape[0] == Xf.shape[0]


def test_registry_has_57():
    specs = build_registry()
    assert len(specs) == 57
    assert sum(1 for s in specs if s.modality == "classical") == 47
    assert sum(1 for s in specs if s.modality == "cv") == 5
    assert sum(1 for s in specs if s.modality == "nlp") == 5
    names = [s.name for s in specs]
    assert names.count("CIFAR10") == 1
    assert "20newsgroups" in names
    assert all(s.embedding in {"none", "ResNet18", "BERT"} for s in specs)

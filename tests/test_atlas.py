"""Atlas completeness and NPZ path integrity."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axion.data.atlas import build_atlas
from axion.data.registry import build_registry
from axion.paths import ATLAS_CSV, DEFAULT_ADBENCH_DATASETS


REQUIRED_COLS = {
    "name",
    "modality",
    "paper_category",
    "embedding",
    "n_files",
    "n_pooled",
    "d",
    "n_anom_pooled",
    "anom_rate_pooled",
    "difficulty",
    "relative_paths",
}


@pytest.fixture(scope="module")
def atlas_df():
    return build_atlas()


def test_atlas_57_rows(atlas_df):
    assert len(atlas_df) == 57
    assert REQUIRED_COLS.issubset(set(atlas_df.columns))
    assert atlas_df["name"].nunique() == 57


def test_atlas_modality_counts(atlas_df):
    counts = atlas_df["modality"].value_counts().to_dict()
    assert counts["classical"] == 47
    assert counts["cv"] == 5
    assert counts["nlp"] == 5


def test_atlas_embeddings(atlas_df):
    classical = atlas_df[atlas_df["modality"] == "classical"]
    assert (classical["embedding"] == "none").all()
    cv = atlas_df[atlas_df["modality"] == "cv"]
    assert (cv["embedding"] == "ResNet18").all()
    nlp = atlas_df[atlas_df["modality"] == "nlp"]
    assert (nlp["embedding"] == "BERT").all()


def test_all_npz_exist(atlas_df):
    root = DEFAULT_ADBENCH_DATASETS
    missing = []
    for _, row in atlas_df.iterrows():
        for rel in str(row["relative_paths"]).split(";"):
            if not (root / rel).exists():
                missing.append(rel)
    assert not missing, f"Missing NPZs: {missing[:10]}"


def test_csv_written():
    assert ATLAS_CSV.exists()
    df = pd.read_csv(ATLAS_CSV)
    assert len(df) == 57


def test_registry_paths_match_atlas(atlas_df):
    specs = {s.name: s for s in build_registry()}
    for _, row in atlas_df.iterrows():
        spec = specs[row["name"]]
        assert ";".join(spec.relative_paths) == row["relative_paths"]

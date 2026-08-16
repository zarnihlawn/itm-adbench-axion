"""ADBench dataset registry (fair atlas + Beat-Paper atlas)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from axion.paths import AXION_ROOT, DEFAULT_ADBENCH_DATASETS

CLASSICAL_FILES: List[str] = [
    "1_ALOI.npz",
    "2_annthyroid.npz",
    "3_backdoor.npz",
    "4_breastw.npz",
    "5_campaign.npz",
    "6_cardio.npz",
    "7_Cardiotocography.npz",
    "8_celeba.npz",
    "9_census.npz",
    "10_cover.npz",
    "11_donors.npz",
    "12_fault.npz",
    "13_fraud.npz",
    "14_glass.npz",
    "15_Hepatitis.npz",
    "16_http.npz",
    "17_InternetAds.npz",
    "18_Ionosphere.npz",
    "19_landsat.npz",
    "20_letter.npz",
    "21_Lymphography.npz",
    "22_magic.gamma.npz",
    "23_mammography.npz",
    "24_mnist.npz",
    "25_musk.npz",
    "26_optdigits.npz",
    "27_PageBlocks.npz",
    "28_pendigits.npz",
    "29_Pima.npz",
    "30_satellite.npz",
    "31_satimage-2.npz",
    "32_shuttle.npz",
    "33_skin.npz",
    "34_smtp.npz",
    "35_SpamBase.npz",
    "36_speech.npz",
    "37_Stamps.npz",
    "38_thyroid.npz",
    "39_vertebral.npz",
    "40_vowels.npz",
    "41_Waveform.npz",
    "42_WBC.npz",
    "43_WDBC.npz",
    "44_Wilt.npz",
    "45_wine.npz",
    "46_WPBC.npz",
    "47_yeast.npz",
]

# Fallback paper categories when atlas row lacks paper_category.
PAPER_CATEGORIES: Dict[str, str] = {
    "ALOI": "Image",
    "annthyroid": "Healthcare",
    "backdoor": "Network",
    "breastw": "Healthcare",
    "campaign": "Finance",
    "cardio": "Healthcare",
    "Cardiotocography": "Healthcare",
    "celeba": "Image",
    "census": "Sociology",
    "cover": "Botany",
    "donors": "Sociology",
    "fault": "Physical",
    "fraud": "Finance",
    "glass": "Forensic",
    "Hepatitis": "Healthcare",
    "http": "Web",
    "InternetAds": "Image",
    "Ionosphere": "Oryctognosy",
    "landsat": "Astronautics",
    "letter": "Image",
    "Lymphography": "Healthcare",
    "magic.gamma": "Physical",
    "mammography": "Healthcare",
    "mnist": "Image",
    "musk": "Chemistry",
    "optdigits": "Image",
    "PageBlocks": "Document",
    "pendigits": "Image",
    "Pima": "Healthcare",
    "satellite": "Astronautics",
    "satimage-2": "Astronautics",
    "shuttle": "Astronautics",
    "skin": "Image",
    "smtp": "Web",
    "SpamBase": "Document",
    "speech": "Linguistics",
    "Stamps": "Document",
    "thyroid": "Healthcare",
    "vertebral": "Biology",
    "vowels": "Linguistics",
    "Waveform": "Physics",
    "WBC": "Healthcare",
    "WDBC": "Healthcare",
    "Wilt": "Botany",
    "wine": "Chemistry",
    "WPBC": "Healthcare",
    "yeast": "Biology",
    "CIFAR10": "Image",
    "FashionMNIST": "Image",
    "MNIST-C": "Image",
    "MVTec-AD": "Image",
    "SVHN": "Image",
    "Agnews": "NLP",
    "Amazon": "NLP",
    "Imdb": "NLP",
    "Yelp": "NLP",
    "20newsgroups": "NLP",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    modality: str
    paper_category: str
    relative_paths: Tuple[str, ...]
    embedding: str = "none"


def _classical_display_name(fname: str) -> str:
    stem = Path(fname).stem
    if "_" in stem:
        return stem.split("_", 1)[1]
    return stem


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    X = np.asarray(z["X"], dtype=np.float32)
    if X.ndim > 2:
        X = X.reshape(len(X), -1)
    y = np.asarray(z["y"]).astype(np.int64).ravel()
    return X, y


def load_dataset_files(
    spec: DatasetSpec,
    adbench_root: Optional[Path] = None,
) -> List[Tuple[np.ndarray, np.ndarray, Path]]:
    root = Path(adbench_root) if adbench_root is not None else DEFAULT_ADBENCH_DATASETS
    out: List[Tuple[np.ndarray, np.ndarray, Path]] = []
    for rel in spec.relative_paths:
        p = root / rel
        X, y = load_npz(p)
        out.append((X, y, p))
    return out


def _specs_from_atlas_csv(atlas_path: Path, *, fair_embeds: bool = True) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []
    with open(atlas_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            modality = row["modality"]
            rels = tuple(p.strip() for p in str(row["relative_paths"]).split(";") if p.strip())
            emb = (row.get("embedding") or "none").strip() or "none"
            if not fair_embeds:
                if emb == "none" and modality == "cv":
                    emb = "ViT"
                elif emb == "none" and modality == "nlp":
                    nlp_embed = row.get("nlp_embed") or "RoBERTa"
                    emb = str(nlp_embed).split("_")[-1].split("/")[0]
            specs.append(
                DatasetSpec(
                    name=name,
                    modality=modality,
                    paper_category=row.get("paper_category")
                    or PAPER_CATEGORIES.get(name, ""),
                    relative_paths=rels,
                    embedding=emb if emb != "none" else "none",
                )
            )
    return specs


def build_registry(
    adbench_root: Optional[Path] = None,
    *,
    cv_folder: str = "CV_by_ResNet18",
    nlp_folder: str = "NLP_by_BERT",
) -> List[DatasetSpec]:
    """Fair Table-1 style registry (ResNet18 / BERT paths from atlas_57.csv)."""
    del adbench_root, cv_folder, nlp_folder  # atlas embeds paths already
    atlas_path = AXION_ROOT / "data" / "atlas_57.csv"
    if not atlas_path.exists():
        raise FileNotFoundError(f"Fair atlas missing: {atlas_path}")
    specs = _specs_from_atlas_csv(atlas_path, fair_embeds=True)
    if len(specs) != 57:
        raise RuntimeError(f"Fair atlas expected 57 datasets, got {len(specs)}")
    return specs


def registry_by_name(
    adbench_root: Optional[Path] = None,
    *,
    cv_folder: str = "CV_by_ResNet18",
    nlp_folder: str = "NLP_by_BERT",
) -> Dict[str, DatasetSpec]:
    return {
        s.name: s
        for s in build_registry(
            adbench_root, cv_folder=cv_folder, nlp_folder=nlp_folder
        )
    }


def build_beat_paper_registry(
    adbench_root: Optional[Path] = None,
    *,
    atlas_csv: Optional[Path] = None,
) -> List[DatasetSpec]:
    """Build 57-dataset registry from beat-paper atlas CSV (per-dataset embed paths)."""
    del adbench_root  # paths are relative to DEFAULT_ADBENCH_DATASETS at load time
    atlas_path = (
        Path(atlas_csv)
        if atlas_csv is not None
        else AXION_ROOT / "data" / "atlas_57_beat_paper.csv"
    )
    if not atlas_path.is_absolute():
        atlas_path = AXION_ROOT / "data" / atlas_path.name
    if not atlas_path.exists():
        raise FileNotFoundError(f"Beat-paper atlas missing: {atlas_path}")

    specs = _specs_from_atlas_csv(atlas_path, fair_embeds=False)
    if len(specs) != 57:
        raise RuntimeError(f"Beat-paper atlas expected 57 datasets, got {len(specs)}")
    return specs


def registry_beat_paper_by_name(
    adbench_root: Optional[Path] = None,
    *,
    atlas_csv: Optional[Path] = None,
) -> Dict[str, DatasetSpec]:
    return {s.name: s for s in build_beat_paper_registry(adbench_root, atlas_csv=atlas_csv)}

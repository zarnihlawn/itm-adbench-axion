"""Atlas-driven category profiles for AXION research / ship paths.

Maps modality + difficulty tags → which score heads and HW knobs to prefer.
Does not override hard locks (backdoor+RARE ban, glass freeze) unless allowlisted.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from axion.paths import ATLAS_CSV


# Classical high-d / ranking-collapse allowlist (full57 weak tail).
HIGH_D_CLASSICAL_DATASETS = frozenset(
    {"speech", "aloi", "internetads", "waveform", "optdigits"}
)


@dataclass
class CategoryProfile:
    bucket: str = "standard_classical"
    modality: str = "classical"
    difficulty: str = "standard"
    # Score heads
    use_ecod: bool = False
    use_copod: bool = False
    use_goad: bool = False
    use_spear: bool = True
    use_rare: bool = False
    use_edge: bool = False
    use_nest: bool = False
    # Fuse strengths (0 = off)
    ecod_alpha: float = 0.0
    copod_alpha: float = 0.0
    goad_alpha: float = 0.0
    # Optional semi view override (None = leave model default)
    vis_alpha_semi: Optional[float] = None
    # HW hints (caller may apply when profile_router_hw is on)
    prefer_huge_n_safe_hw: bool = False
    prefer_vectorized: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_ATLAS_CACHE: Optional[Dict[str, Dict[str, str]]] = None


def load_atlas(path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    global _ATLAS_CACHE
    if _ATLAS_CACHE is not None and path is None:
        return _ATLAS_CACHE
    csv_path = Path(path) if path is not None else ATLAS_CSV
    rows: Dict[str, Dict[str, str]] = {}
    if not csv_path.is_file():
        return rows
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = str(row.get("name", "")).strip()
            if name:
                rows[name] = row
                rows[name.lower()] = row
    if path is None:
        _ATLAS_CACHE = rows
    return rows


def _tags(difficulty: str) -> set:
    return {t.strip() for t in str(difficulty or "").split("|") if t.strip()}


def resolve_profile(
    dataset_hint: Optional[str] = None,
    *,
    n: Optional[int] = None,
    d: Optional[int] = None,
    is_semi: bool = True,
    research_embeds: bool = False,
    atlas_path: Optional[Path] = None,
) -> CategoryProfile:
    """Resolve a category profile from atlas row and/or runtime n,d."""
    hint = (dataset_hint or "").strip()
    atlas = load_atlas(atlas_path)
    row = atlas.get(hint) or atlas.get(hint.lower()) or {}
    modality = str(row.get("modality") or "classical").lower()
    difficulty = str(row.get("difficulty") or "standard")
    tags = _tags(difficulty)
    try:
        n_atlas = int(float(row.get("n_pooled") or 0))
    except (TypeError, ValueError):
        n_atlas = 0
    try:
        d_atlas = int(float(row.get("d") or 0))
    except (TypeError, ValueError):
        d_atlas = 0
    try:
        anom = float(row.get("anom_rate_pooled") or 0.0)
    except (TypeError, ValueError):
        anom = 0.0
    n_eff = int(n) if n is not None else n_atlas
    d_eff = int(d) if d is not None else d_atlas

    # Modality overrides from hint when atlas miss
    h = hint.lower()
    if any(k in h for k in ("cifar", "fashion", "mnist", "svhn", "mvtec")):
        modality = "cv"
    elif any(k in h for k in ("agnews", "amazon", "imdb", "yelp", "20news")):
        modality = "nlp"

    # Prefer atlas difficulty tags when a row exists. Runtime n only adds large_n
    # (train caps) and never invents tiny_n that would lock mid datasets like breastw.
    if row:
        if n_eff >= 50_000 or n_atlas >= 50_000:
            tags.add("large_n")
        if d_eff >= 400 or d_atlas >= 400:
            tags.add("high_d")
    else:
        if n_eff > 0 and n_eff < 500:
            tags.add("tiny_n")
        if n_eff >= 50_000:
            tags.add("large_n")
        if d_eff >= 400:
            tags.add("high_d")
        if anom > 0 and anom < 0.01:
            tags.add("extreme_imbalance")

    # Hard locks
    if "glass" in h:
        return CategoryProfile(
            bucket="tiny_classical_lock",
            modality="classical",
            difficulty="tiny_n",
            use_spear=False,
            use_rare=False,
            notes="glass lock: MCS-primary only",
        )
    if "backdoor" in h:
        return CategoryProfile(
            bucket="backdoor_lock",
            modality="classical",
            difficulty="standard",
            use_spear=True,
            use_rare=False,
            use_ecod=False,
            prefer_huge_n_safe_hw=True,
            prefer_vectorized=False,
            notes="never RARE/ECOD on backdoor; safe HW; keep MCS ship",
        )

    if modality == "cv":
        svhn = "svhn" in h
        return CategoryProfile(
            bucket="cv_embed",
            modality="cv",
            difficulty=difficulty,
            use_spear=False,
            use_rare=False,
            use_edge=True,
            use_nest=bool(research_embeds),
            prefer_vectorized=bool(research_embeds),
            vis_alpha_semi=0.35 if (svhn and research_embeds) else None,
            notes="CV embed lane; SVHN research: NEST + higher vis",
        )
    if modality == "nlp":
        imdb = "imdb" in h
        sentiment = imdb or ("amazon" in h) or ("yelp" in h)
        return CategoryProfile(
            bucket="nlp_embed",
            modality="nlp",
            difficulty=difficulty,
            use_spear=False,
            use_rare=False,
            use_edge=not imdb,
            use_nest=bool(research_embeds) and not imdb,
            prefer_vectorized=bool(research_embeds),
            notes=(
                "Imdb: edge/nest off; sentiment sets prefer E5 on research"
                if sentiment
                else "topic NLP: EDGE; research may NEST"
            ),
        )

    # Classical buckets
    if "tiny_n" in tags:
        return CategoryProfile(
            bucket="tiny_classical",
            modality="classical",
            difficulty="tiny_n",
            use_spear=False,
            use_rare=False,
            prefer_huge_n_safe_hw=False,
            notes="tiny-n: MCS + SCALE; no SPEAR/RARE",
        )

    # High-d weak-tail allowlist wins over large_n tagging
    if h in HIGH_D_CLASSICAL_DATASETS:
        return CategoryProfile(
            bucket="high_d_classical",
            modality="classical",
            difficulty=difficulty or "high_d",
            use_spear=bool(is_semi),
            use_rare=False,
            use_ecod=True,
            use_goad=bool(is_semi),
            ecod_alpha=0.12,
            goad_alpha=0.08,
            notes="high_d classical weak tail: SPEAR + stronger ECOD/GOAD",
        )

    if "extreme_imbalance" in tags or ("large_n" in tags and anom < 0.02):
        use_rare = is_semi and any(k in h for k in ("fraud", "cover"))
        return CategoryProfile(
            bucket="huge_imbalance_classical",
            modality="classical",
            difficulty="|".join(sorted(tags)) or "large_n",
            use_spear=False,
            use_rare=use_rare,
            use_ecod=True,
            use_copod=True,
            ecod_alpha=0.08,
            copod_alpha=0.10,
            prefer_huge_n_safe_hw=True,
            prefer_vectorized=False,
            notes="legacy MCS + COPOD/ECOD; RARE only allowlisted",
        )
    if "large_n" in tags:
        return CategoryProfile(
            bucket="large_n_classical",
            modality="classical",
            difficulty="large_n",
            use_spear=False,
            use_rare=False,
            use_ecod=True,
            ecod_alpha=0.20,
            prefer_huge_n_safe_hw=True,
            prefer_vectorized=False,
            notes="large-n classical: stronger ECOD + safe HW",
        )

    # standard mid classical (cardio/breastw stay mild alphas)
    return CategoryProfile(
        bucket="standard_classical",
        modality="classical",
        difficulty=difficulty or "standard",
        use_spear=bool(is_semi),
        use_rare=False,
        use_ecod=True,
        use_goad=bool(is_semi),
        ecod_alpha=0.05,
        goad_alpha=0.05,
        notes="mid classical: SPEAR + ECOD + GOAD-lite",
    )


__all__ = [
    "CategoryProfile",
    "HIGH_D_CLASSICAL_DATASETS",
    "load_atlas",
    "resolve_profile",
]

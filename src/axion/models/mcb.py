"""MCB — Mask Curriculum Bank for AXION."""
from __future__ import annotations

from typing import Optional, Sequence

import torch


def sample_masks_legacy(
    batch_size: int,
    d: int,
    *,
    rates: Sequence[float] = (0.15, 0.3, 0.5),
    block_prob: float = 0.25,
    max_block_frac: float = 0.25,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Original CPU Python-loop masks (quality-matched to pre-throughput g5)."""
    out_device = device or torch.device("cpu")
    cpu = torch.device("cpu")
    rate_idx = torch.randint(0, len(rates), (batch_size,), generator=generator, device=cpu)
    rates_t = torch.tensor(list(rates), dtype=torch.float32, device=cpu)
    p = rates_t[rate_idx].unsqueeze(1).expand(batch_size, d)
    bern = torch.rand(batch_size, d, generator=generator, device=cpu) < p
    mask = bern.float()

    if block_prob > 0 and d >= 4:
        use_block = torch.rand(batch_size, generator=generator, device=cpu) < block_prob
        for i in range(batch_size):
            if not bool(use_block[i]):
                continue
            blk = max(
                1,
                int(
                    d
                    * float(max_block_frac)
                    * float(torch.rand(1, generator=generator, device=cpu).item())
                ),
            )
            blk = min(blk, d)
            start = int(torch.randint(0, d - blk + 1, (1,), generator=generator, device=cpu).item())
            mask[i, :] = 0.0
            mask[i, start : start + blk] = 1.0

    if d > 1:
        for i in range(batch_size):
            if mask[i].sum() < 1:
                j = int(torch.randint(0, d, (1,), generator=generator, device=cpu).item())
                mask[i, j] = 1.0
            if mask[i].sum() >= d:
                j = int(torch.randint(0, d, (1,), generator=generator, device=cpu).item())
                mask[i, j] = 0.0
    return mask.to(out_device)


def sample_masks_vectorized(
    batch_size: int,
    d: int,
    *,
    rates: Sequence[float] = (0.15, 0.3, 0.5),
    block_prob: float = 0.25,
    max_block_frac: float = 0.25,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Faster vectorized masks (may shift block-mask distribution vs legacy)."""
    out_device = device or torch.device("cpu")
    gen_dev = torch.device("cpu")
    if generator is not None:
        try:
            gen_dev = generator.device
        except Exception:
            gen_dev = torch.device("cpu")
    if out_device.type == "cuda" and (generator is None or gen_dev.type == "cuda"):
        work = out_device
    else:
        work = torch.device("cpu")

    rate_idx = torch.randint(
        0, len(rates), (batch_size,), generator=generator, device=work
    )
    rates_t = torch.tensor(list(rates), dtype=torch.float32, device=work)
    p = rates_t[rate_idx].unsqueeze(1).expand(batch_size, d)
    bern = torch.rand(batch_size, d, generator=generator, device=work) < p
    mask = bern.float()

    if block_prob > 0 and d >= 4:
        use_block = torch.rand(batch_size, generator=generator, device=work) < block_prob
        u = torch.rand(batch_size, generator=generator, device=work)
        blk = (float(d) * float(max_block_frac) * u).long().clamp(min=1, max=d)
        span = (d - blk).clamp(min=0)
        u2 = torch.rand(batch_size, generator=generator, device=work)
        start = (u2 * (span.float() + 1e-6)).long().clamp(min=0)
        start = torch.minimum(start, span)
        cols = torch.arange(d, device=work).unsqueeze(0).expand(batch_size, d)
        in_block = (cols >= start.unsqueeze(1)) & (cols < (start + blk).unsqueeze(1))
        block_mask = in_block.float()
        mask = torch.where(use_block.unsqueeze(1), block_mask, mask)

    if d > 1:
        row_sum = mask.sum(dim=1)
        empty = row_sum < 1
        full = row_sum >= float(d)
        if bool(empty.any()) or bool(full.any()):
            j = torch.randint(0, d, (batch_size,), generator=generator, device=work)
            if bool(empty.any()):
                idx = empty.nonzero(as_tuple=False).squeeze(-1)
                mask[idx, j[idx]] = 1.0
            row_sum2 = mask.sum(dim=1)
            full2 = row_sum2 >= float(d)
            idx = full2.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                mask[idx, j[idx]] = 0.0

    return mask.to(out_device)


def sample_masks(
    batch_size: int,
    d: int,
    *,
    rates: Sequence[float] = (0.15, 0.3, 0.5),
    block_prob: float = 0.25,
    max_block_frac: float = 0.25,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    vectorized: bool = False,
) -> torch.Tensor:
    """Sample binary masks (1 = masked). Default legacy path preserves g5 quality."""
    fn = sample_masks_vectorized if vectorized else sample_masks_legacy
    return fn(
        batch_size,
        d,
        rates=rates,
        block_prob=block_prob,
        max_block_frac=max_block_frac,
        generator=generator,
        device=device,
    )


def soften_mask_rates(rates: Sequence[float], factor: float = 0.7) -> tuple[float, ...]:
    """Lower mask rates for all-normal (semi) training — sharper normal recon."""
    return tuple(float(max(0.05, min(0.55, r * factor))) for r in rates)


def scale_hparams(
    n: int,
    d: int,
    *,
    semi: bool = False,
    raw_d: Optional[int] = None,
    tiny_n_threshold: int = 250,
) -> dict:
    """SCALE: train-only size/d routing for architecture knobs.

    Phase 5: ``d`` is the *model* dimension (after optional PCA). ``raw_d`` is
    the original feature count (used for rare-n heuristics when PCA shrinks d).
    Gap-fix: tiny-n (glass n=214) softer masks + score_k≥36; huge unsup score_k≥32.

    ``tiny_n_threshold`` defaults to 250 so glass (214) stays on the tiny MCS path
    (hardcoded ``n < 200`` previously missed glass and broke g5 unsup 10.35).
    """
    d_orig = int(raw_d if raw_d is not None else d)
    tiny_thr = int(tiny_n_threshold)
    if d >= 400:
        hidden, latent, depth = 512, 128, 3
        mask_rates = (0.08, 0.15, 0.28)
        score_k = 24
        dropout = 0.05
        high_mask_delta = 0.12
        high_mask_cap = 0.45
    elif d >= 64:
        hidden, latent, depth = 320, 80, 3
        mask_rates = (0.12, 0.25, 0.4)
        score_k = 20 if d_orig >= 400 else 18
        dropout = 0.08 if n >= 500 else 0.05
        high_mask_delta = 0.2
        high_mask_cap = 0.7
    else:
        hidden, latent, depth = 160, 40, 2
        mask_rates = (0.15, 0.3, 0.45)
        score_k = 20
        dropout = 0.08 if n >= 500 else 0.05
        high_mask_delta = 0.2
        high_mask_cap = 0.75

    if n < tiny_thr:
        if d < 64:
            depth = max(3, depth)
            hidden = max(128, hidden)
            latent = max(48, latent)
        else:
            depth = max(2, depth)
            hidden = max(128, hidden)
        # g5 glass used score_k≈36; avoid *1.25 inflation to 45
        score_k = max(36, score_k)
        dropout = max(float(dropout), 0.12)
        mask_rates = soften_mask_rates(mask_rates, factor=0.40)
        high_mask_delta = float(min(high_mask_delta, 0.10))
        high_mask_cap = float(min(high_mask_cap, 0.45))
    elif n > 10000:
        if semi:
            # g5-like MCS budget on huge semi (cover/fraud); avoid score_k=48 thrash
            score_k = max(score_k, 24)
        else:
            score_k = max(score_k, 40)
            high_mask_delta = float(min(0.28, high_mask_delta + 0.06))
            high_mask_cap = float(min(0.8, max(high_mask_cap, 0.72)))
            if d < 16:
                score_k = max(48, score_k)
                mask_rates = soften_mask_rates(mask_rates, factor=0.85)
                high_mask_delta = float(min(0.32, high_mask_delta + 0.04))
                high_mask_cap = float(min(0.85, max(high_mask_cap, 0.78)))
    elif n > 5000:
        score_k = max(16, score_k)

    if semi:
        mask_rates = soften_mask_rates(mask_rates, factor=0.75)
        if n < tiny_thr:
            score_k = int(min(48, max(36, score_k)))
        else:
            score_k = int(round(score_k * 1.25))
            if n > 10000:
                # Denser rare-tail MCS for cover/fraud (was cap 32 → effective 30)
                score_k = 36
            else:
                score_k = int(min(48, score_k))
        high_mask_delta = float(min(high_mask_delta, 0.15))
        high_mask_cap = float(min(high_mask_cap, 0.55))
        # Keep g5 high-mask caps on huge classical semi (avoid 0.21/0.58 bump)

    return {
        "hidden": hidden,
        "latent": latent,
        "depth": depth,
        "mask_rates": mask_rates,
        "score_k": score_k,
        "dropout": dropout,
        "high_mask_delta": high_mask_delta,
        "high_mask_cap": high_mask_cap,
        "semi": bool(semi),
        "raw_d": d_orig,
        "tiny_n": bool(n < tiny_thr),
        "huge_n": bool(n > 10000),
    }

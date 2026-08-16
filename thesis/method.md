# AXION method (Phase 2–5 + SPEAR)

**AXION** = Adaptive cross-feature Interaction Observation Network.

Orthogonal to AnoDDAE (no diffusion schedule, no full-sum recon-over-t, no DDAE-C contrastive).

## Components

| Acronym | Role |
|---------|------|
| **MCB** | Mask Curriculum Bank — Bernoulli + block masks; multi-rate |
| **FX-Enc** | Residual MLP on `[x ⊙ (1−m) ‖ m]` |
| **HPD** | Heteroscedastic head `(μ, log σ²)` for masked cells |
| **MCS** | Monte-Carlo score: hybrid MAE+NLL over K masks × rate banks |
| **LATCH** | Diagonal Mahalanobis on fully-visible latent `z` |
| **SCALE** | Train-only `n,d` → hidden / depth / K / mask rates (tiny-n / huge-n branches) |
| **PCA** | Train-only whitening when raw `d≥400`; variance-capped with `pca_k_min` |
| **HEDGE** | High-d PCA residual energy fused into the score (CV strong; NLP semi off) |
| **SPEAR** | Semi-only PR-tail ranking extension on frozen multi-view scores (classical / low-d) |

## Training

- Loss: Gaussian NLL on masked cells only
- Early stop: val NLL from train carve (`val_loss`) — never test PR
- GPU: AMP fp16 + larger batch
- **Semi:** softer masks, `semi_epoch_boost`
- **Tiny-n (`n<200`):** softer masks, `score_k≥28`, depth≥3 on low-d, `tiny_epoch_boost`
- **Huge-n unsup:** `score_k≥32`, stronger LATCH + MAE-heavier hybrid
- **High-d:** PCA on train only; network in PCA space
- **SPEAR (after freeze):** synthesize hard negatives from normals; train tiny pairwise / soft-AP ranker on view features; never touches recon NLL

## Score

\[
s(x)=\mathrm{z}_{\mathrm{train}}(\mathrm{MCS})+\alpha\,\mathrm{z}_{\mathrm{train}}(\mathrm{LATCH})+\beta\,\mathrm{z}_{\mathrm{train}}(\mathrm{HEDGE})+\gamma\,\mathrm{z}_{\mathrm{train}}(\mathrm{SPEAR})
\]

| Setting | \(\alpha\) | \(\beta\) | \(\gamma\) (SPEAR) | Hybrid |
|---------|------------|-----------|--------------------|--------|
| Unsup default | 0.40 | 0.35 if high-d CV else NLP 0.25 / 0 | **0** | mae 0.60 / nll 0.40 |
| Unsup huge-n | 0.50 (0.55 mid-d large) | as above | **0** | mae 0.70 / nll 0.30 |
| Classical semi | **0.0** | 0 | **0.35** (if fitted) | mae 0.80 / nll 0.20 |
| High-d CV semi | 0.35 | **0.65** | **0** (classical_only) | mae 0.80 / nll 0.20 |
| High-d NLP semi | 0.35 | **0.0** | **0** (Imdb blocked) | mae 0.80 / nll 0.20 |

SPEAR views: MCS, LATCH, VIS, Disagree, HPD, C-HEDGE. Synth: swap / mix-recon / extreme. See `results/axion_spear/diagnosis.md`.

## Dual-track gates (ship)

| Track | Meaning |
|-------|---------|
| A | Classical-8 formal margins |
| B | Embed calibrated (mean PR ≥27.5, Imdb ROC ≥50.5) |
| C | 12-ds unsup full margin + semi ROC margin (not impossible semi PR) |

`full57` requires A+B+C. See `VAST_RUN.md`.

## Protocol

AnoDDAE paper-faithful splits (`PROTOCOL.md`). Not ADBench `DataGenerator`.

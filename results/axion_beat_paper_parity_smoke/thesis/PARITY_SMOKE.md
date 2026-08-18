# Parity smoke (Phase A): gap datasets vs dual_lift claim

Run: `bash scripts/smoke_parity_gap4.sh` with `PY` pointing at the project venv.

## Root causes fixed

1. Broken `data/embeds_alt*` symlinks (`../project/...` resolved inside the slim repo). Relinked to `../../project/axion/data/embeds_alt(_whitened)`.
2. GMM `n_components` omitted from frozen map: catalog defaults + claim-metric enrich (`n_components=1` ViT, `2` E5) via `beat_paper_run_patches.py`.
3. `gmm_scores` no longer falls back to adaptive `min(8, n//50)` when unset.

## Claim vs smoke (seed 111, `gpu_beat_paper.yaml`)

| Dataset | Setting | Claim PR | Smoke PR | dPR |
|---------|---------|----------|----------|-----|
| CIFAR10 | unsupervised | 89.40 | 89.40 | 0.00 |
| FashionMNIST | semi | 91.27 | 91.25 | -0.02 |
| FashionMNIST | unsupervised | 88.97 | 88.97 | 0.00 |
| MNIST-C | semi | 73.93 | 73.91 | -0.02 |
| MNIST-C | unsupervised | 65.31 | 70.24 | +4.94 |
| Amazon | semi | 35.68 | 35.68 | 0.00 |
| Amazon | unsupervised | 22.16 | 22.16 | 0.00 |

PASS: within a few PR of claim on all four gap datasets.

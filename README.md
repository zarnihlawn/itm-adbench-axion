# AXION (itm-adbench-axion)

**A**daptive cross-feature **I**nteraction **O**bservation **N**etwork: anomaly detection on [ADBench](https://github.com/Minqi824/ADBench) (57 datasets), evaluated against **AnoDDAE** ([arXiv:2508.00758](https://arxiv.org/abs/2508.00758)).

This repository is **AXION only** (latest Beat-Paper / dual_lift claim). Large run logs, embeddings, and legacy AdaDDAE trees are not included.

## Claim snapshot (dual_lift)

Mean PR / ROC over ADBench-57 (seed-111, mean-over-variants):

| Setting | Dual lift | Paper AnoDDAE | Δ |
|---------|-----------|---------------|---|
| Semi-supervised | **68.16 / 89.38** | 61.36 / 83.17 | +6.80 / +6.21 |
| Unsupervised | **46.68 / 84.52** | 32.77 / 74.08 | +13.91 / +10.44 |

Claim margin (paper+3 on all four): pass. Semi stretch and unsup PR≥45: pass. Unsup ROC 88 remains aspirational.

Canonical artifacts: [`results/axion_beat_paper_dual_lift/`](results/axion_beat_paper_dual_lift/).

## Layout

```text
.
├── README.md
├── docs/AXION.md          # Claim detail and full-run recipe
├── configs/               # default.yaml + gpu_beat_paper.yaml
├── src/axion/             # Model and training code
├── scripts/               # beat_paper_* + dual_lift assemble
├── tests/
└── results/axion_beat_paper_dual_lift/
```

## Requirements

- Python ≥ 3.10
- ADBench datasets as a sibling checkout: `../ADBench/adbench/datasets`
- Optional: paper AnoDDAE at `../AnoDDAE/`

## Setup

```bash
git clone git@github.com:zarnihlawn/itm-adbench-axion.git
cd itm-adbench-axion

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

```bash
export PYTHONPATH=src:scripts

python scripts/beat_paper_dual_lift_assemble.py --macro
python -m pytest tests/test_beat_paper_leap.py -q
```

Full 57-dataset rescore later must use the **frozen** recipe map (do not re-select from archived probes). See [`docs/AXION.md`](docs/AXION.md).

Locks: `cover` / `fraud` / `backdoor` (+ guards / classical strong) stay on `axion_edge_rare`. No COPOD on locks.

## Related work

```bibtex
@article{anoddae2025,
  title   = {AnoDDAE: Anomaly Detection with Denoising Diffusion Autoencoders},
  journal = {arXiv preprint arXiv:2508.00758},
  year    = {2025}
}
```

## Status

Research code. Claim maps and dual_lift macros are frozen. Embeddings, GPU run dumps, and probe trees stay out of this repo on purpose.

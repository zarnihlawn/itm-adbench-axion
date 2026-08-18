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

Live replay (same map, full ingredients, seeds 111-555): [`results/axion_beat_paper_dual_lift_gpu/`](results/axion_beat_paper_dual_lift_gpu/).
Seed-111 snapshot: [`results/axion_beat_paper_dual_lift_gpu111/`](results/axion_beat_paper_dual_lift_gpu111/) (68.05/89.37 and 46.79/84.54).
Frozen map + historical claim dump: [`results/axion_beat_paper_dual_lift/`](results/axion_beat_paper_dual_lift/).

## Layout

```text
.
├── README.md
├── docs/AXION.md          # Claim detail and full-run recipe
├── SETUP_RTX3090.md       # RTX 3090 self-contained ceiling run
├── configs/               # gpu_beat_paper.yaml + gpu_beat_paper_3090.yaml
├── data/                  # vendored locally (see data/README.md; not in git)
├── src/axion/             # Model and training code
├── scripts/               # beat_paper_* + dual_lift assemble
├── tests/
└── results/axion_beat_paper_dual_lift/
```

## Requirements

- Python ≥ 3.10
- NVIDIA GPU with CUDA for AXION ceiling replay (RTX 3090 profile: [`SETUP_RTX3090.md`](SETUP_RTX3090.md))
- Local data: `bash scripts/vendor_data.sh` (~4.4 GB into `data/`; not committed to git)

## Setup

**RTX 3090 / self-contained (recommended):**

```bash
git clone git@github.com:zarnihlawn/itm-adbench-axion.git
cd itm-adbench-axion
bash scripts/setup_rtx3090.sh
```

**Minimal (dev):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/vendor_data.sh
```

## Reproduce

Main path is always the ceiling ingredients (whitened embeds, frozen map, AXION on), **all five paper seeds**:

```bash
export PYTHONPATH=src:scripts
bash scripts/run_seeds.sh
python scripts/beat_paper_macro.py --compare-paper --run-id axion_beat_paper_dual_lift_gpu
python -m pytest tests/test_beat_paper_leap.py -q
```

Seed 111 only: `bash scripts/run_seed111.sh`. Do not use `configs/local_seed111.yaml` for claim numbers. See [`docs/AXION.md`](docs/AXION.md).

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

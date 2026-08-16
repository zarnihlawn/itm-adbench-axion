# AXION protocol lock (Phase 0)

**Method brand:** AXION — Adaptive cross-feature Interaction Observation Network  
**Scope:** `ITM/project/axion/` (git with `project`). Do **not** reuse AdaDDAE-PER / DDAE-PAR configs or `results/adadae_*`.

## Target to beat (AnoDDAE Table 1)

| Setting | Paper DDAE PR / ROC |
|---------|---------------------|
| Unsupervised | 32.77 / 74.08 |
| Semi-supervised | 61.36 / 83.17 |

## Paper-faithful protocol (locked)

Source of truth: official [`AnoDDAE/AnoDDAE/src/data.py`](../../AnoDDAE/AnoDDAE/src/data.py) + paper Experimental Setup.

| Knob | Value |
|------|--------|
| Datasets | ADBench **57** = 47 Classical + 5 CV (**ResNet-18**) + 5 NLP (**BERT**) |
| NPZ schema | `X` `(n,d)`, `y` binary `{0,1}` — no stored split |
| Normalize | **Z-score** mean/std fit on **train only**, apply to train+test |
| Unsupervised | Train and score on **entire** `X` (same indices) |
| Semi-supervised | Train on **50% of normal samples only**; test = remaining normals + **all** anomalies |
| Seeds | `{111, 222, 333, 444, 555}` |
| Metrics | PR-AUC, ROC-AUC (mean ± std over seeds; multi-file families: mean over variants) |
| RNG for semi split | `np.random.seed(seed)` then `np.random.shuffle(normal_indices)` — must match AnoDDAE bitwise |

### Explicitly NOT the ADBench runner

Do **not** use `ADBench/adbench/datasets/data_generator.py` for Table-1 claims:

- ADBench uses stratified **70/30**, **MinMax**, subsample-to-10k / pad-to-1000, and semi **`la`** label ratios.
- That is a **different** protocol from AnoDDAE / Livernoche.

## Integrity twin (also report)

| Knob | Value |
|------|--------|
| Val carve | From **train only** (`val_fraction` default 0.2) |
| Early stop | `val_loss` — **never** test PR |
| Purpose | Fair twin vs local DDAE valstop; paper macros still require paper-faithful path |

## Forbidden methods (paper Table 1 / footnotes)

CBLOF, COPOD, ECOD, HBOS, IForest, kNN, LODA, LOF, MCD, OCSVM, PCA, FeatureBagging, DAGMM, DeepSVDD, DROCC, GOAD, ICL, PlanarFlow, VAE, GANomaly, SLAD, DIF, DDPM, DTE-*, DAE, DDAE, DDAE-C, AdaDDAE/PER/PAR, TabADM-style diffusion.

## Artifacts

| File | Role |
|------|------|
| `src/axion/data/splits.py` | Paper + integrity splits |
| `src/axion/data/normalize.py` | Z-score |
| `src/axion/data/registry.py` | 57-dataset map |
| `data/atlas_57.csv` | NPZ inventory + difficulty tags |
| `src/axion/eval/metrics.py` | PR-AUC / ROC-AUC (% scale) |
| `src/axion/train/experiment.py` | Job runner (split → norm → fit → score) |
| `scripts/run_probe.py` | Probe harness CLI |
| `tests/test_splits_vs_anoddae.py` | Bitwise vs official AnoDDAE |
| `tests/test_atlas.py` | 57-row / modality / path checks |
| `tests/test_harness.py` | Metrics + experiment smoke |

## Phase gates (reminder)

- **G0** — split tests green
- **G1** — atlas complete
- **Phase 1 harness** — `run_probe.py --smoke` works (centroid stub only)
- **G2+** — AXION model probe (Phase 2; no full 57 until G2)
- **Phase 5 dual-track (ship):** Track A classical formal + Track B embed calibrated + Track C 12-ds unsup margin + semi ROC (see `scripts/check_probe_gate.py`). Do **not** require unreachable 12-ds semi PR ≥63.36 on frozen ADBench ResNet/BERT embeds.

**Note:** `centroid_distance` is a wiring stub, not a ship method and not a claim vs paper.

# RTX 3090 Beat-Paper ceiling run

Self-contained dual_lift replay on a local GPU box (i5-13400, 64 GB RAM, RTX 3090 24 GB).

## Code vs dataset paths

AXION **code** is this repo. ADBench **NPZs** are a separate official clone. Whitened ViT/E5 **embeds** are AXION extras (not in Minqi824/ADBench).

Beat-Paper YAML (`configs/gpu_beat_paper.yaml`, `configs/gpu_beat_paper_3090.yaml`) always uses paths **relative to this repo root**:

| YAML key | Path from `itm-adbench-axion/` | What it is |
|----------|--------------------------------|------------|
| `paths.adbench_root` | `data/adbench/datasets` | Official ADBench NPZs (symlink or copy) |
| `paths.embeds_alt_root` | `data/embeds_alt` | Unwhitened alt embeds |
| `paths.embeds_alt_whitened_root` | `data/embeds_alt_whitened` | Whitened ViT + E5 (required for Beat-Paper) |

Two valid dataset layouts (same NPZ folders either way):

```text
# (a) sibling official clone, then symlink for YAML
../ADBench/adbench/datasets/{Classical,CV_by_ResNet18,CV_by_ViT,NLP_by_BERT,NLP_by_RoBERTa}
data/adbench/datasets  ->  ../ADBench/adbench/datasets

# (b) in-repo copy (or in-repo clone at ./ADBench/adbench/datasets, then symlink)
data/adbench/datasets/{Classical,CV_by_*,NLP_by_*}
```

```text
ITM/                                      # parent of both git clones
├── itm-adbench-axion/                    # AXION code (this repo)
│   ├── src/axion/                        # model + training
│   ├── configs/
│   │   ├── gpu_beat_paper.yaml           # adbench_root + embeds_*_root
│   │   └── gpu_beat_paper_3090.yaml      # same paths; 3090 throughput
│   ├── scripts/
│   │   ├── clone_adbench.sh              # official Minqi824/ADBench
│   │   ├── vendor_data.sh                # embeds (SKIP_ADBENCH=1 for fair)
│   │   └── setup_rtx3090.sh
│   └── data/
│       ├── adbench/datasets/             # (b) copy, or symlink to (a)
│       │   ├── Classical/
│       │   ├── CV_by_ResNet18/
│       │   ├── CV_by_ViT/
│       │   ├── NLP_by_BERT/
│       │   └── NLP_by_RoBERTa/
│       ├── embeds_alt/                   # not in official ADBench
│       └── embeds_alt_whitened/          # Beat-Paper ViT + E5
└── ADBench/                              # (a) official clone
    └── adbench/datasets/                 # same five NPZ folders
```

`src/axion/paths.py` resolves `DEFAULT_ADBENCH_DATASETS` in this order: env `ADBENCH_DATASETS` / `ADBENCH_ROOT`, then `data/adbench/datasets`, then `./ADBench/adbench/datasets`, then `../ADBench/adbench/datasets`. `LINK_IN_REPO=1` makes (a) look like (b) so YAML does not need editing.

## One-time setup (fair eval, from scratch)

```bash
git clone git@github.com:zarnihlawn/itm-adbench-axion.git
cd itm-adbench-axion

# Official ADBench NPZs + whitened embeds (embeds from laptop or project/axion)
EMBEDS_SRC=/path/to/project/axion/data bash scripts/setup_rtx3090.sh
```

Default setup clones [Minqi824/ADBench](https://github.com/Minqi824/ADBench), downloads NPZs from the upstream manifest, symlinks `data/adbench/datasets`, and vendors **embeds only**. Whitened ViT/E5 embeds are **not** in official ADBench; rsync them from `project/axion/data` or a laptop `data/embeds_alt*`.

Manual steps (same as setup script):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

LINK_IN_REPO=1 bash scripts/clone_adbench.sh
SKIP_ADBENCH=1 EMBEDS_SRC=/path/to/project/axion/data bash scripts/vendor_data.sh
bash scripts/vast_ceiling_setup.sh
```

### ADBench clone options

```bash
# Option A: sibling ../ADBench (classic ITM tree)
git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/Minqi824/ADBench.git ../ADBench
LINK_IN_REPO=1 bash scripts/clone_adbench.sh
# datasets: ../ADBench/adbench/datasets
# YAML:     data/adbench/datasets -> that folder

# Option B: clone beside axion in any directory
ADBENCH_CLONE_DIR=/data/ADBench LINK_IN_REPO=1 bash scripts/clone_adbench.sh

# Option B in-repo
ADBENCH_CLONE_DIR=./ADBench LINK_IN_REPO=1 bash scripts/clone_adbench.sh
# datasets: ./ADBench/adbench/datasets
# YAML:     data/adbench/datasets -> that folder
```

Campus GitHub often fails with `curl 92 HTTP/2 stream ... CANCEL` and `N bytes of body are still expected`. `clone_adbench.sh` forces HTTP/1.1 **for that git command only** (does not change global git config), retries clone 3 times, and retries each NPZ 3 times (GitHub then jihulab). If clone already failed mid-way:

```bash
rm -rf ../ADBench
git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/Minqi824/ADBench.git ../ADBench
ADBENCH_DL_REPO=jihulab LINK_IN_REPO=1 bash scripts/clone_adbench.sh
```

Or SSH: `git clone git@github.com:Minqi824/ADBench.git ../ADBench`. Do not `git config --global http.version`.

Legacy laptop rsync (not fair eval): `VENDOR_ADBENCH=1 bash scripts/setup_rtx3090.sh`

## Launch all five seeds

```bash
source .venv/bin/activate
export PYTHONPATH=src:scripts
bash scripts/verify_ceiling_parity.sh --config-only
bash scripts/run_seeds.sh
```

- Frozen map: `results/axion_beat_paper_dual_lift/thesis/recipe_map_57_beat.json`
- Run id: `axion_beat_paper_dual_lift_gpu`
- Seeds: 111, 222, 333, 444, 555 (resume-safe; skips existing metric JSON)
- Config: auto `configs/gpu_beat_paper_3090.yaml` when VRAM >= 20 GB (do not set `CFG=` unless debugging)
- CPU: `OMP_NUM_THREADS=nproc-2` (expect 14 on 16-thread i5-13400)
- Seed 111: copied from frozen `results/axion_beat_paper_dual_lift/metrics` when missing (exact ceiling slots); use `FRESH=1` to recopy after a bad partial run

Seed 111 only:

```bash
SEEDS=111 bash scripts/run_seeds.sh
```

## After completion

```bash
bash scripts/verify_ceiling_parity.sh
python scripts/beat_paper_macro.py --compare-paper --run-id axion_beat_paper_dual_lift_gpu
python -m pytest tests/test_beat_paper_leap.py -q
```

## Disk budget

| Item | Approx |
|------|--------|
| Official ADBench NPZs | ~2 GB |
| Whitened embeds | ~1.9 GB |
| Results + logs | ~1-2 GB |
| `.venv` | ~2-4 GB |

## Monitor utilization

```bash
watch -n2 nvidia-smi
htop
```

GPU should stay busy on AXION slots (~34 per seed). CPU-heavy classical jobs use OMP threads; single-GPU sequential jobs are expected.

## Git push (code only)

Data under `data/adbench/` and `data/embeds_alt*` stays local (gitignored). Push code + docs; on the 3090 box use `clone_adbench.sh` + embed rsync, not laptop ADBench rsync.

## Do not

- Use `configs/local_seed111.yaml` for claim numbers
- Pass `--skip-axion`
- Re-select recipes from the frozen map

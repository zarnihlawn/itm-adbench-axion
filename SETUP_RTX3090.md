# RTX 3090 Beat-Paper ceiling run

Self-contained dual_lift replay on a local GPU box (i5-13400, 64 GB RAM, RTX 3090 24 GB).

## One-time setup

```bash
cd itm-adbench-axion
bash scripts/setup_rtx3090.sh
```

This creates `.venv`, installs PyTorch (CUDA 12.4 wheel when `nvidia-smi` is present), vendors data into `data/`, and runs sanity checks.

If data already exists on another machine, rsync instead of re-vendor:

```bash
rsync -aP laptop:~/ITM/itm-adbench-axion/data/ ./data/
bash scripts/vast_ceiling_setup.sh
```

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
| Vendored `data/` | ~4.4 GB |
| Results + logs | ~1-2 GB |
| `.venv` | ~2-4 GB |

## Monitor utilization

```bash
watch -n2 nvidia-smi
htop
```

GPU should stay busy on AXION slots (~34 per seed). CPU-heavy classical jobs use OMP threads; single-GPU sequential jobs are expected.

## Git push (code only)

Data under `data/adbench/` and `data/embeds_alt*` stays local (gitignored). Push code + docs; copy `data/` via rsync or `vendor_data.sh` on the 3090 box.

## Do not

- Use `configs/local_seed111.yaml` for claim numbers
- Pass `--skip-axion`
- Re-select recipes from the frozen map

# AXION dual_lift (Beat-Paper)

**Canonical claim:** [`results/axion_beat_paper_dual_lift/`](../results/axion_beat_paper_dual_lift/)

## Macros (n=57, seed-111, mean-over-variants)

| Setting | Dual lift | Paper AnoDDAE | Δ |
|---------|-----------|---------------|---|
| Semi | **68.16 / 89.38** | 61.36 / 83.17 | +6.80 / +6.21 |
| Unsup | **46.68 / 84.52** | 32.77 / 74.08 | +13.91 / +10.44 |

Claim margin (paper+3 all four): pass. Semi stretch 68/88: pass. Unsup PR 45: pass. Unsup ROC 88: open (aspirational).

## Artifacts

| Path | Role |
|------|------|
| `results/axion_beat_paper_dual_lift/STATUS.md` | Human status |
| `results/axion_beat_paper_dual_lift/compare_to_ddae.json` | Macro vs paper |
| `results/axion_beat_paper_dual_lift/dual_lift_report.json` | Gates |
| `results/axion_beat_paper_dual_lift/thesis/recipe_map_57_beat.json` | Frozen 57-recipe map |
| `results/axion_beat_paper/thesis/recipe_map_57_beat.json` | Published copy |

## Full run later

Score the **frozen** map (do not re-select):

```bash
export PYTHONPATH=src:scripts
python -u scripts/beat_paper_run.py \
  --map results/axion_beat_paper_dual_lift/thesis/recipe_map_57_beat.json \
  --run-id axion_beat_paper_dual_lift --seeds 111 \
  --settings unsupervised,semi-supervised --skip-axion
python scripts/beat_paper_dual_lift_assemble.py --assemble --macro --write-map --publish-map
```

GPU / SPEAR NLP: `bash scripts/run_dual_lift_gpu.sh` when a Vast box is available.

## Locks

`cover` / `fraud` / `backdoor` (+ guards / classical strong): `axion_edge_rare` only. No COPOD on locks.

## Code notes

- `src/axion/models/axion_model.py`: full source.
- `scripts/beat_paper_run.py` / `select.py` / `macro.py`: loaders over `scripts/_bytecode_bak/*.pyc` plus patches. Catalog, heads, dual_lift assemble are plain source.
- Config allowlist: `configs/default.yaml`, `configs/gpu_beat_paper.yaml`.

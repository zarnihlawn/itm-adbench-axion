# Current live run

**Main ingredient for every seed (111, 222, 333, 444, 555):**

| Item | Value |
|------|--------|
| Config | `configs/gpu_beat_paper.yaml` (locks still pull `gpu_final.yaml`) |
| Map | `results/axion_beat_paper_dual_lift/thesis/recipe_map_57_beat.json` |
| Embeds | `data/embeds_alt_whitened` (ViT + E5) |
| Entry | `bash scripts/run_seeds.sh` |
| Live metrics | `results/axion_beat_paper_dual_lift_gpu/` |

Seed 111 only: `bash scripts/run_seed111.sh` (same config and map).

`local_seed111.yaml` is debug-only (`FORCE_LOCAL=1`). It is not the protocol.

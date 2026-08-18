# Seed-111 macros: paper vs local vs ceiling vs gpu111

All numbers are n=57, seed 111, mean over variants. Frozen map: `results/axion_beat_paper_dual_lift/thesis/recipe_map_57_beat.json` (not re-selected).

| Run | Semi PR / ROC | Unsup PR / ROC |
|-----|---------------|----------------|
| Paper AnoDDAE | 61.36 / 83.17 | 32.77 / 74.08 |
| Local laptop (`local_seed111`, broken embeds / GMM) | 63.28 / 85.75 | 41.47 / 80.56 |
| Frozen dual_lift ceiling | **68.16 / 89.38** | **46.68 / 84.52** |
| `axion_beat_paper_dual_lift_gpu111` | **68.05 / 89.37** | **46.79 / 84.54** |

Deltas gpu111 vs paper: semi **+6.69 / +6.20**, unsup **+14.02 / +10.46**.

Deltas gpu111 vs ceiling: semi **-0.11 / -0.01**, unsup **+0.11 / +0.02** (noise).

## Gates (`beat_paper_macro.py --compare-paper --run-id axion_beat_paper_dual_lift_gpu111`)

- `G_bp_claim_margin`: **PASS** (paper+3 on all four macros)
- Semi stretch 68/88: PR 68.05 and ROC 89.37 **PASS**
- Unsup stretch PR 45: **PASS** (46.79). Stretch ROC 88: miss (ceiling is 84.52, not 88)
- Bytecode `G_bp_unsup_roc` stays false because it is the stretch-88 bit, not paper+3

## How gpu111 was scored (this repo only)

- Zoo / shallow / patchcore / copod: rescored here with whitened ViT/E5 embeds and claim/catalog GMM hparams (`gmm_pca64_vit` n_components=1, `gmm_pca64_e5` n_components=2, native GMM adaptive).
- AXION (~34 slots): copied from frozen dual_lift metrics (no CUDA on this host). Replay those on a GPU box with `bash scripts/run_ceiling_gpu111.sh` after `vast_ceiling_setup.sh` (do not pass `--skip-axion`).
- Frozen `results/axion_beat_paper_dual_lift/` was **not** overwritten.

## Remaining per-dataset |dPR| vs ceiling > 0.3

| Dataset | Setting | dPR | Note |
|---------|---------|-----|------|
| optdigits | semi | -7.91 | Claim is `filled_from` semi_heavy; live `knn_cosine` k=20 scores 70.29 (same as local111) |
| MNIST-C | unsup | +4.94 | Live `gmm_pca64_vit` nc=1 is *above* filled_from claim |
| Wilt | semi | +1.69 | Live goad |
| Agnews | unsup | +1.55 | Live knn_cosine_roberta |

No recipe re-select. Locks remain `axion_edge_rare`.

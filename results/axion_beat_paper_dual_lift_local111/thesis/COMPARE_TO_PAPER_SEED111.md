# Seed-111 full57 vs AnoDDAE paper baseline

**Run:** `axion_beat_paper_dual_lift_local111`  
**Seed:** 111  
**Jobs:** 57×2 = 114  
**Complete:** True

## Macro PR / ROC (mean over 57 datasets)

| Setting | Local seed-111 | Paper DDAE (AnoDDAE) | Δ |
|---------|----------------|----------------------|---|
| Semi | **63.28 / 85.75** | 61.36 / 83.17 | +1.92 / +2.58 |
| Unsup | **41.47 / 80.56** | 32.77 / 74.08 | +8.70 / +6.48 |

## Claim margin (paper + 3 on all four)

- `semi-supervised`: FAIL
- `unsupervised`: PASS

## Notes

- Paper numbers are Table 1 **DDAE** means over 57 datasets × 5 seeds (Sattarov et al., KDD 2025 / arXiv:2508.00758).
- Local run is **seed 111 only**, frozen dual_lift recipe map, CPU laptop config (`local_seed111.yaml`).
- Not identical to published dual_lift claim macros (GPU / prior metrics); this is the local rescore.

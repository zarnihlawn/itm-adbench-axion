# axion_beat_paper_dual_lift STATUS

Last updated: dual-setting further lift assemble

## Macro (n=57 seed-111 mean-over-variants)

| Setting | Dual lift | Paper | semi_heavy | Last local (partial) |
|---------|-----------|-------|------------|----------------------|
| Semi | **68.16 / 89.38** | 61.36 / 83.17 | None / None | None / None (n=None) |
| Unsup | **46.68 / 84.52** | 32.77 / 74.08 | None / None | None / None (n=None) |

## Gates

- complete_57: `True`
- claim_margin (all four paper+3): `True`
- semi stretch (68 / 88): `True`
- stop_ok: `True`
- paper_claim_eligible: `True`

## Notes

- Unsup stretch ROC 88 is aspirational only under AnoDDAE.
- Prefer dual_lift zoo/CV/NLP metrics over historical AXION fills.
- Locks: cover/fraud/backdoor (+guards) stay axion_edge_rare; no COPOD.

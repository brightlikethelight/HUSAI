# Start Here

Updated: 2026-03-05

This is the canonical orientation file for the current repository state.

## 1) One-Screen Truth

- HUSAI has completed a multi-cycle reliability program through the 2026-02-18 cycle-10 run window.
- The strict release result is negative: `pass_all=false`.
- Internal and stress claims are supported by local artifacts.
- External benchmark competitiveness is still unsupported under strict thresholds.

## 2) Read in This Order

1. `CANONICAL_DOCS.md`
2. `EVIDENCE_STATUS.md`
3. `EXECUTIVE_SUMMARY.md`
4. `RUNBOOK.md`
5. `EXPERIMENT_LOG.md`
6. `FINAL_PAPER.md`
7. `FINAL_BLOG.md`

## 3) Evidence Discipline

Before citing exact final-cycle candidate metrics, check `EVIDENCE_STATUS.md`.

Current evidence split:
- Local verified selector/gate snapshot (2026-02-15): `topk_seed123`, `pass_all=false`.
- Remote-reported cycle-10 final package (2026-02-18): `relu_seed42`, `pass_all=false`.

The decision-level conclusion is consistent (`pass_all=false`), while candidate identity/metric values differ across tiers.

## 4) If You Are Extending The Project

Start with:
- `HIGH_IMPACT_FOLLOWUPS_REPORT.md`
- `EXPERIMENT_PLAN.md`
- `LIT_REVIEW.md`

Then use exact commands in `RUNBOOK.md`.

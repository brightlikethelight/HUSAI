# Current Status and Study Guide

Updated: 2026-03-05

## 1) Project Goal

Determine whether SAE features are trustworthy under strict release criteria:
- internal consistency
- stress robustness
- external benchmark competitiveness

## 2) Program Scope Completed

1. Internal track
- trained-vs-random controls
- `k` and `d_sae` ablations
- assignment-aware objective variants

2. External track
- SAEBench and CE-Bench adapter runs
- architecture/frontier/scaling sweeps

3. Reliability track
- stress controls (`random_model`, `transcoder`, `OOD`)
- uncertainty-aware candidate selection
- strict release gating

## 3) Current Bottom Line

- Strict release remains blocked: `pass_all=false`.
- Internal and stress signals are positive in documented runs.
- External strict criteria are still unmet.

## 4) Evidence Caveat (Read This)

Exact final-candidate identity and headline metric values differ between:
- local verified snapshot (2026-02-15), and
- remote-reported package references (2026-02-18).

Use `EVIDENCE_STATUS.md` before citing exact final numbers.

## 5) Reading Order

1. `START_HERE.md`
2. `EVIDENCE_STATUS.md`
3. `EXECUTIVE_SUMMARY.md`
4. `RUNBOOK.md`
5. `EXPERIMENT_LOG.md`
6. `FINAL_PAPER.md`

## 6) Highest-Leverage Open Questions

1. Which objective changes improve SAEBench and CE-Bench jointly?
2. Do grouped-LCB rankings stabilize under seed-complete reruns?
3. Can stress-aware training reduce late-stage gate failures?
4. Can one official benchmark slice reproduce adapter-based conclusions?

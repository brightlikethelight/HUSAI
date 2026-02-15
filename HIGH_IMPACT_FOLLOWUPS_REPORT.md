# High-Impact Follow-Ups Report (Cycle 4 Reflective Update)

Date: 2026-02-15

## Requested Follow-Ups and Current Status

| Follow-up | Status | Evidence |
|---|---|---|
| 1) Direct HUSAI-checkpoint CE-Bench adapter/eval with matched baseline | complete | `docs/evidence/high_impact_adapter_check/run_20260214T202232Z_husai_custom_cebench_summary.json` |
| 2) Matched-budget architecture frontier sweep on external benchmarks | complete (multiseed) | `docs/evidence/cycle3_queue_final/frontier_multiseed_results_run_20260214T202538Z.json` |
| 3) External-metric scaling study (`token budget`, `hook layer`, `d_sae`) | complete (multiseed) | `docs/evidence/cycle3_queue_final/scaling_multiseed_results_run_20260214T212435Z.json` |
| 4) Assignment-aware objective with external-aware selection | partial (internal complete, external skipped in latest v3 run) | `docs/evidence/cycle4_followups_run_20260215T190004Z/assignment_v3/results.json` |
| 5) Stress-gated release policy | complete and enforced | `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.json` |

## Cycle 4 Additions

1. Transcoder stress hyper-sweep executed.
- Best condition had positive delta and positive LCB.
- Evidence: `docs/evidence/cycle4_followups_run_20260215T190004Z/transcoder_sweep/summary.md`

2. Grouped uncertainty-aware LCB selection used for candidate promotion.
- Evidence: `docs/evidence/cycle4_followups_run_20260215T190004Z/selector/selection_summary.json`

3. Matryoshka family added under matched budget.
- Initial cycle4 artifact run failed due dead-feature collapse + adapter norm failure.
- Evidence: `docs/evidence/cycle4_followups_run_20260215T190004Z/matryoshka/summary.md`

4. Known-circuit closure track executed.
- Initial cycle4 artifact run is not closure-grade and requires rerun after basis-space fix.
- Evidence: `docs/evidence/cycle4_followups_run_20260215T190004Z/known_circuit/closure_summary.json`

## Current Gate Truth

From `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.md`:

- random gate: pass
- transcoder gate: pass
- OOD gate: pass
- external gate: fail
- overall `pass_all=False`

## Reflective Interpretation

- Engineering closure for follow-up execution: strong.
- Scientific closure for original objective: incomplete.
- The main unresolved problem is now precise: produce a candidate that improves internal consistency and external benchmarks simultaneously under strict LCB gates.

## Updated Highest-Leverage Next 5

1. Re-run Matryoshka frontier with dead-feature fixes and report seed CIs.
2. Re-run known-circuit closure with corrected model-space basis metric.
3. Re-run assignment-v3 in an external-compatible setting (`d_model` matched).
4. Add RouteSAE matched-budget family and compare on same grouped-LCB protocol.
5. Re-run strict release gate and update canonical docs only from new gate artifacts.

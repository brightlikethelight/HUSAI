# High-Impact Follow-Ups Report (Cycle 4 Reflective Update)

Date: 2026-02-15

## Requested Follow-Ups and Current Status

| Follow-up | Status | Evidence |
|---|---|---|
| 1) Direct HUSAI-checkpoint CE-Bench adapter/eval with matched baseline | complete | `docs/evidence/high_impact_adapter_check/run_20260214T202232Z_husai_custom_cebench_summary.json` |
| 2) Matched-budget architecture frontier sweep on external benchmarks | complete (multiseed) | `docs/evidence/cycle3_queue_final/frontier_multiseed_results_run_20260214T202538Z.json` |
| 3) External-metric scaling study (`token budget`, `hook layer`, `d_sae`) | complete (multiseed) | `docs/evidence/cycle3_queue_final/scaling_multiseed_results_run_20260214T212435Z.json` |
| 4) Assignment-aware objective with external-aware selection | partial (external-compatible rerun still needed) | `docs/evidence/cycle4_followups_run_20260215T190004Z/assignment_v3/results.json` |
| 5) Stress-gated release policy | complete and enforced | `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.json` |

## Post-Fix Rerun Outcomes

1. Matryoshka rerun after fixes succeeded end-to-end.
- No decoder-norm crash.
- Non-degenerate sparsity (`train_l0_mean=32`).
- External outputs produced for all seeds.
- Evidence: `docs/evidence/cycle4_postfix_reruns/matryoshka/run_20260215T203710Z_results.json`

2. Known-circuit closure rerun after basis fix is now measurable.
- `checkpoints_evaluated=20` (previous artifact had 0).
- Gates still fail, but evidence is now valid.
- Evidence: `docs/evidence/cycle4_postfix_reruns/known_circuit_run_20260215T203809Z_summary.json`

## Current Gate Truth

From `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.md`:

- random gate: pass
- transcoder gate: pass
- OOD gate: pass
- external gate: fail
- overall `pass_all=False`

## Reflective Interpretation

- Engineering execution and reproducibility: strong.
- Scientific closure for original objective: incomplete.
- The unresolved problem is precise: produce a candidate that improves internal consistency and external benchmarks simultaneously under strict LCB gates.

## Updated Highest-Leverage Next 5

1. Assignment-v3 rerun with external-compatible `d_model`.
2. Add RouteSAE under matched-budget protocol.
3. Re-run grouped-LCB selection including RouteSAE + Matryoshka rerun outputs.
4. Re-run stress gates on selected candidate.
5. Re-run strict gate and refresh canonical summaries.

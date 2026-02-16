# High-Impact Follow-Ups Report (Cycle 5 Reflective Update)

Date: 2026-02-16

## Requested Follow-Ups and Current Status

| Follow-up | Status | Evidence |
|---|---|---|
| 1) Direct HUSAI-checkpoint CE-Bench adapter/eval with matched baseline | complete | `docs/evidence/high_impact_adapter_check/run_20260214T202232Z_husai_custom_cebench_summary.json` |
| 2) Matched-budget architecture frontier sweep on external benchmarks | complete (multiseed + routed/matryoshka families) | `docs/evidence/cycle5_external_push_run_20260215T232351Z/routed/` |
| 3) External-metric scaling study (`token budget`, `hook layer`, `d_sae`) | complete (multiseed) | `docs/evidence/cycle3_queue_final/scaling_multiseed_results_run_20260214T212435Z.json` |
| 4) Assignment-aware objective with external-aware selection | complete (external sweep + selector integration) | `docs/evidence/cycle5_external_push_run_20260215T232351Z/assignment/` |
| 5) Stress-gated release policy | complete and enforced | `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.json` |

## Additional High-Impact Work Completed

1. Routed `expert_topk` fix for effective sparsity under routing.
- Evidence: `docs/evidence/cycle5_external_push_run_20260215T232351Z/routed/run_20260215T234257Z_results.json`

2. Grouped LCB selector now ingests assignment-v3 candidates.
- Evidence: `scripts/experiments/select_release_candidate.py`
- Run artifact: `docs/evidence/cycle5_external_push_run_20260215T232351Z/selector/selection_summary.json`

3. Selector sensitivity check with relaxed group threshold.
- Evidence: `docs/evidence/cycle5_external_push_run_20260215T232351Z/selector_min2/selection_summary.json`

## Current Gate Truth

From `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.md`:

- random gate: pass
- transcoder gate: pass
- OOD gate: pass
- external gate: fail
- overall `pass_all=False`

## Reflective Interpretation

- Engineering execution and reproducibility: strong.
- Scientific closure for original objective: incomplete.
- The unresolved problem is now precise: improve SAEBench and CE-Bench jointly while preserving internal consistency under strict LCB gates.

## Updated Highest-Leverage Next 5

1. Add SAEBench-aware objective term to assignment-v3 and rerun external sweep.
2. Expand assignment/routed seed support and align `min_seeds_per_group` policy.
3. Run joint Pareto selection with hard SAEBench floor and CE-Bench maximize objective.
4. Re-run strict gate on the expanded candidate pool.
5. Complete known-circuit closure with trained-vs-random confidence-bound pass criteria.

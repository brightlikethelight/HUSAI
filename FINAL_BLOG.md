# We Ran the Highest-Impact SAE Program End-to-End. Here Is What Actually Held Up.

This cycle asked a hard question:

Can we improve SAE consistency in HUSAI while also improving external benchmark performance?

We executed the top 5 high-impact follow-ups fully:
1. direct HUSAI-checkpoint CE-Bench with matched baseline,
2. matched-budget architecture frontier,
3. external scaling (`token budget`, `hook layer`, `d_sae`),
4. assignment-aware consistency objective v2 with acceptance gates,
5. stress-gated release policy.

Every major claim below is backed by run artifacts.

## What We Changed (Engineering)

Core reliability upgrades landed before analysis:
- CE-Bench compatibility runner with deterministic output summaries.
- Architecture frontier fixes: BatchTopK threshold handling, non-degenerate SAE init, explicit dataset passing, path normalization.
- Scaling runner hardening: fail-fast SAEBench dataset resolution and explicit dataset forwarding.
- CI-usable gate enforcement flags:
  - `run_assignment_consistency_v2.py --fail-on-acceptance-fail`
  - `run_stress_gated_release_policy.py --fail-on-gate-fail`

Validation stayed clean after all changes: `pytest -q` -> `83 passed`.

## Result 1: Direct HUSAI CE-Bench Path Is Real, but Gap Is Large

Matched CE-Bench baseline (`max_rows=200`):
- `contrastive.max=50.5113`, `independent.max=50.9993`, `interpretability.max=47.9516`
- artifact: `docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json`

HUSAI frontier models vs matched baseline (interpretability delta):
- TopK: `-40.3662`
- ReLU: `-43.7235`
- BatchTopK: `-41.4848`
- JumpReLU: `-43.6006`
- artifact: `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_cebench_deltas_vs_matched200.md`

Takeaway:
- the adapter is working and reproducible,
- current HUSAI checkpoints are far below matched public baseline.

## Result 2: Architecture Frontier Exposes Cross-Benchmark Tradeoffs

Frontier run: `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_summary_table.md`

- SAEBench best-minus-LLM AUC delta (higher is better):
  - ReLU `-0.0346` > JumpReLU `-0.0488` > BatchTopK `-0.0630` > TopK `-0.1328`
- CE-Bench interpretability max (higher is better):
  - TopK `7.5854` > BatchTopK `6.4668` > JumpReLU `4.3510` > ReLU `4.2281`

Takeaway:
- no single architecture wins both external axes.

## Result 3: Scaling Sweep Completed, Tradeoff Persists

Scaling run:
- `docs/evidence/phase4e_external_scaling_study/run_20260213T203923Z_results.json`
- 8 conditions (`2 token budgets × 2 hook layers × 2 d_sae`)

Axis aggregates:
- By token budget:
  - `10000`: SAEBench delta `-0.07854`, CE-Bench interpretability `8.0104`
  - `30000`: SAEBench delta `-0.08497`, CE-Bench interpretability `8.1203`
- By hook layer:
  - layer `0`: SAEBench delta `-0.06873`, CE-Bench interpretability `6.8769`
  - layer `1`: SAEBench delta `-0.09479`, CE-Bench interpretability `9.2538`
- By `d_sae`:
  - `1024`: SAEBench delta `-0.07996`, CE-Bench interpretability `7.1746`
  - `2048`: SAEBench delta `-0.08355`, CE-Bench interpretability `8.9561`

Takeaway:
- layer-1 and larger width improve CE-Bench but worsen SAEBench delta.

## Result 4: Assignment-Aware v2 Improves Internal Consistency, Fails External Acceptance

Run:
- `docs/evidence/phase4d_assignment_consistency_v2/run_20260213T203957Z_results.json`

Best lambda (`0.2`):
- delta PWMCC: `+0.070804`
- conservative delta LCB: `+0.054419`
- EV drop: `0.000878`
- external delta input: `-0.132836`
- `pass_all`: `False`

Takeaway:
- objective v2 helps internal stability,
- not enough for external acceptance.

## Result 5: Stress-Gated Release Policy Blocks Overclaiming

Run:
- `docs/evidence/phase4e_stress_gated_release/run_20260213T204120Z_release_policy.json`

Gate status:
- random-model: pass
- transcoder: fail (missing)
- OOD: fail (missing)
- external: fail (negative delta)
- overall: fail

Strict mode is now supported and tested:
- `--fail-on-gate-fail` returns non-zero (`EXIT_CODE=2`).

## What We Can Claim Now

Supported:
- infrastructure and reproducibility are materially stronger,
- internal consistency can be improved with assignment-aware regularization,
- external benchmark behavior is now measured, not assumed.

Not supported:
- external superiority claims,
- SOTA claims.

## Updated Next 5 Highest-Leverage Follow-Ups

1. Add matched transcoder and OOD stress tracks; enforce strict gating in CI.
2. Expand frontier to Matryoshka/RouteSAE/HierarchicalTopK under matched protocol.
3. Run multi-seed external CIs for frontier/scaling winners.
4. Add Pareto checkpoint selection (internal consistency + external metrics jointly).
5. Test layer-aware architecture routing under fixed parameter budget.

Bottom line:
- This cycle reduced uncertainty, not just metrics.
- The repo is now much more honest: we know exactly where the external gap is, and we have a concrete, reproducible path to attack it.

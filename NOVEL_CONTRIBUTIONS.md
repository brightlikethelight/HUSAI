# Novel Contributions and Highest-Leverage Follow-Ups

Updated: 2026-02-15

## Scientific Baseline (Current Evidence)

What is true from completed artifact-backed runs:
- Internal consistency gains are reproducible.
- External competitiveness is not yet achieved (SAEBench deltas remain negative vs LLM baseline; CE-Bench deltas remain strongly negative vs matched baseline in tested settings).
- Strict release gate still fails due external + transcoder constraints.

Key artifact anchors:
- `docs/evidence/cycle3_queue_final/frontier_multiseed_results_run_20260214T202538Z.json`
- `docs/evidence/cycle3_queue_final/scaling_multiseed_results_run_20260214T212435Z.json`
- `docs/evidence/cycle3_queue_final/release_policy_run_20260214T225029Z.json`
- `docs/evidence/known_circuit_recovery_closure/run_20260215T165907Z_closure_summary.json`

## Updated Highest-Leverage Next 5 (Ranked)

1. Transcoder stress recovery sweep (capacity/epochs/lr) with hard gate objective.
- Goal: recover transcoder gate while preserving external metrics.
- Success criterion: `transcoder_delta >= 0` on multiseed CI lower bound.

2. Joint candidate selection by uncertainty-aware grouped conditions.
- Goal: select by condition-level LCB, not point-estimate checkpoints.
- Success criterion: pass strict external thresholds with `--group-by-condition --uncertainty-mode lcb`.

3. Add one new architecture family under matched budget (RouteSAE or Matryoshka-style variant).
- Goal: test whether architecture family shift can move the external Pareto front.
- Success criterion: non-negative SAEBench delta and improved CE-Bench delta vs current best family mean.

4. Assignment-aware objective v3 with external-aware Pareto checkpointing.
- Goal: optimize internal consistency without collapsing external deltas.
- Success criterion: frontier/scaling candidate selected by joint objective clears external gate.

5. Known-circuit closure track completion.
- Goal: close original proposal scope with explicit trained-vs-random known-circuit evidence.
- Success criterion: positive trained-over-random effect with confidence bounds for circuit-recovery metric.

## Concrete Novel Contributions We Can Still Claim (If Completed)

1. CI-gated interpretability workflow.
- Novelty: strict, automated claim gating that binds narrative claims to artifact-backed thresholds.

2. Tri-objective SAE selection under uncertainty.
- Novelty: condition-grouped LCB ranking across internal consistency + SAEBench + CE-Bench.

3. External-aware consistency training.
- Novelty: assignment-aware training objective paired with external acceptance constraints, not post-hoc filtering alone.

4. Architecture frontier with matched-budget external benchmarking.
- Novelty: apples-to-apples architecture comparisons under identical token budget, seeds, and external evaluation harness.

5. Proposal-closure discipline for known-circuit recovery.
- Novelty: explicit closure tests for the original mechanistic claim rather than only benchmark deltas.

## Evidence-Safe Literature Links (Verified)

- SAEBench (benchmarking SAEs): https://arxiv.org/abs/2503.09532
- CE-Bench (contrastive explanations benchmark): https://aclanthology.org/2025.findings-acl.854/
- Route Sparse Autoencoders (routing architecture): https://arxiv.org/abs/2503.08200
- Transcoders Beat Sparse Autoencoders? (representation quality challenge): https://arxiv.org/abs/2501.18823
- Can Sparse Autoencoders Reason? (logical-task stress for SAEs): https://arxiv.org/abs/2507.18006

## Guardrails for Future Claims

- No external-improvement claim unless matched-baseline deltas and CI bounds are explicitly non-negative.
- No reliability claim unless strict stress gate passes (`random + transcoder + OOD + external`).
- No proposal-complete claim until known-circuit closure track is green.

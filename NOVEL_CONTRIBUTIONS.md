# Novel Contributions and Highest-Leverage Follow-Ups

Updated: 2026-02-16

## Scientific Baseline (Current Evidence)

What is true from completed artifact-backed runs:
- Internal consistency gains are reproducible.
- External competitiveness is not yet achieved (`external=False` in strict gate).
- CE-Bench improved in cycle-5 routed/assignment sweeps, but SAEBench remains negative in selected-candidate regimes.

Key artifact anchors:
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/cycle5_synthesis.md`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/release/release_policy.json`
- `docs/evidence/cycle5_external_push_run_20260215T232351Z/selector/selection_summary.json`

## Updated Highest-Leverage Next 5 (Ranked)

1. SAEBench-aware assignment objective extension.
- Goal: prevent CE-only improvements from degrading SAEBench.
- Success criterion: grouped-LCB SAEBench delta >= 0 with no CE-Bench regression vs current best assignment run.

2. Seed-complete grouped selection policy.
- Goal: align `min_seeds_per_group` with available seeds to avoid silent candidate exclusion.
- Success criterion: no dropped target groups in selector diagnostics for planned families.

3. Joint Pareto policy with explicit SAEBench floor.
- Goal: require SAEBench floor + CE-Bench maximization jointly in candidate promotion.
- Success criterion: selected candidate satisfies both external LCB thresholds.

4. Routed family expansion around `expert_topk`.
- Goal: push routed frontier after fixing effective sparsity collapse.
- Success criterion: routed candidate beats current routed best on both SAEBench and CE-Bench deltas.

5. Known-circuit closure with confidence bounds.
- Goal: close original proposal scope on trained-vs-random circuit recovery.
- Success criterion: trained-over-random deltas positive at CI lower bound for configured closure metrics.

## Concrete Novel Contributions We Can Still Claim (If Completed)

1. CI-gated interpretability workflow.
- Novelty: strict automated claim gating that binds narrative to external/stress thresholds.

2. Tri-objective SAE selection under uncertainty.
- Novelty: grouped-LCB selection over internal consistency + SAEBench + CE-Bench.

3. External-aware consistency training.
- Novelty: assignment-aware training coupled to external acceptance, not post-hoc filtering only.

4. Matched-budget architecture frontier with routed/matryoshka families.
- Novelty: apples-to-apples external comparisons under fixed token budget and hook protocol.

5. Proposal-closure discipline for known-circuit recovery.
- Novelty: explicit closure tests for mechanistic claims instead of benchmark-only narratives.

## Evidence-Safe Literature Links (Verified)

- SAEs trained on same data learn different features: https://arxiv.org/abs/2501.16615
- Feature consistency priority: https://arxiv.org/abs/2505.20254
- SAEBench: https://arxiv.org/abs/2503.09532
- CE-Bench: https://arxiv.org/abs/2509.00691
- Route Sparse Autoencoders: https://arxiv.org/abs/2503.08200
- Nested Sparse Autoencoders (Matryoshka): https://arxiv.org/abs/2503.17547
- PolySAE: https://arxiv.org/abs/2602.01322
- Transcoders vs SAEs: https://arxiv.org/abs/2501.18823
- Random-control caution: https://arxiv.org/abs/2501.17727

## Guardrails for Future Claims

- No external-improvement claim unless matched-baseline deltas and CI bounds are non-negative.
- No reliability claim unless strict stress gate passes (`random + transcoder + OOD + external`).
- No proposal-complete claim until known-circuit closure track is green.

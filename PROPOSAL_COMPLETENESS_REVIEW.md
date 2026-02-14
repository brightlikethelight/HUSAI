# Proposal Completeness Review (2026-02-14)

## Executive Verdict
The project is no longer in an exploratory state; it is in an evidence-backed state with reproducible external harnesses, but it is **not yet scientifically complete** relative to the original proposal.

What is complete:
- End-to-end reliability and benchmark plumbing (SAEBench + CE-Bench paths, artifact manifests, CI smoke/tests).
- Multiple internal consistency studies and controlled ablations.
- External benchmark execution proving a real performance gap.

What is incomplete:
- The original “Goldilocks zone” hypothesis is only partially tested externally.
- Ground-truth circuit recovery (e.g., Tracr-style known circuits) is not yet closed.
- Multi-seed external uncertainty for frontier/scaling is still sparse or in-progress.
- Stress-gated release remains blocked by missing transcoder/OOD artifacts and negative external deltas.

## Original Proposal vs Current Status

| Proposal area | Intended outcome | Current status | Evidence |
|---|---|---|---|
| Phase 1 controlled sweeps (50+ SAEs, seeds/architecture/sparsity/width) | Broad controlled map with trajectories | **Partial** | `results/experiments/phase4c_core_ablations/`, `results/experiments/adaptive_l0_calibration/`, `results/experiments/phase4b_architecture_frontier_external/` |
| Phase 2 deep analysis (matching + known circuit recovery + geometry) | Validate recovered structure against known ground truth | **Partial** | Matching/consistency analyses present; direct Tracr-circuit recovery remains open |
| Phase 3 better tools + reproducibility | Reproducible OSS-quality framework | **Strong** | `RUNBOOK.md`, CI workflows, `EXPERIMENT_LOG.md`, `scripts/analysis/verify_experiment_consistency.py` |
| External benchmark alignment | SAEBench + CE-Bench-grounded claims | **Strong infrastructure, weak performance** | `docs/evidence/phase4e_cebench_matched200/`, `docs/evidence/phase4e_external_scaling_study/`, `docs/evidence/phase4b_architecture_frontier_external/` |
| Release rigor | Gate claims through stress + controls | **Implemented, currently failing** | `docs/evidence/phase4e_stress_gated_release/run_20260213T204120Z_release_policy.json` |

## What We Actually Learned

1. Internal consistency can be improved materially.
- Assignment-aware v2 achieved positive internal gains with low EV drop.
- Evidence: `docs/evidence/phase4d_assignment_consistency_v2/run_20260213T203957Z_results.json`.

2. External metrics remain the bottleneck.
- SAEBench best-minus-LLM AUC remains negative in tested conditions.
- CE-Bench matched-baseline deltas remain strongly negative for current custom checkpoints.
- Evidence: `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_cebench_deltas_vs_matched200.md`, `docs/evidence/high_impact_adapter_check/run_20260214T202232Z_husai_custom_cebench_summary.json`.

3. Benchmark tradeoffs are architecture- and layer-sensitive.
- Frontier and scaling indicate non-aligned optima between SAEBench and CE-Bench.
- Evidence: `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_summary_table.md`, `docs/evidence/phase4e_external_scaling_study/run_20260213T203923Z_summary_table.md`.

## Critical Risk Register (Brutally Honest)

## P0
1. Claim-risk mismatch in legacy consistency audit.
- The old audit reported `overall_pass=True` while newer external/stress gates fail.
- Fix applied: `scripts/analysis/verify_experiment_consistency.py` now incorporates assignment-v2 + stress gates.

2. External benchmark underperformance vs matched baseline.
- Current custom checkpoints are far below matched CE-Bench baseline.
- This blocks any strong external interpretability claim.

## P1
3. Proposal coverage gap on known-circuit recovery.
- Core promise included known ground-truth recovery; this remains not fully closed.

4. Limited external CIs for final decision-making.
- Many external tables are still seed-limited; uncertainty remains underpowered.

5. Stress-gate incompleteness in practice.
- Gate code exists, but required transcoder/OOD artifact generation is not yet integrated as routine closure.

## P2
6. Narrative drift risk across long-form docs.
- Some docs can still read as stronger than artifact-backed external performance.

## Is Anything Potentially Erroring?

Likely issue fixed this cycle:
- Consistency audit now reflects modern gate failures instead of legacy-only pass criteria.

Still worth rechecking:
1. CE-Bench score comparability assumptions across model/SAE families.
- Ensure matched-baseline comparisons always include equal `max_rows`, model, hook, and eval settings.

2. External summary object selection for gates.
- Ensure release gates consume the best candidate summary from the latest multiseed run, not stale single-seed files.

3. Documentation language around “validated findings”.
- Keep external claims explicitly scoped to current negative deltas.

## Internet-Grounded Landscape: Where We Can Be Novel Next

Primary sources point to a credible novelty path:
- SAEBench emphasizes broad evals and external validity pressure (ICML 2025): https://proceedings.mlr.press/v267/karvonen25a.html
- CE-Bench provides contrastive external scoring and claims improved discriminative power: https://arxiv.org/abs/2509.00691
- Control pressure from “Transcoders Beat SAEs” and random-model critiques raises bar for robust claims: https://arxiv.org/abs/2501.18823, https://arxiv.org/abs/2501.17727
- Architecture frontier is expanding (Matryoshka, RouteSAE, HierarchicalTopK, PolySAE):
  - https://arxiv.org/abs/2503.17547
  - https://aclanthology.org/2025.emnlp-main.346/
  - https://aclanthology.org/2025.emnlp-main.515/
  - https://arxiv.org/abs/2602.01322

High-probability novel contribution for HUSAI:
- **A multi-objective external frontier paper**: explicitly optimize and report the Pareto between internal seed consistency and external benchmark deltas under strict stress gates.
- This is more novel and publishable than one more internal-only consistency variant.

## Highest-Leverage Next Experiments (B200-Ready)

1. Finish multiseed external architecture frontier and compute CIs.
- Status: in progress (`run_20260214T202538Z`).
- Acceptance: all 4 architectures x 5 seeds complete with SAEBench + CE-Bench deltas and variance.

2. Multiseed external scaling (token budget/hook layer/d_sae) under matched CE-Bench baseline.
- Suggested seeds: `42,123,456`.
- Acceptance: axis-level means + 95% CIs + best-condition selection policy.

3. Stress closure run: transcoder + OOD + strict release gate.
- Generate fresh artifacts with current best checkpoint.
- Acceptance: `run_stress_gated_release_policy.py --fail-on-gate-fail` as canonical release test.

4. Ground-truth recovery track (proposal closure item).
- Add Tracr/known-circuit benchmarks and evaluate circuit recovery stability across seeds.
- Acceptance: recover known circuit-aligned features above random control with CIs.

5. Architecture expansion under matched budget.
- Add at least one of Matryoshka/RouteSAE/HierarchicalTopK in the same external protocol.
- Acceptance: beat current best external delta or improve Pareto frontier without violating gates.

## Final Assessment
We are close to a strong, honest finish, but not done. The repo is now technically mature enough for high-impact experiments; the remaining work is scientific: close the external gap, complete stress controls, and finish proposal-promised ground-truth recovery.

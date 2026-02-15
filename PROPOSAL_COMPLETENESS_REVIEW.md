# Proposal Completeness Review

Date: 2026-02-15

## Executive Verdict

The project is now engineering-complete and evidence-backed for the major reliability and benchmark goals. It is not yet scientifically complete relative to the original ambition of finding a clearly transferable "Goldilocks zone" with strong external gains.

## Original Proposal vs Current Status

| Proposal area | Intended outcome | Current status | Evidence |
|---|---|---|---|
| Phase 1 controlled sweeps (seeds/architecture/sparsity/width) | Broad controlled map | strong | `results/experiments/phase4c_core_ablations/`, `results/experiments/phase4b_architecture_frontier_external_multiseed/`, `results/experiments/phase4e_external_scaling_study_multiseed/` |
| Phase 2 deep analysis (matching, geometry, known-circuit recovery) | Recover and validate structure against known ground truth | partial | Matching and consistency analyses present; known-circuit recovery remains open |
| Phase 3 reproducibility tooling | Clean, repeatable OSS-grade stack | strong | `RUNBOOK.md`, `.github/workflows/ci.yml`, `EXPERIMENT_LOG.md`, `scripts/analysis/verify_experiment_consistency.py` |
| External benchmark alignment (SAEBench + CE-Bench) | Externally grounded claims | strong infrastructure, weak performance | `docs/evidence/cycle3_queue_final/frontier_multiseed_results_run_20260214T202538Z.json`, `docs/evidence/cycle3_queue_final/scaling_multiseed_results_run_20260214T212435Z.json` |
| Stress-gated release rigor | Promote claims only if controls pass | implemented and failing correctly | `docs/evidence/cycle3_queue_final/release_policy_run_20260214T225029Z.json` |

## What We Learned

1. Internal consistency can be improved robustly.
- Assignment-aware v2 improves internal metrics with low EV loss.

2. External transfer remains unresolved.
- SAEBench deltas remain negative and CE-Bench deltas vs matched baseline remain strongly negative.

3. Tradeoffs are systematic, not accidental.
- Architecture and scaling move SAEBench and CE-Bench in different directions.

4. Claim hygiene is now strong.
- Strict gates prevent unsupported external claims.

## Key Open Gaps

1. Known-ground-truth circuit recovery closure from original proposal.
2. External-positive candidate that satisfies strict release gates.
3. Multi-objective method that improves both internal and external metrics together.

## Potential Error/Claim Risk Checks

1. Ensure gate external summary always points to the intended candidate and run.
- Current gate input in cycle-3 used `relu_seed42` SAEBench summary; selection policy should be explicit.

2. Keep matched-baseline comparability strict.
- Same model/hook/max_rows/protocol for every CE-Bench comparison.

3. Keep narrative tied to gate truth.
- Any claim of external progress should require fresh gate evidence with `pass_all=true`.

## Novel Contribution Opportunity (Internet-Grounded)

Most credible path to novelty now:
- A multi-objective external frontier paper that explicitly models and reports the Pareto tradeoff between internal consistency and external benchmark performance under strict stress gates.

Why this is credible:
- SAEBench and CE-Bench pressure external validity and expose conflicting preferences.
- Recent literature keeps raising control standards (random-model controls, transcoder comparisons).
- HUSAI already has the infrastructure to run this rigorously.

## Final Assessment

The project is now in a publishable methodological state for an honest negative/nuanced result: internal consistency progress with unresolved external transfer. To fully satisfy the original proposal, prioritize known-circuit recovery and a gate-passing external candidate.

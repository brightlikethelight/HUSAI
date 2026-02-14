# Novel Contributions and Highest-Leverage Follow-Ups

Updated: 2026-02-13

## Reality Check (Post-Cycle 2)

What is now done end-to-end:
- Direct HUSAI CE-Bench adapter path exists and runs.
- Matched-budget architecture frontier (TopK/ReLU/BatchTopK/JumpReLU) is complete on external metrics.
- External scaling study (`token budget`, `hook layer`, `d_sae`) is complete.
- Assignment-aware consistency objective v2 is complete with explicit acceptance gates.
- Stress-gated release policy is complete and now supports fail-fast CI behavior.

What the evidence says:
- External delta remains negative across tested settings.
- Internal consistency can be improved (assignment-v2), but external acceptance still fails.
- Architecture choice creates cross-benchmark tradeoffs, not a clear global winner.

## Evidence Pointers

- CE-Bench matched baseline summary:
  - `docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json`
- Architecture frontier:
  - `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_results.json`
  - `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_summary_table.md`
- Scaling study:
  - `docs/evidence/phase4e_external_scaling_study/run_20260213T203923Z_results.json`
  - `docs/evidence/phase4e_external_scaling_study/run_20260213T203923Z_summary_table.md`
- Assignment v2:
  - `docs/evidence/phase4d_assignment_consistency_v2/run_20260213T203957Z_results.json`
- Stress policy:
  - `docs/evidence/phase4e_stress_gated_release/run_20260213T204120Z_release_policy.json`

## Updated Highest-Leverage Next 5 (Ranked)

1. Add transcoder + OOD stress tracks and enforce strict release gating in CI.
- Why now:
  - stress policy currently fails because transcoder/OOD evidence is missing.
- Deliverable:
  - `transcoder_results.json`, `ood_results.json`, and CI step that runs `run_stress_gated_release_policy.py --fail-on-gate-fail`.

2. Expand external architecture frontier beyond 4 families (Matryoshka/RouteSAE/HierarchicalTopK).
- Why now:
  - current frontier exposes tradeoffs but no positive external-delta region.
- Deliverable:
  - matched-budget run with identical protocol and uncertainty report.

3. Run multi-seed external CIs for frontier + scaling winners.
- Why now:
  - most external sweeps are single-seed; claim confidence is underpowered.
- Deliverable:
  - >=3 seeds for selected configs, with CI tables for SAEBench/CE-Bench deltas.

4. Introduce Pareto checkpoint selection (consistency + external metrics jointly).
- Why now:
  - internal-only optimization (assignment-v2) does not satisfy external gate.
- Deliverable:
  - model selection script that ranks checkpoints by Pareto dominance and acceptance constraints.

5. Test layer-aware architecture routing under fixed parameter budget.
- Why now:
  - scaling indicates layer-specific metric preferences (layer1 helps CE-Bench, hurts SAEBench delta).
- Deliverable:
  - composite model (`layer0` family + `layer1` family) benchmarked against single-family baselines.

## Rare but High-Value Novel Ideas

1. Benchmark-aware curriculum for SAE training
- Increase CE-Bench- and SAEBench-aligned slices over training stages to reduce cross-metric collapse.

2. Assignment-aware external regularizer
- Extend v2 matching with external proxy constraints (e.g., probe separability priors) during training.

3. Hook-adaptive sparsity schedules
- Learn per-layer `k`/`d_sae` schedules from activation geometry rather than fixed global settings.

4. External-metric early stopping
- Stop on external validation deltas, not reconstruction-only metrics.

5. Claims ledger automation
- CI-generated claim table from JSON artifacts; block unsupported narrative updates by default.

## Primary Sources

- SAEBench (ICML 2025): https://proceedings.mlr.press/v267/karvonen25a.html
- CE-Bench (2025): https://arxiv.org/abs/2509.00691
- JumpReLU: https://arxiv.org/abs/2407.14435
- BatchTopK: https://arxiv.org/abs/2412.06410
- Matryoshka SAEs: https://arxiv.org/abs/2503.17547
- RouteSAE: https://aclanthology.org/2025.emnlp-main.346/
- HierarchicalTopK: https://aclanthology.org/2025.emnlp-main.515/
- Transcoders Beat SAEs: https://arxiv.org/abs/2501.18823
- Random-transformer control framing: https://arxiv.org/abs/2501.17727

## 2026-02-14 Literature Refresh and Novelty Upgrade

Additional primary-source signals:
- SAEBench 2026 release notes emphasize broader benchmark coverage and architectural diversity in reported results:
  - https://github.com/adamkarvonen/SAEBench
- MIB benchmark (2025) extends evaluation toward richer mechanistic interpretability behaviors:
  - https://arxiv.org/abs/2504.13151
- PolySAE (2026) reports stronger reconstruction/sparsity/interpretability tradeoffs at scale:
  - https://arxiv.org/abs/2602.01322
- Taming polysemanticity via SAE recovery theory (2025):
  - https://arxiv.org/abs/2506.14002

Concrete novelty opportunities for HUSAI (high confidence):
1. Multi-objective frontier paper:
- define and optimize the Pareto surface over `internal consistency` vs `SAEBench delta` vs `CE-Bench delta`.
- novelty: explicit tri-objective externalization, not single-metric optimization.

2. External-gated consistency training:
- combine assignment-aware objective with external proxy constraints and strict release gates.
- novelty: consistency training that is benchmark-aware by construction.

3. Architecture-by-layer policy:
- choose architecture family per hook layer under a fixed parameter budget.
- novelty: layer-specialized SAE architecture policy learned from external metric profiles.

4. Circuit-grounded closure track:
- add Tracr/known-circuit recovery quality as a co-equal axis with SAEBench/CE-Bench.
- novelty: links stability claims to identifiable ground truth, closing original proposal scope.

5. Automatic claim ledger:
- machine-generated claim table from artifacts with CI gate enforcement.
- novelty: prevents publication drift between narrative and measured evidence.

# Novel Contributions and Highest-Leverage Follow-Ups

Updated: 2026-02-13

## Current Reality Check

Completed this cycle:
- CI fail-fast smoke + quality gates are active.
- Absolute-path fragility in core experiment scripts is removed.
- Phase 4a/4c/4e runs were re-executed on RunPod B200 with manifests.
- Official SAEBench command execution succeeded:
  - `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/`
- HUSAI custom SAEBench checkpoint path is now fully executed across 3 seeds:
  - `run_20260213T024329Z`, `run_20260213T031247Z`, `run_20260213T032116Z`
  - aggregate evidence: `docs/evidence/phase4e_husai_custom_multiseed/summary.json`

Current blockers:
- CE-Bench is still not executed in this environment.
- HUSAI custom SAEBench AUC remains below baseline logreg probes (mean delta `-0.0518`).
- Objective-level consistency regularization v1 remains statistically unresolved.

## Literature-Grounded Opportunity Map

Anchors:
- Consistency framing: https://arxiv.org/abs/2501.16615, https://arxiv.org/abs/2505.20254
- Benchmark standards: https://proceedings.mlr.press/v267/karvonen25a.html, https://arxiv.org/abs/2509.00691
- Architecture frontier: https://arxiv.org/abs/2407.14435, https://arxiv.org/abs/2412.06410, https://arxiv.org/abs/2503.17547, https://aclanthology.org/2025.emnlp-main.346/, https://aclanthology.org/2025.emnlp-main.515/
- Competing views/controls: https://arxiv.org/abs/2501.18823, https://arxiv.org/abs/2501.17727, https://arxiv.org/abs/2602.01322

## Ranked Next 5 Highest-Leverage Follow-Ups (Now)

1. CE-Bench execution for HUSAI + baseline targets
- Why #1:
  - External claim quality is currently bottlenecked by missing CE-Bench evidence.
- Concrete work:
  - Add reproducible CE-Bench run path in `scripts/experiments/run_official_external_benchmarks.py` with explicit env notes and manifests.
- Success criteria:
  - Completed CE-Bench runs with `commands.json`, `summary.md`, logs, and per-seed table.

2. Matched-budget architecture frontier sweep on external stack
- Why #2:
  - Current HUSAI external gap is systematic; architecture choice likely higher leverage than small objective tweaks.
- Methods:
  - TopK, JumpReLU, BatchTopK, Matryoshka, RouteSAE, HierarchicalTopK.
- Success criteria:
  - At least one variant improves external AUC with CI excluding zero versus current HUSAI baseline.

3. External-metric-focused scaling study (data/layer/width)
- Why #3:
  - Current runs are stable across seeds but uniformly under baseline, indicating capacity/data mismatch rather than seed noise.
- Concrete work:
  - Sweep activation token budget, hook layer, and `d_sae` under fixed compute envelopes.
- Success criteria:
  - Demonstrable movement in SAEBench/CE-Bench metrics with full uncertainty reporting.

4. Consistency-objective v2 (assignment-aware)
- Why #4:
  - v1 consistency regularizer effect is near zero under CI.
- Concrete variants:
  - Hungarian/OT-based cross-seed alignment losses,
  - joint multi-seed anchor training.
- Success criteria:
  - Positive consistency gain with CI excluding zero and EV drop <= 5%, plus no external regression.

5. Stress-gated release policy (transcoder + random-model + OOD)
- Why #5:
  - Prevents optimistic narrative drift and forces robust comparisons.
- Concrete work:
  - Add matched transcoder baseline,
  - random-transformer control,
  - OOD sensitivity checks.
- Success criteria:
  - Claim updates blocked in CI when stress suite fails.

## Rare but High-Value Novel Contributions We Can Target

1. Geometry-conditioned architecture selector
- Learn policy mapping activation geometry to best SAE family/sparsity.

2. Dual-objective model selection
- Optimize external benchmark score and consistency delta jointly rather than sequentially.

3. Polysemantic-aware consistency training
- Adapt PolySAE ideas to improve stability while preserving external utility.

4. Cross-task dictionary transfer under consistency constraints
- Train on one task family, evaluate transfer and alignment on another with controlled shift.

5. Claims ledger automation
- Auto-generate claim tables from artifact JSONs and fail CI on unsupported statements.

## Immediate Execution Sequence

1. Land CE-Bench reproducible run path and execute baseline + HUSAI targets.
2. Launch matched-budget architecture frontier sweep.
3. Run external-metric scaling study (`tokens/layer/d_sae`).
4. Implement assignment-aware consistency objective v2.
5. Enforce stress-gated release checks for narrative updates.

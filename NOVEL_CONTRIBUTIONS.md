# Novel Contributions and Highest-Leverage Follow-Ups

Updated: 2026-02-12

## Current Reality Check

Completed this cycle:
- CI fail-fast smoke + incremental quality gates are in place.
- Absolute path fragility in core experiment scripts is removed.
- Phase 4a/4c/4e internal runs were re-executed on RunPod B200.
- Official SAEBench command execution succeeded through the harness:
  - `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/`
- Claim-consistency audit is script-enforced:
  - `scripts/analysis/verify_experiment_consistency.py`

What remains blocked:
- CE-Bench is still not executed in this environment.
- Official run used public SAEBench SAE target; HUSAI-to-SAEBench adapter path is not yet end-to-end benchmarked.
- Current objective-level consistency regularization remains statistically unresolved.

## Literature-Grounded Opportunity Map

Anchors:
- Consistency problem and framing: https://arxiv.org/abs/2501.16615, https://arxiv.org/abs/2505.20254
- External benchmark standards: https://proceedings.mlr.press/v267/karvonen25a.html, https://arxiv.org/abs/2509.00691
- Architecture frontier: https://arxiv.org/abs/2407.14435, https://arxiv.org/abs/2412.06410, https://arxiv.org/abs/2503.17547, https://aclanthology.org/2025.emnlp-main.346/, https://aclanthology.org/2025.emnlp-main.515/
- Frontier controls/alternatives: https://arxiv.org/abs/2501.18823, https://arxiv.org/abs/2501.17727

## Ranked Next 5 Highest-Leverage Follow-Ups (Now)

1. Benchmark HUSAI checkpoints directly in official harnesses (SAEBench then CE-Bench)
- Why #1:
  - Current official run proves tooling works but does not yet benchmark HUSAI-produced dictionaries end-to-end.
- Concrete work:
  - Build adapter/export path from `results/saes/*` into SAEBench custom SAE format.
  - Add CE-Bench execution path in `run_official_external_benchmarks.py` with pinned env + manifests.
- Success criteria:
  - Benchmark artifacts for HUSAI checkpoints with reproducible command manifests.
  - No SOTA claims unless both benchmark results and uncertainty are reported.

2. Matched-budget architecture frontier sweep
- Why #2:
  - Literature suggests architecture choice often dominates small objective tweaks.
- Candidate methods:
  - TopK (current), JumpReLU, BatchTopK, Matryoshka, HierarchicalTopK, RouteSAE.
- Protocol:
  - Same activations, same seed budget, same optimizer budget, same eval stack.
- Success criteria:
  - Pareto improvement over current TopK baseline on at least 2 of 3 axes:
    - consistency delta,
    - reconstruction quality,
    - external benchmark score.

3. Consistency-objective v2: assignment-aware or joint multi-seed training
- Why #3:
  - v1 regularizer gain is tiny and CI crosses zero.
- Concrete variants:
  - permutation-aware alignment loss (Hungarian or OT-style matching),
  - joint multi-seed training with shared feature anchors,
  - cross-model agreement losses inspired by feature-circuit work.
- Success criteria:
  - Positive trained-PWMCC gain with 95% CI excluding zero and EV drop <= 5%.

4. Transcoder control arm under identical budgets
- Why #4:
  - Recent evidence indicates transcoders can outperform SAEs in some interpretability regimes.
- Concrete work:
  - Add a matched-parameter transcoder baseline into the same training/eval harness.
  - Compare consistency + functional metrics + intervention behavior.
- Success criteria:
  - Explicit regime map where SAE wins and where transcoder wins.

5. Random-initialization and stress-test gate before claims
- Why #5:
  - Recent work shows random-model representations can be stronger than expected; this is a high-value falsification control.
- Concrete work:
  - Add random-transformer control suite, OOD transfer checks, perturbation/noise robustness, and data-fraction scaling.
  - Gate writeup updates on stress-suite pass + consistency audit pass.
- Success criteria:
  - Reproducible stress-test table with confidence intervals and explicit failure modes.

## Rare but High-Value Novel Contributions We Can Target

1. Stability-conditioned architecture selection
- Learn a policy that selects architecture and sparsity (k, d_sae) based on activation geometry (effective rank, anisotropy).

2. Dual-objective model selection
- Treat external benchmark score and consistency delta as co-primary optimization objectives, not sequential filters.

3. Cross-task dictionary transfer with consistency constraints
- Train on one algorithmic task, evaluate feature reuse and alignment on a second task with controlled covariate shift.

4. Benchmark-aware curriculum for sparsity
- Start with higher-k for reconstruction stability, anneal to lower-k when consistency saturates.

5. Claim ledger automation
- Auto-generate a machine-readable "claims ledger" from result JSONs and benchmark files; CI fails on unsupported narrative claims.

## Immediate Execution Sequence

1. Implement HUSAI SAE export/adapter and execute official SAEBench on HUSAI checkpoints.
2. Execute CE-Bench from the same harness with manifest logging.
3. Launch matched-budget architecture frontier sweep.
4. Implement consistency-objective v2 and compare against v1 and baseline.
5. Add random-model + OOD stress gates and enforce them in release flow.

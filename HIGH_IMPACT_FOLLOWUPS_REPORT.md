# High-Impact Follow-Ups: Execution Report

Date: 2026-02-12

## Scope

This report consolidates the highest-impact follow-ups executed in this cycle:
1. CI + smoke workflow hardening.
2. Absolute-path portability fixes in experiment/analysis runners.
3. Phase 4a trained-vs-random reproduction with manifest logging.
4. Phase 4c core ablations (k sweep and d_sae sweep) with uncertainty-aware summaries.
5. External benchmark alignment, including official SAEBench harness execution.

## Executive Findings

- Engineering reliability materially improved (CI, smoke, path normalization, dependency fix).
- Internal consistency findings remain directionally positive but modest in absolute size on remote rerun.
- Adaptive low-L0 findings remain the strongest internal positive signal.
- Official SAEBench command execution completed successfully, but external results in this setup are not competitive against baseline probes.
- CE-Bench is still not executed in this environment.

## Follow-up 1: CI + Smoke Workflow

Evidence:
- `.github/workflows/ci.yml`
- `scripts/ci/smoke_pipeline.sh`

Status:
- Added fail-fast smoke job before quality checks.
- Added incremental lint/typecheck gate plus full pytest in CI.

Result:
- Local/remote smoke paths execute successfully.
- Full test suite remains green (`83 passed`).

## Follow-up 2: Remove Absolute Path Fragility

Evidence:
- `scripts/experiments/run_phase4a_reproduction.py`
- `scripts/experiments/run_core_ablations.py`
- `scripts/experiments/run_external_benchmark_slice.py`

Status:
- Relative and workspace-rooted paths are normalized in all three scripts.
- Artifact references now store repo-relative paths where possible.

Result:
- Remote RunPod execution no longer depends on local absolute directories.

## Follow-up 3: Phase 4a Full Reproduction (Remote B200)

Artifact:
- `results/experiments/phase4a_trained_vs_random/results.json`

Readout (5 trained seeds vs 5 random controls):
- trained mean PWMCC: `0.300059`
- random mean PWMCC: `0.298829`
- delta: `+0.001230`
- ratio: `1.0041`
- one-sided Mann-Whitney p-value: `8.629e-03`
- conclusion: `training_signal_present`

Interpretation:
- Statistical signal exists, but effect size is small in this rerun.

## Follow-up 4: Core Ablations with CIs (Remote B200)

Artifact:
- `results/experiments/phase4c_core_ablations/run_20260212T200711Z/results.json`

Best k-sweep condition:
- `k=8`, `d_sae=128`
- delta PWMCC: `+0.009773`
- ratio: `1.0398`
- EV mean: `0.2282`

Best d_sae-sweep condition:
- `d_sae=64`, `k=32`
- delta PWMCC: `+0.119986`
- ratio: `1.5272`
- EV mean: `0.2941`

Interpretation:
- Geometry (`d_sae`) can dominate effect magnitude.
- Stability-reconstruction tradeoffs remain substantial across regimes.

## Follow-up 5: External Benchmark Slice + Official SAEBench

### 5a) Internal benchmark-aligned slice

Artifact:
- `results/experiments/phase4e_external_benchmark_slice/benchmark_slice.json`

Readout:
- internal gating pass: `True`
- ready for external benchmark claim: `False`

### 5b) Official SAEBench harness execution (completed)

Harness run artifact:
- `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/`

Command status:
- `commands.json`: SAEBench attempted `True`, success `True`, return code `0`

Environment preflight:
- SAEBench module available: `True`
- CE-Bench module available: `False`

Produced official SAE-probes outputs:
- `/tmp/sae_bench_probe_results/`
- matched datasets: `113`

Aggregate SAE-vs-baseline (best SAE over k in {1,2,5} minus baseline logreg):
- test_f1 mean delta: `-0.0952` (wins 12, losses 81, ties 20)
- test_acc mean delta: `-0.0513` (wins 18, losses 72, ties 23)
- test_auc mean delta: `-0.0651` (wins 21, losses 88, ties 4)

Additional diagnostic:
- best-k by dataset: `{k=5: 74, k=2: 14, k=1: 25}`
- baseline mean test_auc: `0.6744`

Interpretation:
- Official benchmark execution is now operational and reproducible.
- External results here do not support SOTA claims.
- CE-Bench remains pending.

## Reliability and Correctness Issues Fixed During Remote Rerun

1. `src/data/` tracking bug from over-broad `.gitignore` rule.
2. NumPy pinning incompatible with TransformerLens dependencies.
3. Relative path failures in core experiment scripts.
4. Official benchmark harness improved to stream subprocess logs to disk (no full-output in-memory buffering).

## Ranked Next 5 Highest-Leverage Follow-Ups

1. Benchmark HUSAI-produced checkpoints directly in official SAEBench + CE-Bench.
2. Run matched-budget architecture frontier sweep (TopK, JumpReLU, BatchTopK, Matryoshka, HierarchicalTopK, RouteSAE).
3. Implement consistency-objective v2 (assignment-aware/joint multi-seed) with strict CI-based acceptance.
4. Add transcoder control arm under identical budgets.
5. Add random-model and OOD stress-test gates before any claim update.

## Primary-Source References

- Paulo & Belrose (2025): https://arxiv.org/abs/2501.16615
- Song et al. (2025): https://arxiv.org/abs/2505.20254
- SAEBench (ICML 2025): https://proceedings.mlr.press/v267/karvonen25a.html
- CE-Bench (2025): https://arxiv.org/abs/2509.00691
- OpenAI SAE scaling/eval: https://arxiv.org/abs/2406.04093
- JumpReLU: https://arxiv.org/abs/2407.14435
- BatchTopK: https://arxiv.org/abs/2412.06410
- Matryoshka SAEs: https://arxiv.org/abs/2503.17547
- Route Sparse Autoencoders: https://aclanthology.org/2025.emnlp-main.346/
- HierarchicalTopK SAEs: https://aclanthology.org/2025.emnlp-main.515/
- Randomly initialized transformer representations: https://arxiv.org/abs/2501.18823
- Transcoders vs SAEs: https://arxiv.org/abs/2501.17727

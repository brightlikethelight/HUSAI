# High-Impact Follow-Ups: Execution Report

Date: 2026-02-13

## Scope

This report consolidates the highest-impact follow-ups executed in this cycle:
1. CI + smoke workflow hardening.
2. Absolute-path portability fixes in `scripts/experiments/*` and `scripts/analysis/*`.
3. Phase 4a trained-vs-random reproduction with manifest logging.
4. Phase 4c core ablations (`k` sweep and `d_sae` sweep) with uncertainty-aware summaries.
5. External benchmark alignment, including official SAEBench runs, direct HUSAI custom SAEBench runs, and official CE-Bench compatibility execution.

## Executive Findings

- Engineering reliability improved materially (CI, smoke, path normalization, dependency fixes, benchmark log streaming, CE-Bench compatibility shims).
- Internal consistency findings remain directionally positive but regime-sensitive and often modest in absolute size.
- Adaptive low-L0 remains the strongest internal positive signal.
- External benchmark evidence is now execution-backed, not preflight-only:
  - official SAEBench command run completed;
  - direct HUSAI custom-checkpoint SAEBench path completed with 3 seeds;
  - official CE-Bench compatibility run completed end-to-end.
- Current HUSAI checkpoint family remains below logreg LLM-baseline on SAEBench SAE-probes (AUC), with low seed variance.
- CE-Bench is now operational for official/public SAE targets in this environment; direct HUSAI-checkpoint CE-Bench support is still the highest-priority remaining benchmark gap.

## Follow-up 1: CI + Smoke Workflow

Evidence:
- `.github/workflows/ci.yml`
- `scripts/ci/smoke_pipeline.sh`

Status:
- Fail-fast smoke job runs before quality checks.
- Incremental lint/typecheck gate plus full `pytest` gate in CI.

Result:
- Local and remote smoke paths execute successfully.
- Full test suite remains green (`83 passed`).

## Follow-up 2: Remove Absolute Path Fragility

Evidence:
- `scripts/experiments/run_phase4a_reproduction.py`
- `scripts/experiments/run_core_ablations.py`
- `scripts/experiments/run_external_benchmark_slice.py`
- `scripts/experiments/multi_architecture_stability.py`
- `scripts/experiments/comprehensive_stability_analysis.py`

Status:
- Core runner paths are repo-relative/workspace-rooted.
- Machine-specific usage examples (`~/miniconda3/...`) removed from experiment docstrings.

Result:
- Remote execution no longer depends on local absolute directories for the primary follow-up path.

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

Best `k`-sweep condition:
- `k=8`, `d_sae=128`
- delta PWMCC: `+0.009773`
- ratio: `1.0398`
- EV mean: `0.2282`

Best `d_sae`-sweep condition:
- `d_sae=64`, `k=32`
- delta PWMCC: `+0.119986`
- ratio: `1.5272`
- EV mean: `0.2941`

Interpretation:
- Geometry (`d_sae`) can dominate effect magnitude.
- Stability-reconstruction tradeoffs remain substantial across regimes.

## Follow-up 5: External Benchmark Alignment and Official Runs

### 5a) Internal benchmark-aligned slice

Artifact:
- `results/experiments/phase4e_external_benchmark_slice/benchmark_slice.json`

Readout:
- internal gating pass: `True`
- ready for external benchmark claim: `False`

### 5b) Official SAEBench harness execution (public SAE target)

Harness run artifact:
- `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/`

Command status:
- `commands.json`: SAEBench attempted `True`, success `True`, return code `0`

Aggregate SAE-vs-baseline (best SAE over `k in {1,2,5}` minus baseline logreg):
- `test_f1` mean delta: `-0.0952`
- `test_acc` mean delta: `-0.0513`
- `test_auc` mean delta: `-0.0651`
- `test_auc` wins/losses/ties: `21 / 88 / 4`

### 5c) Official HUSAI custom-checkpoint SAEBench path (multi-seed)

HUSAI custom run artifacts:
- seed 42: `results/experiments/phase4e_external_benchmark_official/run_20260213T024329Z/`
- seed 123: `results/experiments/phase4e_external_benchmark_official/run_20260213T031247Z/`
- seed 456: `results/experiments/phase4e_external_benchmark_official/run_20260213T032116Z/`

Tracked evidence copies:
- `docs/evidence/phase4e_husai_custom_multiseed/summary.json`
- `docs/evidence/phase4e_husai_custom_multiseed/summary.md`

Per-seed best AUC (all at `k=5`):
- seed 42: `0.623311`
- seed 123: `0.622244`
- seed 456: `0.622249`

Multi-seed aggregate:
- best AUC mean ± std: `0.622601 ± 0.000615`
- best AUC 95% CI: `[0.621905, 0.623297]`
- delta AUC vs LLM baseline mean ± std: `-0.051801 ± 0.000615`
- delta AUC vs LLM baseline 95% CI: `[-0.052496, -0.051105]`

Interpretation:
- HUSAI custom benchmark path is operational and reproducible.
- Results are consistently below the LLM baseline and do not support SOTA claims.

### 5d) Official CE-Bench compatibility run (public SAE target)

Primary run artifact:
- `results/experiments/phase4e_external_benchmark_official/run_20260213T103218Z/`

Tracked evidence copies:
- `docs/evidence/phase4e_cebench_official/run_20260213T103218Z_harness_summary.md`
- `docs/evidence/phase4e_cebench_official/run_20260213T103218Z_commands.json`
- `docs/evidence/phase4e_cebench_official/run_20260213T103218Z_cebench_results.json`
- `docs/evidence/phase4e_cebench_official/run_20260213T103218Z_cebench_metrics_summary.json`

Command status:
- `commands.json`: CE-Bench attempted `True`, success `True`, return code `0`

CE-Bench metric snapshot (`total_rows=5000`):
- `contrastive_score_mean.max`: `49.1142`
- `independent_score_mean.max`: `53.6982`
- `interpretability_score_mean.max`: `47.4812`
- SAE target: `pythia-70m-deduped-res-sm / blocks.0.hook_resid_pre`

Interpretation:
- CE-Bench execution path is now operational in this environment.
- CE-Bench metric scale differs from SAEBench AUC/F1 and should not be directly compared numerically.
- Legacy CE-Bench relative-output behavior (`scores_dump.txt` append) was identified and patched for deterministic run-local outputs.

## Reliability and Correctness Issues Fixed During This Program

1. `.gitignore` tracking bug from over-broad `src/data/` exclusion.
2. NumPy/TransformerLens compatibility pinning issue.
3. Relative path failures in core experiment scripts.
4. Official benchmark harness updated to stream subprocess logs to disk.
5. HUSAI custom SAEBench dataset auto-inference bug fixed (empty dataset list issue).
6. CE-Bench compatibility drift fixed (`sae_lens.toolkit` aliasing, multiprocessing shim, `stw.Stopwatch` API mismatch).
7. CE-Bench runner now writes run-local metrics summaries and cleans run-local relative outputs before execution.

## Ranked Next 5 Highest-Leverage Follow-Ups

1. Add direct HUSAI-checkpoint CE-Bench adapter/evaluation path with matched baselines and manifests.
2. Run a matched-budget architecture frontier sweep on external benchmarks (TopK, JumpReLU, BatchTopK, Matryoshka, RouteSAE, HierarchicalTopK).
3. Improve HUSAI external metrics with data/width/layer scaling experiments and confidence intervals.
4. Implement consistency-objective v2 (assignment-aware/joint multi-seed) with explicit external-metric acceptance criteria.
5. Add transcoder + random-model + OOD stress controls as mandatory release gates.

## Primary-Source References

- Paulo & Belrose (2025): https://arxiv.org/abs/2501.16615
- Song et al. (2025): https://arxiv.org/abs/2505.20254
- SAEBench (ICML 2025): https://proceedings.mlr.press/v267/karvonen25a.html
- SAEBench repository: https://github.com/adamkarvonen/SAEBench
- CE-Bench (2025): https://arxiv.org/abs/2509.00691
- CE-Bench repository: https://github.com/Yusen-Peng/CE-Bench
- OpenAI SAE scaling/eval: https://arxiv.org/abs/2406.04093
- JumpReLU: https://arxiv.org/abs/2407.14435
- BatchTopK: https://arxiv.org/abs/2412.06410
- Matryoshka SAEs: https://arxiv.org/abs/2503.17547
- Route Sparse Autoencoders: https://aclanthology.org/2025.emnlp-main.346/
- HierarchicalTopK SAEs: https://aclanthology.org/2025.emnlp-main.515/
- Transcoders Beat SAEs: https://arxiv.org/abs/2501.18823
- Automated metrics vs random transformers: https://arxiv.org/abs/2501.17727

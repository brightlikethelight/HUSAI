# Reliability-First Evaluation of SAE Consistency in HUSAI: Internal Gains, External Reality Check

## Abstract
We conducted a reliability-first, benchmark-aware research cycle for sparse autoencoder (SAE) consistency in the HUSAI repository. Before new hypothesis testing, we fixed reproducibility-critical issues (missing tracked data package, dependency incompatibility, and path portability failures), added fail-fast CI smoke, and hardened artifact manifests. We then ran remote GPU reproductions and ablations: trained-vs-random consistency remained statistically positive but small (`delta PWMCC +0.001230`, one-sided `p=8.629e-03`), while geometry ablations showed large regime dependence (`d_sae=64, k=32` yielded `delta +0.119986`). We also completed official SAEBench harness execution (`returncode=0`) and analyzed 113 matched probe datasets; aggregate SAE-vs-baseline deltas were negative (mean `test_auc` delta `-0.0651`). These results support strong claims about engineering rigor and internal regime effects, but do not support SOTA-facing external claims. We provide a ranked experiment roadmap centered on direct HUSAI benchmark integration, architecture-frontier baselines, and stronger consistency objectives.

## 1. Introduction
Feature consistency across random seeds is a core requirement for reliable mechanistic interpretations. Prior work shows SAEs trained on identical data can still produce divergent feature sets, challenging naive reproducibility assumptions. This repository focuses on controlled algorithmic settings to isolate consistency dynamics and test interventions.

This paper reports an end-to-end cycle that explicitly prioritizes:
- reproducibility and environment correctness,
- artifact-grounded conclusions,
- external-benchmark reality checks.

## 2. Related Work
Consistency and benchmark framing:
- Paulo & Belrose (2025): seed-dependent divergence in SAE features [https://arxiv.org/abs/2501.16615].
- Song et al. (2025): consistency should be first-class for interpretability practice [https://arxiv.org/abs/2505.20254].
- SAEBench (ICML 2025): broad benchmark evidence, warning against over-reliance on narrow proxies [https://proceedings.mlr.press/v267/karvonen25a.html].
- CE-Bench (2025): judge-free benchmark aligned with SAEBench trends [https://arxiv.org/abs/2509.00691].

Method and architecture frontier:
- OpenAI scaling/evaluation framework [https://arxiv.org/abs/2406.04093].
- JumpReLU [https://arxiv.org/abs/2407.14435], BatchTopK [https://arxiv.org/abs/2412.06410], Matryoshka SAEs [https://arxiv.org/abs/2503.17547].
- RouteSAE [https://aclanthology.org/2025.emnlp-main.346/] and HierarchicalTopK [https://aclanthology.org/2025.emnlp-main.515/].

Control and alternative-representation signals:
- Randomly initialized transformer representations can remain strong on interpretable tasks [https://arxiv.org/abs/2501.18823].
- Transcoders can outperform SAEs in some interpretability settings [https://arxiv.org/abs/2501.17727].

## 3. Repository Reliability Program
Implemented reliability upgrades:
- CI: `.github/workflows/ci.yml`
- smoke pipeline: `scripts/ci/smoke_pipeline.sh`
- path portability fixes in:
  - `scripts/experiments/run_phase4a_reproduction.py`
  - `scripts/experiments/run_core_ablations.py`
  - `scripts/experiments/run_external_benchmark_slice.py`
- benchmark harness hardening:
  - `scripts/experiments/run_official_external_benchmarks.py`

Critical issues fixed during remote rerun:
1. `.gitignore` rule unintentionally excluded `src/data/`.
2. NumPy constraints were incompatible with TransformerLens requirements.
3. Relative-path failures caused script breakage on clean remote environments.

## 4. Experimental Protocol

### 4.1 Internal consistency metrics
Primary:
- Pairwise decoder PWMCC (trained and random controls)
- trained-vs-random deltas and effect size

Secondary:
- MSE reconstruction
- explained variance (EV)

### 4.2 Reproduction and ablation runs
- Phase 4a trained-vs-random reproduction with fixed seeds.
- Phase 4c k-sweep and d_sae-sweep under shared configuration.
- Phase 4e benchmark-aligned internal slice.

### 4.3 External benchmark run
- Official harness run:
  - `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/`
- SAEBench command executed through harness with logs and manifests.

## 5. Results

### 5.1 Phase 4a (remote B200)
- trained mean PWMCC: `0.300059`
- random mean PWMCC: `0.298829`
- delta: `+0.001230`
- one-sided p-value: `8.629e-03`

Conclusion:
- Positive but small consistency signal in this rerun.

### 5.2 Phase 4c core ablations
Best k-sweep condition:
- `k=8, d_sae=128`, delta `+0.009773`, ratio `1.0398`.

Best d_sae-sweep condition:
- `d_sae=64, k=32`, delta `+0.119986`, ratio `1.5272`.

Conclusion:
- Hyperparameter geometry drives large variance in consistency outcomes.

### 5.3 Adaptive L0 follow-up (strongest internal positive)
- Matched comparison (`k=4` vs `k=32`) showed trained PWMCC gain `+0.05701` with positive bootstrap CI.

Conclusion:
- Low-L0 calibration is currently the strongest validated intervention in this repo.

### 5.4 Official SAEBench execution
Harness status:
- command attempted and succeeded (`returncode=0`).

Aggregate analysis over 113 matched probe datasets (best SAE over `k in {1,2,5}` minus baseline logreg):
- mean `test_f1` delta: `-0.0952`
- mean `test_acc` delta: `-0.0513`
- mean `test_auc` delta: `-0.0651`
- `test_auc` wins/losses/ties: `21/88/4`

Conclusion:
- External benchmark evidence in this setup is negative relative to baseline probes.

## 6. Discussion

### 6.1 What is strongly supported
- Reliability and reproducibility posture improved substantially.
- Internal consistency effects exist and are highly regime-dependent.
- Adaptive L0 remains a practical consistency lever.
- Official benchmark execution infrastructure works end-to-end.

### 6.2 What is not supported
- SOTA claims.
- Claims that current method externally dominates benchmark baselines.

### 6.3 Why this is still high-value progress
- Negative external signals are actionable: they prevent overclaiming and sharpen the next experiment frontier.
- The repository now supports faster, cleaner falsification cycles.

## 7. Limitations
- CE-Bench has not yet been executed in this environment.
- Official run used a public SAEBench SAE target; HUSAI checkpoint adapter path remains incomplete.
- Current task family remains narrow relative to broad-language benchmark diversity.

## 8. Reproducibility Checklist
- CI smoke + quality: `.github/workflows/ci.yml`
- smoke script: `scripts/ci/smoke_pipeline.sh`
- experiment ledger: `EXPERIMENT_LOG.md`
- consistency audit: `scripts/analysis/verify_experiment_consistency.py`
- official benchmark harness: `scripts/experiments/run_official_external_benchmarks.py`
- external run artifacts: `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/`

## 9. Ranked Next Steps
1. Benchmark HUSAI-produced checkpoints directly in SAEBench and CE-Bench.
2. Execute matched-budget architecture frontier suite (TopK/JumpReLU/BatchTopK/Matryoshka/RouteSAE/HierarchicalTopK).
3. Implement assignment-aware consistency objective v2 with strict CI acceptance rules.
4. Add transcoder baseline arm under identical budgets.
5. Add random-model and OOD stress-test gates before any narrative claim update.

## 10. Broader Impact and Research Hygiene
This cycle demonstrates that rigorous negative results can be as valuable as positive ones when they are reproducible, well-instrumented, and benchmark-grounded. The main risk in this area is premature interpretability claims from unstable or unbenchmarked representations. The repository now has better safeguards against that failure mode, but enforcement depends on maintaining benchmark-first reporting discipline.

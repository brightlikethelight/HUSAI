# Adaptive L0 Calibration for SAE Consistency in HUSAI

## Abstract
Sparse autoencoder (SAE) interpretability workflows are sensitive to cross-seed instability, so we executed a reliability-first then science-first program in this repository. After hardening the pipeline (CI smoke + reproducible runners + path portability), we ran two high-impact follow-ups: (1) adaptive L0 calibration with retrain, and (2) a consistency-first objective sweep via decoder-alignment regularization. The adaptive L0 experiment provides strong evidence that lower-L0 settings can substantially improve cross-seed feature consistency in this task regime: with matched seeds/epochs, `k=4` outperforms `k=32` by `+0.0570` trained PWMCC (95% bootstrap CI `[+0.0548, +0.0592]`). The consistency-regularization sweep shows only weak directional gains (`+0.00067`, CI crossing zero), so it is currently exploratory rather than validated. We conclude that in this repository, L0 calibration is a high-leverage control knob for stability, while objective-level consistency regularization needs stronger formulations and larger studies.

## 1. Problem Statement
This repository studies whether SAEs trained on identical activations but different random seeds learn reproducible features. Recent work reports substantial seed sensitivity in SAE feature dictionaries, raising concerns for downstream interpretability claims.

Working question for this phase:
- Which interventions materially improve reproducibility in this repo's modular-arithmetic setting without collapsing reconstruction quality?

## 2. Related Work
Key context used to ground this study:
- Paulo & Belrose (2025): SAEs trained on same data can learn different features, with architecture-dependent consistency behavior. Source: https://arxiv.org/abs/2501.16615
- SAEBench (ICML 2025): benchmarking indicates proxy metrics alone are insufficient; practical evaluation requires broader suites. Source: https://proceedings.mlr.press/v267/karvonen25a.html
- Song et al. (2025): consistency should be a first-class objective in mechanistic interpretability workflows. Source: https://arxiv.org/abs/2505.20254
- OpenAI SAE scaling/evaluation (2024): large-scale SAE design and evaluation principles. Source: https://arxiv.org/abs/2406.04093
- Architectural advances (JumpReLU, BatchTopK, Matryoshka SAEs):
  - https://arxiv.org/abs/2407.14435
  - https://arxiv.org/abs/2412.06410
  - https://arxiv.org/abs/2503.17547
- CE-Bench (2025): external judge-free benchmark approach aligned with SAEBench signals. Source: https://arxiv.org/abs/2509.00691

## 3. Repository Reliability Preconditions
Before science execution, we ensured infrastructure correctness:
- CI workflow with fail-fast smoke and quality gates (`.github/workflows/ci.yml`)
- reproducible smoke script (`scripts/ci/smoke_pipeline.sh`)
- portable paths (removed hardcoded absolute roots in analysis/experiments scripts)
- stable experiment runners with manifest logging

Regression status:
- `pytest tests -q`: 83 passed
- local smoke pipeline: pass

## 4. Methods

### 4.1 Metrics
Primary:
- PWMCC on normalized decoder columns (symmetric max-cosine mean)
- trained-vs-random delta PWMCC
- conservative lower bound (trained CI lower - random CI upper)

Secondary:
- explained variance (EV)
- MSE reconstruction
- L0 observed sparsity

Uncertainty:
- bootstrap 95% confidence intervals over pairwise statistics

### 4.2 Experiment A: Adaptive L0 Calibration + Retrain
Runner:
- `scripts/experiments/run_adaptive_l0_calibration.py`

Protocol:
1. Search `k in {4,8,12,16,24,32,48,64}` at fixed `d_sae=128`.
2. Select `k` by maximizing conservative delta LCB, subject to EV floor (`>=0.20`).
3. Retrain selected `k` on expanded seeds.
4. Run fair matched-control retrain at `k=32` with identical seed/epoch budget.

Artifacts:
- selected run: `results/experiments/adaptive_l0_calibration/run_20260212T145416Z/`
- matched control run: `results/experiments/adaptive_l0_calibration/run_20260212T145727Z/`

### 4.3 Experiment B: Consistency-First Objective Sweep
Runner:
- `scripts/experiments/run_consistency_regularization_sweep.py`

Protocol:
- Fix geometry `d_sae=128, k=4`.
- Train reference model (seed 42), then train other seeds with
  `loss = MSE + lambda * alignment_penalty(decoder, ref_decoder)`.
- Sweep `lambda in {0, 1e-4, 5e-4, 1e-3, 2e-3}`.
- Select lambda by conservative delta LCB under EV-drop constraint (`<=0.05`).

Artifacts:
- `results/experiments/consistency_objective_sweep/run_20260212T145529Z/`

## 5. Results

### 5.1 Adaptive L0 Search
At `d_sae=128`, search-phase trained-random deltas monotonically favored lower `k` values.

| k | trained PWMCC | random PWMCC | delta | conservative delta LCB | EV |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.25991 | 0.24556 | +0.01435 | +0.01016 | 0.2644 |
| 8 | 0.25650 | 0.24556 | +0.01094 | +0.00651 | 0.3165 |
| 12 | 0.25507 | 0.24556 | +0.00952 | +0.00516 | 0.3528 |
| 16 | 0.25420 | 0.24556 | +0.00864 | +0.00434 | 0.3808 |
| 24 | 0.25352 | 0.24556 | +0.00797 | +0.00419 | 0.4247 |
| 32 | 0.25224 | 0.24556 | +0.00669 | +0.00308 | 0.4575 |
| 48 | 0.25142 | 0.24556 | +0.00586 | +0.00202 | 0.5061 |
| 64 | 0.25139 | 0.24556 | +0.00584 | +0.00199 | 0.5338 |

Selected by criterion: **k=4**.

### 5.2 Retrain and Fair Control
Retrain (`k=4`, 8 seeds, 40 epochs):
- trained PWMCC `0.32191`
- random PWMCC `0.24624`
- delta `+0.07567`
- conservative LCB `+0.07256`
- EV `0.53170`

Matched control (`k=32`, same seeds/epochs):
- trained PWMCC `0.26490`
- random PWMCC `0.24624`
- delta `+0.01866`
- conservative LCB `+0.01676`
- EV `0.68875`

Direct effect (`k=4` - `k=32`, trained PWMCC):
- `+0.05701`
- bootstrap 95% CI `[+0.05482, +0.05921]`

Interpretation:
- Strong consistency gain at lower L0 in this regime.
- Clear stability-quality tradeoff (higher consistency at `k=4`, higher EV at `k=32`).

### 5.3 Consistency-Regularization Sweep

| lambda | trained PWMCC | random PWMCC | delta | conservative delta LCB | EV | alignment-to-ref |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.27422 | 0.24556 | +0.02866 | +0.02456 | 0.35892 | 0.27247 |
| 0.0001 | 0.27437 | 0.24556 | +0.02881 | +0.02476 | 0.35893 | 0.27287 |
| 0.0005 | 0.27449 | 0.24556 | +0.02893 | +0.02492 | 0.35894 | 0.27324 |
| 0.001 | 0.27462 | 0.24556 | +0.02906 | +0.02510 | 0.35897 | 0.27365 |
| 0.002 | 0.27489 | 0.24556 | +0.02933 | +0.02543 | 0.35897 | 0.27444 |

Selected lambda: `0.002`.

Baseline vs selected:
- trained PWMCC gain: `+0.00067`
- bootstrap 95% CI for gain: `[-0.00246, +0.00376]`

Interpretation:
- Directionally positive, but not statistically resolved in this run.
- No detectable EV penalty at tested lambdas.

## 6. Discussion

### 6.1 What is strongly supported
- Adaptive L0 calibration is a high-leverage and reproducible intervention for consistency improvement in this repo's setting.
- The consistency-quality tradeoff is not hypothetical; it appears directly in matched-control retrains.

### 6.2 What remains inconclusive
- Current consistency-regularization formulation shows only small gains; confidence interval includes zero.
- This likely needs richer coupling (e.g., multi-model joint training, assignment-aware regularizers, or larger-scale seeds/tasks).

### 6.3 External-claim status
- Internal benchmark-aligned gating can pass.
- An official benchmark harness is now implemented and artifact-logged: `scripts/experiments/run_official_external_benchmarks.py` with run `results/experiments/phase4e_external_benchmark_official/run_20260212T151416Z/`.
- Official external benchmark claim remains blocked until SAEBench/CE-Bench commands are actually executed through that harness.

## 7. Limitations
- Task family is still narrow (algorithmic modular arithmetic).
- CPU-only runs constrain scale and architecture breadth.
- Pairwise PWMCC is useful but not sufficient for downstream utility claims.
- Literature suggests broader functional metrics should complement overlap metrics.

## 8. Reproducibility Checklist
- CI workflow: `.github/workflows/ci.yml`
- smoke pipeline: `scripts/ci/smoke_pipeline.sh`
- experiment log: `EXPERIMENT_LOG.md`
- follow-up report: `HIGH_IMPACT_FOLLOWUPS_REPORT.md`
- adaptive L0 runner: `scripts/experiments/run_adaptive_l0_calibration.py`
- consistency sweep runner: `scripts/experiments/run_consistency_regularization_sweep.py`
- official benchmark harness: `scripts/experiments/run_official_external_benchmarks.py`
- result-consistency audit: `scripts/analysis/verify_experiment_consistency.py`
- latest consistency report: `results/analysis/experiment_consistency_report.md`

## 9. Next Technical Steps (Ranked)
1. Execute official SAEBench and CE-Bench commands through the new harness and publish score manifests.
2. Integrate adaptive-L0 selector into default training pipeline (auto-calibration mode).
3. Replace single-reference regularizer with assignment-aware or joint multi-seed objectives.
4. Add modern architecture frontier baselines (JumpReLU, BatchTopK, Matryoshka, HierarchicalTopK, RouteSAE) under matched compute.
5. Add causal-faithfulness metrics and OOD stress tests as co-primary model-selection endpoints.

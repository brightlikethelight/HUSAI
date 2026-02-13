# Reliability-First Evaluation of SAE Consistency in HUSAI: Internal Gains and External Benchmark Reality

## Abstract
We executed a reliability-first research cycle for sparse autoencoder (SAE) consistency in HUSAI. We first fixed reproducibility-critical issues (CI smoke coverage, path portability, benchmark harness logging, and artifact-grounded consistency checks), then ran remote GPU reproductions and external benchmarks. Internal trained-vs-random consistency remained statistically positive but small (`delta PWMCC +0.001230`, one-sided `p=8.629e-03`), while ablations showed strong regime dependence (`d_sae=64, k=32` yielded `delta +0.119986`). We completed official SAEBench harness execution and direct HUSAI custom-checkpoint SAEBench evaluation across three seeds. HUSAI custom results were stable across seeds but below baseline probes (best AUC mean `0.622601`, delta vs baseline `-0.051801`, 95% CI `[-0.052496, -0.051105]`). These results support stronger claims about engineering rigor and uncertainty-aware evaluation, but do not support external SOTA-style claims. We provide a ranked roadmap centered on CE-Bench execution and architecture/objective variants designed to improve external metrics under strict reproducibility constraints.

## 1. Introduction
Feature-level reproducibility is foundational for mechanistic interpretability. Prior work shows SAEs trained on identical data with different seeds can learn divergent dictionaries, challenging naive interpretation stability assumptions.

This cycle targeted two goals:
1. make the repository operationally trustworthy across environments,
2. evaluate consistency interventions under external benchmark pressure.

## 2. Related Work
Consistency and evaluation framing:
- Paulo & Belrose (2025): https://arxiv.org/abs/2501.16615
- Song et al. (2025): https://arxiv.org/abs/2505.20254
- SAEBench (ICML 2025): https://proceedings.mlr.press/v267/karvonen25a.html
- CE-Bench (2025): https://arxiv.org/abs/2509.00691

Architecture/objective frontier:
- OpenAI scaling/eval framework: https://arxiv.org/abs/2406.04093
- JumpReLU: https://arxiv.org/abs/2407.14435
- BatchTopK: https://arxiv.org/abs/2412.06410
- Matryoshka: https://arxiv.org/abs/2503.17547
- RouteSAE: https://aclanthology.org/2025.emnlp-main.346/
- HierarchicalTopK: https://aclanthology.org/2025.emnlp-main.515/

Controls and alternative hypotheses:
- Transcoders Beat SAEs: https://arxiv.org/abs/2501.18823
- Metrics vs random-transformer control: https://arxiv.org/abs/2501.17727

## 3. Reliability Program and Experimental Setup

### 3.1 Reliability upgrades
- CI workflow: `.github/workflows/ci.yml`
- fail-fast smoke pipeline: `scripts/ci/smoke_pipeline.sh`
- path portability improvements in core runners
- benchmark harness logging fix: stream stdout/stderr to files
- claim-consistency check script: `scripts/analysis/verify_experiment_consistency.py`

### 3.2 Core protocol
- internal consistency metric: decoder PWMCC (trained vs random controls)
- uncertainty reporting: CIs and effect sizes where applicable
- benchmark logging: command/config hash/artifact manifests

### 3.3 Remote environment
- platform: RunPod single NVIDIA B200
- execution mode: reproducible command scripts with per-run artifacts under `results/experiments/`

## 4. Results

### 4.1 Internal reproduction (Phase 4a)
Artifact:
- `results/experiments/phase4a_trained_vs_random/results.json`

Results:
- trained mean PWMCC: `0.300059`
- random mean PWMCC: `0.298829`
- delta: `+0.001230`
- one-sided Mann-Whitney p-value: `8.629e-03`

Interpretation:
- trained-vs-random signal exists but is small in this rerun.

### 4.2 Core ablations (Phase 4c)
Artifact:
- `results/experiments/phase4c_core_ablations/run_20260212T200711Z/results.json`

Best `k`-sweep condition:
- `k=8`, `d_sae=128`, delta `+0.009773`

Best `d_sae`-sweep condition:
- `d_sae=64`, `k=32`, delta `+0.119986`

Interpretation:
- geometry/hyperparameter regime materially changes consistency outcomes.

### 4.3 Official SAEBench execution (public SAE target)
Artifact:
- `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/`

Aggregate result (best SAE over `k in {1,2,5}` minus logreg baseline; 113 datasets):
- mean `test_f1` delta: `-0.0952`
- mean `test_acc` delta: `-0.0513`
- mean `test_auc` delta: `-0.0651`

Interpretation:
- external evidence is negative in this setup.

### 4.4 HUSAI custom-checkpoint SAEBench (multi-seed)
Run artifacts:
- `results/experiments/phase4e_external_benchmark_official/run_20260213T024329Z/`
- `results/experiments/phase4e_external_benchmark_official/run_20260213T031247Z/`
- `results/experiments/phase4e_external_benchmark_official/run_20260213T032116Z/`

Tracked aggregate summary:
- `docs/evidence/phase4e_husai_custom_multiseed/summary.json`

| Seed | Run ID | Best k | Best AUC | Baseline AUC | Delta AUC |
|---:|---|---:|---:|---:|---:|
| 42 | run_20260213T024329Z | 5 | 0.623311 | 0.674402 | -0.051091 |
| 123 | run_20260213T031247Z | 5 | 0.622244 | 0.674402 | -0.052158 |
| 456 | run_20260213T032116Z | 5 | 0.622249 | 0.674402 | -0.052153 |

Aggregate:
- best AUC mean ± std: `0.622601 ± 0.000615`
- best AUC 95% CI: `[0.621905, 0.623297]`
- delta AUC mean ± std: `-0.051801 ± 0.000615`
- delta AUC 95% CI: `[-0.052496, -0.051105]`

Interpretation:
- custom checkpoint path is reproducible and low-variance,
- but consistently below baseline probes.

## 5. Discussion

### 5.1 Supported claims
- engineering/reproducibility posture is substantially improved,
- internal consistency effects are real and regime-sensitive,
- external benchmark pathway now includes direct HUSAI checkpoint evaluation.

### 5.2 Unsupported claims
- external superiority and SOTA-style claims are not supported by current data.

### 5.3 Why this is still high-value progress
- This cycle converted uncertain benchmark narratives into hard, reproducible evidence.
- Negative external results narrowed the search space for future work.

## 6. Limitations
- CE-Bench has not yet been executed in this environment.
- Current external runs use one model/hook family; broader generalization remains untested.
- Stress controls (random-model/OOD/transcoder) are not yet release-gated.

## 7. Ranked Next Steps
1. Execute CE-Bench for HUSAI and baseline targets with full manifests.
2. Run matched-budget architecture frontier sweep on external metrics.
3. Run external-metric scaling (`token budget`, `hook layer`, `d_sae`).
4. Implement assignment-aware consistency objective v2 and re-evaluate externally.
5. Add stress-gated claim policy (transcoder control, random-model control, OOD).

## 8. Reproducibility Checklist
- CI and smoke: `.github/workflows/ci.yml`, `scripts/ci/smoke_pipeline.sh`
- experiment ledger: `EXPERIMENT_LOG.md`
- consistency audit: `scripts/analysis/verify_experiment_consistency.py`
- benchmark harness: `scripts/experiments/run_official_external_benchmarks.py`
- multi-seed custom external evidence: `docs/evidence/phase4e_husai_custom_multiseed/`

## 9. Broader Impact
A core risk in interpretability research is overclaiming from unstable or weakly benchmarked representations. This cycle emphasizes a benchmark-first, uncertainty-aware reporting standard that makes failures visible and therefore actionable.

# Experiment Program (Phase 4)

Date: 2026-02-12
Goal: take this repository from partially broken state to a rigorous, benchmark-aligned SAE stability research program.

## 0) Gating Principle

No headline experiments until the pipeline is reproducibly executable.

Gate A (must pass first):
1. baseline transformer train script runs from documented command path
2. activation extraction runs from documented path
3. at least one SAE training path runs end-to-end
4. one e2e smoke test in CI passes on CPU

## 1) Research Hypotheses

H1. **Rank/Sparsity Matching Hypothesis**
- Stability improves when `d_sae` and sparsity `k` are matched to activation effective rank.

H2. **Stability-Quality Tradeoff Hypothesis**
- Reconstruction quality and feature consistency are partially decoupled; some settings improve one at the expense of the other.

H3. **Architecture Effect Hypothesis**
- Modern architectures (JumpReLU/Matryoshka/JSAE-style variants) can shift Pareto front vs classic TopK/ReLU.

H4. **Metric Robustness Hypothesis**
- Stability conclusions are only trustworthy when random baselines and causal checks are included.

H5. **Task-to-Benchmark Transfer Hypothesis**
- Patterns from modular arithmetic transfer partially, but not fully, to benchmark-style settings.

## 2) Phase 4a - Reproduce Current Repo Claims

Experiment 4a.1: Baseline TopK replication
- Change list: none (after reliability fixes)
- Config: `d_model=128`, `d_sae=1024`, `k=32`, 5 seeds
- Metrics: PWMCC mean/std, EV, MSE, dead-neuron %, L0
- Runtime estimate: 2-5 GPU hours (or slower CPU fallback)
- Failure modes: SAE path still broken; unstable env; artifact naming drift

Experiment 4a.2: Random baseline replication
- Change list: add standard random-decoder/control generation script if missing
- Metrics: trained-vs-random PWMCC gap with confidence intervals
- Runtime estimate: <1 GPU hour or CPU
- Failure modes: mismatch in decoder normalization conventions

Deliverable:
- Repro table matching `results/analysis/trained_vs_random_pwmcc.json` plus refreshed run manifest.

## 3) Phase 4b - Baseline Suite

Experiment 4b.1: Linear/No-sparsity AE baseline
- Hypothesis: reconstruction can remain high with weaker consistency constraints
- Change list: add linear AE baseline trainer
- Metrics: EV/MSE, PWMCC, random-baseline ratio

Experiment 4b.2: ReLU vs TopK apples-to-apples
- Hypothesis: architecture impacts Pareto frontier, not just scalar quality
- Change list: standardized training budget and seeds for both
- Metrics: PWMCC, EV, MSE, L0, dead-neuron %, runtime

Experiment 4b.3: External baseline import
- Hypothesis: SAELens reference settings provide stronger comparators
- Change list: add compatibility harness to run one SAELens baseline config cleanly
- Metrics: same as above plus benchmark compatibility checks

## 4) Phase 4c - Core Ablations

Experiment 4c.1: `k` sweep at fixed `d_sae`
- Sweep: `k in {4,8,16,24,32,48,64}`
- Expected delta: non-trivial stability curve and Pareto shifts

Experiment 4c.2: `d_sae` sweep at fixed `k`
- Sweep: `d_sae in {64,80,128,256}` equivalent expansion settings
- Expected delta: best stability near effective-rank-matched regime

Experiment 4c.3: Decoder normalization variants
- Compare: on/off and axis conventions (after correctness fix)
- Expected delta: measurable effect on stability and dead features

Experiment 4c.4: Training length/dynamics
- Checkpoints over epochs for stability trajectory
- Expected delta: stability may improve with training length in some regimes

## 5) Phase 4d - SOTA-Chasing Variants

Experiment 4d.1: JumpReLU variant
- Why: strong modern baseline in literature
- Expected delta: improved consistency-quality frontier in some regimes

Experiment 4d.2: Matryoshka/JSAE-inspired variant
- Why: hierarchical/joint sparsity may improve structural robustness
- Expected delta: better multiscale consistency under fixed compute

Experiment 4d.3: Theory-guided regularization (GBA-style principles)
- Why: improve identifiability pressure
- Expected delta: stronger trained-vs-random separation

## 6) Phase 4e - Stress Tests

Experiment 4e.1: Task transfer
- modular arithmetic -> copy task -> one additional algorithmic task

Experiment 4e.2: Data regime sensitivity
- data fraction sweep (10%, 30%, 70%, 100%)

Experiment 4e.3: Activation source sensitivity
- layer and hook-point sweeps (`resid_post`, `mlp_out`, `attn_out`)

Experiment 4e.4: Benchmark-lite external validation
- at least one CE-Bench-like or SAEBench-like slice

## 7) Metrics and Logging Spec

Required metrics (every run):
- primary: PWMCC mean/std, trained-vs-random ratio
- quality: EV, MSE
- sparsity: L0, dead-neuron %, activation frequency stats
- provenance: seed, architecture, `d_sae`, `k`, task, layer/hook, commit hash
- runtime: wall-clock, device, batch size, tokens/samples

Artifact layout:
- `results/experiments/<exp_name>/<run_id>/`
  - `config.yaml`
  - `metrics.json`
  - `summary.md`
  - `plots/*.png`
  - `checkpoints/*.pt`

Run naming convention:
- `<phase>_<task>_<arch>_dsae<d>_k<k>_seed<seed>_<date>`

## 8) Compute and Timeline Estimates

Assuming small-model algorithmic tasks:
- single SAE run: ~30-60 min GPU (or 3-6h CPU)
- 5-seed condition: ~2-5 GPU hours
- full Phase 4a+4b+4c core: ~20-40 GPU hours

Proposed schedule:
1. Week 1: reliability gates + Phase 4a
2. Week 2: Phase 4b + 4c
3. Week 3: Phase 4d pilot + 4e stress tests
4. Week 4: analysis consolidation + writeup artifacts

## 9) Failure Handling

If a run fails:
1. record failure in `EXPERIMENT_LOG.md` with command/config/stack trace
2. tag as infra issue vs scientific negative result
3. only rerun after fix commit is recorded

## 10) Success Criteria

Minimum:
- reproducible end-to-end pipeline with CI smoke checks
- trained-vs-random baseline reproduced with manifest
- at least one robust ablation map (`k` or `d_sae`) with confidence intervals

Target:
- clear stability-quality Pareto analysis across >=2 architectures
- one external benchmark-aligned validation slice
- publication-grade figures/tables with artifact provenance

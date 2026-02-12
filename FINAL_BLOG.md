# We Ran the Highest-Impact SAE Follow-Ups. Here’s What Actually Worked.

HUSAI asks a simple question with big consequences for interpretability:

If you train SAEs on the same activations with different seeds, do you get the same features?

This cycle focused on four high-leverage updates:
1. Adaptive L0 calibration (choose `k` systematically, then retrain)
2. Consistency-first training objective sweep
3. Official SAEBench/CE-Bench harness preflight
4. Automated artifact-driven claim-consistency audit

## Setup in one paragraph

All follow-up runs used layer-1 answer-position activations from the same transformer checkpoint, TopK SAEs, multi-seed evaluation, random-decoder controls, and bootstrap confidence intervals. We tracked both consistency and quality:
- consistency: PWMCC and trained-vs-random delta
- quality: explained variance (EV) and MSE

## Follow-up #1: Adaptive L0 calibration

### What we did

We added `scripts/experiments/run_adaptive_l0_calibration.py`.

It performs:
- search over `k` values at fixed `d_sae=128`
- conservative selection criterion (maximize trained-random delta lower bound)
- EV floor constraint
- expanded retrain at selected `k`

Then we ran a fair control: same retrain seeds/epochs at `k=32`.

### What happened

`k=4` was selected.

Matched retrains:
- `k=4`: trained PWMCC `0.32191`, random `0.24624`, delta `+0.07567`
- `k=32`: trained PWMCC `0.26490`, random `0.24624`, delta `+0.01866`

Direct trained-PWMCC improvement (`k=4 - k=32`):
- `+0.05701`
- 95% CI: approximately `[+0.055, +0.059]`

Interpretation:
- In this regime, L0 calibration is a major consistency lever.
- Tradeoff remains real: `k=32` retains higher EV, `k=4` retains higher consistency.

## Follow-up #2: Consistency-first objective sweep

### What we did

We added `scripts/experiments/run_consistency_regularization_sweep.py`.

Protocol:
- Train a reference SAE (seed 42)
- Train other seeds with
  `loss = MSE + lambda * alignment_penalty(decoder, ref_decoder)`
- Sweep `lambda in {0, 1e-4, 5e-4, 1e-3, 2e-3}`

### What happened

Best lambda by our criterion: `0.002`.

Effect size stayed small:
- baseline (`lambda=0`): delta `+0.02866`
- best (`lambda=0.002`): delta `+0.02933`
- gain: `+0.00067`
- 95% CI for gain crosses zero

Interpretation:
- Directionally positive, but statistically unresolved in this run.
- This is a pilot signal, not a robust win yet.

## Follow-up #3: Official benchmark harness (new)

We added `scripts/experiments/run_official_external_benchmarks.py` to standardize SAEBench/CE-Bench execution with manifests and logs.

Preflight artifact run:
- `results/experiments/phase4e_external_benchmark_official/run_20260212T151416Z/`

Preflight result:
- SAEBench module: not installed in this workspace
- CE-Bench module: not installed in this workspace
- local SAE checkpoints indexed: `5`

Interpretation:
- We now have a reproducible execution harness.
- External benchmark claims remain blocked until official commands are actually executed through it.

## Follow-up #4: Result-consistency audit (new)

We added `scripts/analysis/verify_experiment_consistency.py` to verify headline conclusions directly from result JSON artifacts.

Outputs:
- `results/analysis/experiment_consistency_report.json`
- `results/analysis/experiment_consistency_report.md`

Current audit status:
- overall pass: `True`
- confirms adaptive low-k advantage over k=32 control
- confirms consistency-regularizer gain remains unresolved

Interpretation:
- This reduces future drift between narrative claims and measurable artifacts.

## What changed in the repo

New scripts:
- `scripts/experiments/run_adaptive_l0_calibration.py`
- `scripts/experiments/run_consistency_regularization_sweep.py`
- `scripts/experiments/run_official_external_benchmarks.py`
- `scripts/analysis/verify_experiment_consistency.py`

Updated docs:
- `RUNBOOK.md`
- `EXPERIMENT_LOG.md`
- `HIGH_IMPACT_FOLLOWUPS_REPORT.md`
- `NOVEL_CONTRIBUTIONS.md`
- `LIT_REVIEW.md`

New Make targets:
- `benchmark-official`
- `audit-results`
- `adaptive-l0`
- `adaptive-l0-control`
- `consistency-sweep`

## What we can now say with confidence

Strong claim:
- Adaptive L0 calibration materially improves cross-seed consistency in this repository’s current setting.

Not-yet-strong claim:
- This first consistency-regularized objective is promising but not validated.

External benchmark status:
- Harness exists; official SAEBench/CE-Bench execution still pending.

## Bottom line

If your goal is reproducible SAE features in this codebase, tune L0 first and enforce artifact-grounded audits. Then run official external benchmarks before making any SOTA-facing claim.

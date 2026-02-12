# We Ran the Highest-Impact SAE Follow-Ups. Here’s What Actually Worked.

HUSAI asks a simple question with big consequences for interpretability:

If you train SAEs on the same activations with different seeds, do you get the same features?

We already stabilized the engineering stack (CI smoke, reproducible runners, fixed pathing/import drift). This round focused on the most important scientific follow-ups:

1. Adaptive L0 calibration (choose `k` systematically, then retrain)
2. Consistency-first training objective sweep

This post reports what changed, what improved, and what is still unresolved.

## The setup in one paragraph

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

### Search result

`k=4` was selected.

Search trend was clear: lower `k` gave larger trained-vs-random consistency deltas, while higher `k` improved EV.

### Fair-control result (the key number)

Matched retrains:
- `k=4`: trained PWMCC `0.32191`, random `0.24624`, delta `+0.07567`
- `k=32`: trained PWMCC `0.26490`, random `0.24624`, delta `+0.01866`

Direct trained-PWMCC improvement (`k=4 - k=32`):
- `+0.05701`
- 95% bootstrap CI `[+0.05482, +0.05921]`

Interpretation:
- In this regime, L0 calibration is not a tiny tweak; it is a major consistency lever.
- The tradeoff is real: `k=32` retains better EV, `k=4` retains better consistency.

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

But effect size was small:
- baseline (`lambda=0`): delta `+0.02866`
- best (`lambda=0.002`): delta `+0.02933`
- gain: `+0.00067`
- 95% CI for gain: `[-0.00246, +0.00376]`

Interpretation:
- Directionally positive, but statistically unresolved in this run.
- This is a pilot signal, not a robust win.

## What changed in the repo

New scripts:
- `scripts/experiments/run_adaptive_l0_calibration.py`
- `scripts/experiments/run_consistency_regularization_sweep.py`

New synthesis doc:
- `HIGH_IMPACT_FOLLOWUPS_REPORT.md`

Updated operational docs:
- `RUNBOOK.md`
- `EXPERIMENT_LOG.md`

New Make targets:
- `adaptive-l0`
- `adaptive-l0-control`
- `consistency-sweep`

## Where to find artifacts

Adaptive L0 runs:
- `results/experiments/adaptive_l0_calibration/run_20260212T145416Z/`
- `results/experiments/adaptive_l0_calibration/run_20260212T145727Z/`

Consistency sweep run:
- `results/experiments/consistency_objective_sweep/run_20260212T145529Z/`

## What we can now say with confidence

Strong claim:
- Adaptive L0 calibration materially improves cross-seed consistency in this repository’s current setting.

Not-yet-strong claim:
- This first consistency-regularized objective is promising but not validated.

External benchmark claim status:
- Still blocked pending official SAEBench/CE-Bench execution (we only have benchmark-aligned internal slices so far).

## Reproduce quickly

```bash
# Adaptive search + retrain
python scripts/experiments/run_adaptive_l0_calibration.py --device cpu

# Matched control at k=32
python scripts/experiments/run_adaptive_l0_calibration.py --device cpu --k-candidates 32

# Consistency-objective sweep
python scripts/experiments/run_consistency_regularization_sweep.py --device cpu --k 4
```

## Bottom line

If your goal is reproducible SAE features in this codebase, tune L0 first. The new consistency regularizer is worth continuing, but it has not yet earned “worked” status by our own statistical bar.

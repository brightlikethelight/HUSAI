# High-Impact Follow-Ups: Execution Report

Date: 2026-02-12

## Scope

This report covers two highest-impact follow-ups from `NOVEL_CONTRIBUTIONS.md`:
1. Adaptive L0 Calibration + Retrain Loop
2. Consistency-First Objective Sweep (decoder-alignment regularization)

All runs are artifact-tracked and reproducible from scripts committed in this repo.

## Experimental Setup

Shared setup:
- Activation source: `results/activations/layer1_answer.pt` (generated from `results/transformer_5000ep/transformer_best.pt`)
- SAE architecture: `TopKSAE` (`src/models/simple_sae.py`)
- Device: CPU
- Training objective: MSE reconstruction (for follow-up runners)
- Stability metric: decoder-weight PWMCC (symmetric max-cosine matching)
- Controls: random decoder baseline matched on `d_model`, `d_sae`, seed count
- Uncertainty: bootstrap 95% CIs + conservative delta lower bound

## Follow-up 1: Adaptive L0 Calibration

Runner:
- `scripts/experiments/run_adaptive_l0_calibration.py`

Primary run artifacts:
- Search + selected retrain: `results/experiments/adaptive_l0_calibration/run_20260212T145416Z/`
- Matched control retrain (`k=32`): `results/experiments/adaptive_l0_calibration/run_20260212T145727Z/`

Search config (`d_sae=128`, 5 seeds, 25 epochs, `k in {4,8,12,16,24,32,48,64}`):
- Selection criterion: maximize conservative lower bound of trained-random delta, with EV floor `>= 0.20`
- Selected `k`: **4**

Search highlights (mean deltas):
- `k=4`: delta `+0.01435`, conservative LCB `+0.01016`, EV `0.2644`
- `k=32`: delta `+0.00669`, conservative LCB `+0.00308`, EV `0.4575`

Retrain at selected `k=4` (8 seeds, 40 epochs):
- trained PWMCC: `0.32191`
- random PWMCC: `0.24624`
- delta: `+0.07567`
- conservative LCB: `+0.07256`
- EV: `0.53170`

Matched control retrain at `k=32` (same seeds/epochs):
- trained PWMCC: `0.26490`
- random PWMCC: `0.24624`
- delta: `+0.01866`
- conservative LCB: `+0.01676`
- EV: `0.68875`

Fair-comparison readout (`k=4` vs `k=32`, trained PWMCC):
- Mean improvement: `+0.05701`
- Bootstrap 95% CI for improvement: `[+0.05482, +0.05921]`

Interpretation:
- Strong evidence that lower L0 (here `k=4`) improves cross-seed consistency over `k=32` for this task regime.
- Tradeoff remains real: `k=4` consistency gain comes with lower EV than `k=32`.

## Follow-up 2: Consistency-First Objective Sweep

Runner:
- `scripts/experiments/run_consistency_regularization_sweep.py`

Artifacts:
- `results/experiments/consistency_objective_sweep/run_20260212T145529Z/`

Config:
- Fixed geometry: `d_sae=128`, `k=4`
- Seeds: reference seed `42` + train seeds `[123,456,789,1011]`
- Lambdas: `{0.0, 1e-4, 5e-4, 1e-3, 2e-3}`
- Epochs: `30`
- Objective for non-reference models: `MSE + lambda * alignment_penalty(decoder, ref_decoder)`

Selection criterion:
- maximize conservative delta LCB under EV-drop constraint (`<= 0.05` vs baseline lambda `0`)
- selected lambda: **0.002**

Best-vs-baseline summary:
- Baseline lambda `0.0`:
  - delta `+0.02866`, EV `0.35892`, alignment-to-ref `0.27247`
- Selected lambda `0.002`:
  - delta `+0.02933`, EV `0.35897`, alignment-to-ref `0.27444`

Statistical readout (trained PWMCC, lambda `0.002` - lambda `0.0`):
- Mean difference: `+0.00067`
- Bootstrap 95% CI: `[-0.00246, +0.00376]`

Interpretation:
- Directionally positive, but effect size is small and not statistically resolved in this run.
- This is currently a **pilot signal**, not strong evidence of objective-level improvement.

## Combined Conclusion

What is strongly supported:
- Adaptive L0 calibration (with fair-control retrains) materially improves consistency metrics in this repository.

What is not yet strongly supported:
- Current consistency-regularization formulation provides only weak, non-significant gains.

Practical recommendation now:
1. Use calibrated low-L0 regime (`k` near 4-8 for `d_sae=128`) when consistency is priority.
2. Keep consistency-regularized objective as exploratory until stronger effect is shown.
3. Preserve both consistency and quality metrics in reporting (PWMCC + EV/MSE).

## Reproduction Commands

Adaptive L0 search + retrain:
```bash
python scripts/experiments/run_adaptive_l0_calibration.py --device cpu
```

Matched `k=32` control retrain:
```bash
python scripts/experiments/run_adaptive_l0_calibration.py --device cpu --k-candidates 32
```

Consistency objective sweep:
```bash
python scripts/experiments/run_consistency_regularization_sweep.py --device cpu --k 4
```

## Literature Context (Primary Sources)

- Paulo & Belrose, 2025, SAEs trained on same data learn different features: https://arxiv.org/abs/2501.16615
- SAEBench (ICML 2025): https://proceedings.mlr.press/v267/karvonen25a.html
- Song et al., consistency-priority position: https://arxiv.org/abs/2505.20254
- OpenAI SAE scaling/eval: https://arxiv.org/abs/2406.04093
- JumpReLU: https://arxiv.org/abs/2407.14435
- BatchTopK: https://arxiv.org/abs/2412.06410
- Matryoshka SAEs: https://arxiv.org/abs/2503.17547
- CE-Bench: https://arxiv.org/abs/2509.00691

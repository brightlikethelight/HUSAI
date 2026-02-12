# High-Impact Follow-Ups: Execution Report

Date: 2026-02-12

## Scope

This report covers four highest-impact follow-ups from `NOVEL_CONTRIBUTIONS.md` and the execution updates completed in this workspace:
1. Adaptive L0 Calibration + Retrain Loop
2. Consistency-First Objective Sweep (decoder-alignment regularization)
3. Official SAEBench/CE-Bench harness preflight (reproducibility + manifest)
4. Automated result-consistency audit (artifact-driven claim checks)

All runs are artifact-tracked and reproducible from scripts committed in this repo.

## Experimental Setup

Shared setup for model experiments:
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
- Bootstrap 95% CI for improvement: `[+0.05475, +0.05924]`

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
- Bootstrap 95% CI: `[-0.00242, +0.00377]`

Interpretation:
- Directionally positive, but effect size is small and not statistically resolved in this run.
- This is currently a pilot signal, not strong evidence of objective-level improvement.

## Follow-up 3: Official SAEBench/CE-Bench Harness (Preflight)

Runner:
- `scripts/experiments/run_official_external_benchmarks.py`

Artifact run:
- `results/experiments/phase4e_external_benchmark_official/run_20260212T151416Z/`

Key outputs:
- `preflight.json`
- `local_sae_index.json`
- `commands.json`
- `summary.md`
- `manifest.json`

Preflight readout:
- SAEBench module availability: `False`
- CE-Bench module availability: `False`
- Local SAE checkpoints indexed for adapter/export: `5`
- No official benchmark command executed in this run (preflight-only by design)

Interpretation:
- The repo now has an official-benchmark execution harness with reproducibility logging.
- The blocking dependency is external benchmark installation/checkout and explicit official command configuration.

## Follow-up 4: Result-Consistency Audit

Runner:
- `scripts/analysis/verify_experiment_consistency.py`

Artifacts:
- `results/analysis/experiment_consistency_report.json`
- `results/analysis/experiment_consistency_report.md`

Readout:
- Overall pass: `True`
- Checks enforced:
  - phase4a training signal remains statistically supported
  - core ablation best-k and best-d_sae deltas remain positive
  - adaptive low-k advantage over k=32 control remains statistically positive
  - consistency-regularizer gain remains unresolved (CI includes zero)

Interpretation:
- This audit codifies conservative claim boundaries directly from artifact JSONs.
- It reduces future drift risk between writeups and measured outcomes.

## Combined Conclusion

What is strongly supported:
- Adaptive L0 calibration (with fair-control retrains) materially improves consistency metrics in this repository.
- The engineering path now supports artifact-grounded consistency checks and official benchmark preflight manifests.

What is not yet strongly supported:
- Current consistency-regularization formulation provides only weak, non-significant gains.
- Official external benchmark performance remains unmeasured until SAEBench/CE-Bench are actually executed.

Practical recommendation now:
1. Use calibrated low-L0 regime (`k` near 4-8 for `d_sae=128`) when consistency is priority.
2. Keep consistency-regularized objective as exploratory until stronger effect is shown.
3. Execute official SAEBench/CE-Bench commands through the new harness before any SOTA claim.
4. Keep the result-consistency audit in the default reporting flow.

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

Official benchmark harness preflight:
```bash
python scripts/experiments/run_official_external_benchmarks.py
```

Result-consistency audit:
```bash
python scripts/analysis/verify_experiment_consistency.py
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

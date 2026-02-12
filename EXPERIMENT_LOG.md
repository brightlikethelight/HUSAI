# Experiment Log

This log records executed commands, outcomes, and artifact paths.

## 2026-02-12 - Smoke and Repro Validation

### Run 1: Baseline transformer smoke train
- Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m scripts.training.train_baseline \
  --config configs/examples/baseline_relu.yaml \
  --epochs 1 \
  --batch-size 128 \
  --no-wandb \
  --save-dir /tmp/husai_demo
```
- Outcome: success
- Key outputs:
  - `/tmp/husai_demo/transformer_final.pt`
- Notes:
  - Validates transformer path via module invocation.

### Run 2: Activation extraction smoke
- Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m scripts.analysis.extract_activations \
  --model-path /tmp/husai_demo/transformer_final.pt \
  --layer 1 \
  --position answer \
  --batch-size 128 \
  --output /tmp/husai_demo/acts.pt
```
- Outcome: success
- Key outputs:
  - `/tmp/husai_demo/acts.pt`
  - `/tmp/husai_demo/acts.meta.pt`

### Run 3: Unit data tests
- Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/unit/test_modular_arithmetic.py -q
```
- Outcome: success
- Result summary: 43 passed

### Run 4: Unit config tests
- Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/unit/test_config.py -q
```
- Outcome: failed
- Result summary: 8 failed, 25 passed
- Notes:
  - Failures are consistent with stale `vocab_size` test assumptions.

### Run 5: Pipeline test
- Command:
```bash
MPLCONFIGDIR=/tmp/mpl KMP_DUPLICATE_LIB_OK=TRUE python tests/test_sae_pipeline.py \
  --transformer-checkpoint results/transformer_5000ep/transformer_best.pt
```
- Outcome: failed
- Failure point:
  - `'ModularArithmeticTransformer' object has no attribute 'd_model'`

### Run 6: SAE CLI import check
- Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m scripts.training.train_sae --help
```
- Outcome: failed
- Failure point:
  - `ModuleNotFoundError: No module named 'scripts.extract_activations'`

### Run 7: SAELens wrapper construction check
- Command: inline Python creating `SAEWrapper` and calling `train_sae`
- Outcome: failed
- Failure point:
  - `TypeError: LanguageModelSAERunnerConfig.__init__() got an unexpected keyword argument 'architecture'`

## Next planned runs (after fixes)

1. Fix import/pathing and SAELens compatibility, then rerun Run 6 and Run 7.
2. Run minimal SAE training smoke with fixed stack.
3. Start Phase 4a multi-seed reproduction.

## 2026-02-12 - Reliability Fix Validation Pass

### Run 8: Main SAE CLI smoke with cached activations
- Command:
```bash
TMPDIR=/tmp KMP_DUPLICATE_LIB_OK=TRUE python scripts/training/train_sae.py \
  --transformer-checkpoint /tmp/husai_demo/transformer_final.pt \
  --config configs/sae/topk_8x_k32.yaml \
  --layer 1 \
  --use-cached-activations /tmp/husai_demo/acts.pt \
  --epochs 1 \
  --batch-size 128 \
  --save-dir /tmp/husai_train_sae \
  --no-wandb \
  --device cpu \
  --quiet
```
- Outcome: success
- Key outputs:
  - `/tmp/husai_train_sae/sae_final.pt`
  - `/tmp/husai_train_sae/training_summary.txt`

### Run 9: `train_simple_sae` test-run smoke
- Command:
```bash
TMPDIR=/tmp MPLCONFIGDIR=/tmp/mpl KMP_DUPLICATE_LIB_OK=TRUE python -m scripts.training.train_simple_sae \
  --transformer results/transformer_5000ep/transformer_best.pt \
  --test-run \
  --architecture topk \
  --save-dir /tmp/husai_simple_test
```
- Outcome: success
- Key outputs:
  - `/tmp/husai_simple_test/sae_final.pt`

### Run 10: Full test suite
- Command:
```bash
TMPDIR=/tmp MPLCONFIGDIR=/tmp/mpl KMP_DUPLICATE_LIB_OK=TRUE pytest tests -q
```
- Outcome: success
- Result summary: 83 passed

### Run 11: run_training wrapper script
- Command:
```bash
./run_training.sh --config configs/examples/baseline_relu.yaml --epochs 1 --batch-size 128 --no-wandb --save-dir /tmp/husai_demo2
```
- Outcome: success
- Key outputs:
  - `/tmp/husai_demo2/transformer_final.pt`

## 2026-02-12 - Follow-up Program Execution (Phase 4a/4c/4e)

### Run 12: Phase 4a full multi-seed reproduction (trained vs random) with manifest
- Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE TMPDIR=/tmp python scripts/experiments/run_phase4a_reproduction.py
```
- Outcome: success
- Key outputs:
  - `results/experiments/phase4a_trained_vs_random/results.json`
  - `results/experiments/phase4a_trained_vs_random/summary.md`
  - `results/experiments/phase4a_trained_vs_random/manifest.json`
  - `results/analysis/trained_vs_random_pwmcc.json` (legacy-compatible refresh)
- Result summary:
  - trained PWMCC mean = `0.301716` (95% CI `[0.301115, 0.302312]`)
  - random PWMCC mean = `0.298829` (95% CI `[0.298161, 0.299439]`)
  - delta = `+0.002886`, ratio = `1.0097`
  - one-sided Mann-Whitney p-value = `1.649e-04`
  - conclusion = `training_signal_present`

### Run 13: Core ablations (initial pass)
- Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE TMPDIR=/tmp MPLCONFIGDIR=/tmp/mpl python scripts/experiments/run_core_ablations.py --device cpu
```
- Outcome: success with quality caveat
- Notes:
  - Initial run used short training and produced negative EV in several conditions.
  - Marked as superseded by Run 14.
- Superseded outputs:
  - `results/experiments/phase4c_core_ablations/run_20260212T091700Z/*`

### Run 14: Core ablations (corrected loss + 20 epochs)
- Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE TMPDIR=/tmp MPLCONFIGDIR=/tmp/mpl python scripts/experiments/run_core_ablations.py --device cpu --epochs 20
```
- Outcome: success
- Key outputs:
  - `results/experiments/phase4c_core_ablations/run_20260212T091848Z/results.json`
  - `results/experiments/phase4c_core_ablations/run_20260212T091848Z/k_sweep_summary.csv`
  - `results/experiments/phase4c_core_ablations/run_20260212T091848Z/d_sae_sweep_summary.csv`
  - `results/experiments/phase4c_core_ablations/run_20260212T091848Z/k_sweep_summary.md`
  - `results/experiments/phase4c_core_ablations/run_20260212T091848Z/d_sae_sweep_summary.md`
  - `results/experiments/phase4c_core_ablations/run_20260212T091848Z/manifest.json`
- Result summary:
  - k sweep (fixed `d_sae=128`): best consistency at `k=8` with PWMCC mean `0.251330` (ratio `1.0138`)
  - d_sae sweep (fixed `k=32`): strongest trained-vs-random lift at `d_sae=64` with PWMCC mean `0.349021` (ratio `1.5245`)
  - reconstruction quality improved monotonically with larger `d_sae` (EV up to `0.5218` at `d_sae=512`)

### Run 15: External benchmark-aligned slice generation (SAEBench/CE-Bench aligned)
- Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/run_external_benchmark_slice.py
```
- Outcome: success
- Key outputs:
  - `results/experiments/phase4e_external_benchmark_slice/benchmark_slice.json`
  - `results/experiments/phase4e_external_benchmark_slice/benchmark_slice.md`
  - `results/experiments/phase4e_external_benchmark_slice/manifest.json`
- Result summary:
  - internal gating pass = `True` (consistency + robustness criteria)
  - official external benchmark claim readiness = `False`
  - explicit blocker retained: official SAEBench/CE-Bench execution not yet run

### Run 16: CI smoke pipeline local validation
- Command:
```bash
TMPDIR=/tmp KMP_DUPLICATE_LIB_OK=TRUE scripts/ci/smoke_pipeline.sh /tmp/husai_ci_smoke_local2
```
- Outcome: success
- Key outputs:
  - `/tmp/husai_ci_smoke_local2/transformer/transformer_final.pt`
  - `/tmp/husai_ci_smoke_local2/acts.pt`
  - `/tmp/husai_ci_smoke_local2/sae/sae_final.pt`

### Run 17: Full test suite regression check
- Command:
```bash
TMPDIR=/tmp MPLCONFIGDIR=/tmp/mpl KMP_DUPLICATE_LIB_OK=TRUE pytest tests -q
```
- Outcome: success
- Result summary: `83 passed in 5.10s`

### Provenance
- Workspace commit at execution time: `535a2df`

### Run 18: Incremental CI-quality command validation (local)
- Command:
```bash
flake8 src/utils/config.py src/data/modular_arithmetic.py \
  scripts/experiments/run_phase4a_reproduction.py \
  scripts/experiments/run_core_ablations.py \
  scripts/experiments/run_external_benchmark_slice.py \
  --max-line-length 130 --extend-ignore=E402 && \
mypy src/utils/config.py src/data/modular_arithmetic.py && \
TMPDIR=/tmp MPLCONFIGDIR=/tmp/mpl KMP_DUPLICATE_LIB_OK=TRUE pytest tests -q
```
- Outcome: success
- Result summary:
  - `mypy`: success on incremental typed subset
  - `pytest`: `83 passed`

## 2026-02-12 - Highest-Impact Follow-up Program

### Run 19: Adaptive L0 calibration (search + retrain)
- Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE TMPDIR=/tmp MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_adaptive_l0_calibration.py --device cpu
```
- Outcome: success
- Run directory:
  - `results/experiments/adaptive_l0_calibration/run_20260212T145416Z/`
- Key results:
  - selected `k=4` (criterion: conservative delta LCB with EV floor)
  - search delta (`k=4`): `+0.01435`
  - retrain at `k=4` (8 seeds, 40 epochs):
    - trained PWMCC `0.32191`
    - random PWMCC `0.24624`
    - delta `+0.07567`
    - conservative LCB `+0.07256`
    - EV `0.53170`

### Run 20: Matched-control retrain for fair comparison (`k=32`)
- Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE TMPDIR=/tmp MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_adaptive_l0_calibration.py --device cpu --k-candidates 32
```
- Outcome: success
- Run directory:
  - `results/experiments/adaptive_l0_calibration/run_20260212T145727Z/`
- Key results (`k=32`, same retrain seeds/epochs):
  - trained PWMCC `0.26490`
  - random PWMCC `0.24624`
  - delta `+0.01866`
  - conservative LCB `+0.01676`
  - EV `0.68875`
- Fair comparison (`k=4` vs `k=32`, trained PWMCC):
  - mean diff `+0.05701`
  - bootstrap 95% CI `[+0.05482, +0.05921]`

### Run 21: Consistency-first objective sweep (decoder-alignment regularization)
- Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE TMPDIR=/tmp MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_consistency_regularization_sweep.py --device cpu --k 4
```
- Outcome: success
- Run directory:
  - `results/experiments/consistency_objective_sweep/run_20260212T145529Z/`
- Sweep values:
  - `lambda in {0.0, 1e-4, 5e-4, 1e-3, 2e-3}`
- Selected lambda:
  - `0.002` (under EV-drop constraint)
- Key results:
  - baseline (`lambda=0`): delta `+0.02866`, EV `0.35892`
  - selected (`lambda=0.002`): delta `+0.02933`, EV `0.35897`
  - trained PWMCC improvement vs baseline: `+0.00067`
  - bootstrap 95% CI for improvement: `[-0.00246, +0.00376]` (not statistically resolved)

### Run 22: Follow-up synthesis report
- Artifact:
  - `HIGH_IMPACT_FOLLOWUPS_REPORT.md`
- Scope:
  - integrates Runs 19-21 with fair-control interpretation and literature-grounded framing.

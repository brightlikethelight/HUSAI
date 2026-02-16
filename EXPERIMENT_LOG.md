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

## 2026-02-12 - Official Benchmark Harness + Result-Consistency Audit

### Run 23: Result-consistency verification against artifact JSONs
- Command:
```bash
python scripts/analysis/verify_experiment_consistency.py
```
- Outcome: success
- Key outputs:
  - `results/analysis/experiment_consistency_report.json`
  - `results/analysis/experiment_consistency_report.md`
- Result summary:
  - overall pass = `True`
  - phase4a training signal: `delta=+0.002886`, `p=1.649e-04`
  - adaptive `k=4` vs control `k=32` trained-PWMCC gain: `+0.05701`
  - gain 95% CI: `[+0.05475, +0.05924]`
  - consistency-regularizer gain unresolved: CI spans zero `[-0.00242, +0.00377]`

### Run 24: Official SAEBench/CE-Bench harness preflight
- Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/run_official_external_benchmarks.py
```
- Outcome: success
- Run directory:
  - `results/experiments/phase4e_external_benchmark_official/run_20260212T151416Z/`
- Key outputs:
  - `preflight.json`
  - `local_sae_index.json`
  - `commands.json`
  - `summary.md`
  - `manifest.json`
- Result summary:
  - SAEBench module availability: `False`
  - CE-Bench module availability: `False`
  - local SAE checkpoints indexed: `5`
  - no official commands executed in this run (preflight-only by design)

### Run 25: Post-edit regression checks for new benchmark/audit scripts
- Commands:
```bash
python scripts/experiments/run_official_external_benchmarks.py --help
python scripts/analysis/verify_experiment_consistency.py --help
flake8 scripts/experiments/run_official_external_benchmarks.py \
  scripts/analysis/verify_experiment_consistency.py --max-line-length 130
KMP_DUPLICATE_LIB_OK=TRUE TMPDIR=/tmp MPLCONFIGDIR=/tmp/mpl pytest tests -q
```
- Outcome: success
- Result summary:
  - new script CLIs parse correctly
  - flake8 passes for the new scripts
  - test suite regression check: `83 passed`

## 2026-02-12 - RunPod B200 High-Impact Execution

### Run 26: Remote smoke gate on RunPod B200
- Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE TMPDIR=/tmp MPLCONFIGDIR=/tmp/mpl make smoke
```
- Outcome: success
- Key outputs:
  - `/tmp/husai_ci_smoke/transformer/transformer_final.pt`
  - `/tmp/husai_ci_smoke/acts.pt`
  - `/tmp/husai_ci_smoke/sae/sae_final.pt`
- Notes:
  - This run validated end-to-end pipeline integrity on remote GPU environment.

### Run 27: Transformer checkpoint generation for follow-up experiments
- Command:
```bash
python -m scripts.training.train_baseline \
  --config configs/examples/baseline_relu.yaml \
  --epochs 5000 \
  --batch-size 2048 \
  --no-wandb \
  --device cuda \
  --save-dir results/transformer_5000ep
```
- Outcome: partial (manually stopped after convergence)
- Key outputs:
  - `results/transformer_5000ep/transformer_best.pt`
- Notes:
  - Val accuracy reached 1.0 at epoch 3; run was intentionally terminated early after confirming stable best checkpoint.

### Run 28: Activation cache generation (layer 1 / answer position)
- Command:
```bash
python -m scripts.analysis.extract_activations \
  --model-path results/transformer_5000ep/transformer_best.pt \
  --layer 1 \
  --position answer \
  --batch-size 4096 \
  --device cuda \
  --output results/activations/layer1_answer.pt
```
- Outcome: success
- Key outputs:
  - `results/activations/layer1_answer.pt`
  - `results/activations/layer1_answer.meta.pt`

### Run 29: 5-seed TopK SAE training for Phase 4a
- Command pattern:
```bash
python -m scripts.training.train_sae \
  --transformer-checkpoint results/transformer_5000ep/transformer_best.pt \
  --config configs/sae/topk_8x_k32.yaml \
  --layer 1 \
  --seed <seed> \
  --epochs 20 \
  --batch-size 2048 \
  --use-cached-activations results/activations/layer1_answer.pt \
  --save-dir results/saes/topk_seed<seed> \
  --no-wandb \
  --device cuda \
  --quiet
```
- Seeds: `42, 123, 456, 789, 1011`
- Outcome: success
- Key outputs:
  - `results/saes/topk_seed42/sae_final.pt`
  - `results/saes/topk_seed123/sae_final.pt`
  - `results/saes/topk_seed456/sae_final.pt`
  - `results/saes/topk_seed789/sae_final.pt`
  - `results/saes/topk_seed1011/sae_final.pt`

### Run 30: Phase 4a trained-vs-random reproduction (5 seeds)
- Command:
```bash
python scripts/experiments/run_phase4a_reproduction.py \
  --sae-root results/saes \
  --trained-seeds 42,123,456,789,1011 \
  --random-seeds 1000,1001,1002,1003,1004 \
  --output-dir results/experiments/phase4a_trained_vs_random \
  --analysis-output results/analysis/trained_vs_random_pwmcc.json
```
- Outcome: success
- Key outputs:
  - `results/experiments/phase4a_trained_vs_random/results.json`
  - `results/experiments/phase4a_trained_vs_random/summary.md`
  - `results/experiments/phase4a_trained_vs_random/manifest.json`
  - `results/analysis/trained_vs_random_pwmcc.json`
- Result summary:
  - trained PWMCC mean = `0.300059`
  - random PWMCC mean = `0.298829`
  - delta = `+0.001230`
  - ratio = `1.0041`
  - one-sided Mann-Whitney p-value = `8.629e-03`
  - conclusion = `training_signal_present`

### Run 31: Phase 4c core ablations (GPU, 5 seeds)
- Command:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/experiments/run_core_ablations.py \
  --transformer-checkpoint results/transformer_5000ep/transformer_best.pt \
  --activations-cache results/activations/layer1_answer.pt \
  --device cuda \
  --epochs 20 \
  --batch-size 2048 \
  --seeds 42,123,456,789,1011 \
  --k-values 8,16,32,64 \
  --dsae-values 64,128,256,512 \
  --fixed-k 32 \
  --fixed-dsae 128 \
  --output-dir results/experiments/phase4c_core_ablations
```
- Outcome: success
- Run directory:
  - `results/experiments/phase4c_core_ablations/run_20260212T200711Z/`
- Result summary:
  - best `k` sweep condition (delta PWMCC): `k=8, d_sae=128`, delta `+0.009773`, ratio `1.0398`, EV `0.2282`
  - best `d_sae` sweep condition (delta PWMCC): `d_sae=64, k=32`, delta `+0.119986`, ratio `1.5272`, EV `0.2941`

### Run 32: External benchmark-aligned slice refresh
- Command:
```bash
python scripts/experiments/run_external_benchmark_slice.py \
  --phase4a-results results/experiments/phase4a_trained_vs_random/results.json \
  --core-ablations-root results/experiments/phase4c_core_ablations \
  --output-dir results/experiments/phase4e_external_benchmark_slice
```
- Outcome: success
- Key outputs:
  - `results/experiments/phase4e_external_benchmark_slice/benchmark_slice.json`
  - `results/experiments/phase4e_external_benchmark_slice/benchmark_slice.md`
  - `results/experiments/phase4e_external_benchmark_slice/manifest.json`

### Run 33: Official SAEBench harness execution (completed)
- Command:
```bash
python scripts/experiments/run_official_external_benchmarks.py \
  --skip-cebench \
  --execute \
  --saebench-command "python -m sae_bench.evals.sparse_probing_sae_probes.main \
    --model_name pythia-70m-deduped \
    --sae_regex_pattern '^pythia-70m-deduped-res-sm$' \
    --sae_block_pattern '^blocks.0.hook_resid_pre$' \
    --setting normal \
    --reg_type l1 \
    --ks 1 2 5 \
    --results_path /tmp/sae_bench_probe_results \
    --output_folder results/experiments/phase4e_external_benchmark_official/saebench_sparse_probing_sae_probes \
    --model_cache_path /tmp/sae_bench_model_cache"
```
- Outcome: success
- Harness run directory:
  - `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/`
- Command status:
  - `saebench`: attempted `True`, success `True`, return code `0`
- Key harness outputs:
  - `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/commands.json`
  - `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/preflight.json`
  - `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/summary.md`
  - `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/logs/saebench.stdout.log`
  - `results/experiments/phase4e_external_benchmark_official/run_20260212T201204Z/logs/saebench.stderr.log`
- External probe outputs:
  - `/tmp/sae_bench_probe_results/`
  - matched SAE-probes result files: `113`
  - matched baseline logreg files: `113`
- Aggregate SAE-vs-baseline deltas (best SAE over `k in {1,2,5}` per dataset):
  - mean `test_f1` delta: `-0.095237`
  - mean `test_acc` delta: `-0.051324`
  - mean `test_auc` delta: `-0.065125`
  - `test_auc` wins/losses/ties: `21 / 88 / 4`
- Additional diagnostics:
  - best-k distribution by dataset: `{5: 74, 2: 14, 1: 25}`
  - baseline mean `test_auc`: `0.674402`
- Interpretation:
  - Official SAEBench execution is operational and reproducible in this environment.
  - This run does not support external-performance claims for this benchmark setup.

### Run 34: Official benchmark harness reliability patch
- Change:
  - Updated `scripts/experiments/run_official_external_benchmarks.py` to stream subprocess stdout/stderr directly to log files instead of buffering full output in memory.
- Why:
  - Improves stability and observability for long-running benchmark commands.
- Validation:
```bash
python scripts/experiments/run_official_external_benchmarks.py --help
flake8 scripts/experiments/run_official_external_benchmarks.py --max-line-length 130
```
- Outcome: success

## 2026-02-13 - RunPod B200: Direct HUSAI Custom SAEBench Multi-Seed

### Run 35: HUSAI custom SAE training on SAEBench activation cache (seed 123)
- Command:
```bash
python scripts/experiments/train_husai_sae_on_cached_activations.py \
  --activation-cache-dir /tmp/sae_bench_model_cache/model_activations_pythia-70m-deduped \
  --activation-glob '*_blocks.0.hook_resid_pre.pt' \
  --max-files 80 \
  --max-rows-per-file 2048 \
  --max-total-rows 150000 \
  --d-sae 2048 \
  --k 32 \
  --epochs 10 \
  --batch-size 4096 \
  --learning-rate 0.001 \
  --seed 123 \
  --device cuda \
  --output-dir results/saes/husai_pythia70m_topk_seed123
```
- Outcome: success
- Key output:
  - `results/saes/husai_pythia70m_topk_seed123/summary.json`
- Result summary:
  - final loss `0.003721`
  - final EV `0.999433`

### Run 36: Official harness execution for HUSAI seed 123 custom checkpoint
- Command:
```bash
python scripts/experiments/run_official_external_benchmarks.py \
  --skip-saebench \
  --skip-cebench \
  --husai-saebench-checkpoint results/saes/husai_pythia70m_topk_seed123/sae_final.pt \
  --husai-saebench-release husai_pythia70m_topk_seed123 \
  --husai-saebench-model-name pythia-70m-deduped \
  --husai-saebench-hook-layer 0 \
  --husai-saebench-hook-name blocks.0.hook_resid_pre \
  --husai-saebench-ks 1,2,5 \
  --husai-saebench-results-path /tmp/husai_saebench_probe_results_seed123 \
  --husai-saebench-model-cache-path /tmp/sae_bench_model_cache \
  --husai-saebench-force-rerun \
  --execute
```
- Outcome: success
- Run directory:
  - `results/experiments/phase4e_external_benchmark_official/run_20260213T031247Z/`
- Result summary:
  - best `k` by AUC: `5`
  - best AUC: `0.622244`
  - baseline AUC: `0.674402`
  - delta AUC: `-0.052158`

### Run 37: HUSAI custom SAE training + official harness (seed 456)
- Training command:
```bash
python scripts/experiments/train_husai_sae_on_cached_activations.py \
  --activation-cache-dir /tmp/sae_bench_model_cache/model_activations_pythia-70m-deduped \
  --activation-glob '*_blocks.0.hook_resid_pre.pt' \
  --max-files 80 \
  --max-rows-per-file 2048 \
  --max-total-rows 150000 \
  --d-sae 2048 \
  --k 32 \
  --epochs 10 \
  --batch-size 4096 \
  --learning-rate 0.001 \
  --seed 456 \
  --device cuda \
  --output-dir results/saes/husai_pythia70m_topk_seed456
```
- Benchmark command:
```bash
python scripts/experiments/run_official_external_benchmarks.py \
  --skip-saebench \
  --skip-cebench \
  --husai-saebench-checkpoint results/saes/husai_pythia70m_topk_seed456/sae_final.pt \
  --husai-saebench-release husai_pythia70m_topk_seed456 \
  --husai-saebench-model-name pythia-70m-deduped \
  --husai-saebench-hook-layer 0 \
  --husai-saebench-hook-name blocks.0.hook_resid_pre \
  --husai-saebench-ks 1,2,5 \
  --husai-saebench-results-path /tmp/husai_saebench_probe_results_seed456 \
  --husai-saebench-model-cache-path /tmp/sae_bench_model_cache \
  --husai-saebench-force-rerun \
  --execute
```
- Outcome: success
- Run directory:
  - `results/experiments/phase4e_external_benchmark_official/run_20260213T032116Z/`
- Result summary:
  - best `k` by AUC: `5`
  - best AUC: `0.622249`
  - baseline AUC: `0.674402`
  - delta AUC: `-0.052153`

### Run 38: Multi-seed aggregate synthesis for HUSAI custom SAEBench
- Inputs:
  - `run_20260213T024329Z` (seed 42)
  - `run_20260213T031247Z` (seed 123)
  - `run_20260213T032116Z` (seed 456)
- Generated artifacts:
  - remote: `results/experiments/phase4e_external_benchmark_official/husai_custom_multiseed/summary.json`
  - remote: `results/experiments/phase4e_external_benchmark_official/husai_custom_multiseed/summary.md`
  - tracked copy: `docs/evidence/phase4e_husai_custom_multiseed/summary.json`
  - tracked copy: `docs/evidence/phase4e_husai_custom_multiseed/summary.md`
- Aggregate result summary:
  - best AUC mean ± std: `0.622601 ± 0.000615`
  - best AUC 95% CI: `[0.621905, 0.623297]`
  - delta AUC vs baseline mean ± std: `-0.051801 ± 0.000615`
  - delta AUC vs baseline 95% CI: `[-0.052496, -0.051105]`
- Interpretation:
  - HUSAI custom external performance is stable across seeds but consistently below baseline.

## 2026-02-13 - CE-Bench Compatibility Closure and Evidence Capture

### Run 39: CE-Bench compatibility rerun surfaced `Stopwatch.stop` API drift
- Command:
```bash
python scripts/experiments/run_official_external_benchmarks.py \
  --skip-saebench \
  --cebench-repo /workspace/CE-Bench \
  --cebench-use-compat-runner \
  --cebench-sae-regex-pattern pythia-70m-deduped-res-sm \
  --cebench-sae-block-pattern blocks.0.hook_resid_pre \
  --cebench-artifacts-path /tmp/ce_bench_artifacts \
  --cebench-force-rerun \
  --execute
```
- Outcome: failed
- Run directory:
  - `results/experiments/phase4e_external_benchmark_official/run_20260213T051046Z/`
- Failure signature:
  - `AttributeError: 'CompatStopwatch' object has no attribute 'stop'`
- Root cause:
  - New `stw` API removed both `start=` and `stop()` methods expected by CE-Bench.

### Run 40: CE-Bench official compatibility run after shim fix
- Command:
```bash
python scripts/experiments/run_official_external_benchmarks.py \
  --skip-saebench \
  --cebench-repo /workspace/CE-Bench \
  --cebench-use-compat-runner \
  --cebench-sae-regex-pattern pythia-70m-deduped-res-sm \
  --cebench-sae-block-pattern blocks.0.hook_resid_pre \
  --cebench-artifacts-path /tmp/ce_bench_artifacts \
  --cebench-force-rerun \
  --execute
```
- Outcome: success
- Run directory:
  - `results/experiments/phase4e_external_benchmark_official/run_20260213T103218Z/`
- Command status:
  - `cebench`: attempted `True`, success `True`, return code `0`
- Key outputs:
  - `summary.md`
  - `commands.json`
  - `logs/cebench.stdout.log`
  - `logs/cebench.stderr.log`
- CE-Bench metric snapshot:
  - `total_rows`: `5000`
  - `contrastive_score_mean.max`: `49.11421939086914`
  - `independent_score_mean.max`: `53.69819771347046`
  - `interpretability_score_mean.max`: `47.481240758132934`
  - SAE target: `pythia-70m-deduped-res-sm / blocks.0.hook_resid_pre`

### Run 41: CE-Bench artifact capture + deterministic output hardening
- Changes:
  - Added CE-Bench metric summary emission in compat runner:
    - `scripts/experiments/run_cebench_compat.py`
  - Added CE-Bench summary block in official harness markdown:
    - `scripts/experiments/run_official_external_benchmarks.py`
  - Added deterministic cleanup of run-local CE-Bench relative outputs (`scores_dump.txt`, `interpretability_eval`) before execution:
    - `scripts/experiments/run_cebench_compat.py`
- Evidence capture:
  - `docs/evidence/phase4e_cebench_official/run_20260213T103218Z_harness_summary.md`
  - `docs/evidence/phase4e_cebench_official/run_20260213T103218Z_commands.json`
  - `docs/evidence/phase4e_cebench_official/run_20260213T103218Z_preflight.json`
  - `docs/evidence/phase4e_cebench_official/run_20260213T103218Z_cebench_metrics_summary.json`
  - `docs/evidence/phase4e_cebench_official/run_20260213T103218Z_cebench_results.json`
- Notes:
  - Captured `scores_dump_line_count=10000` reflects legacy append behavior from earlier CE-Bench runs.
  - Deterministic cleanup fix is now in place for subsequent runs.

## 2026-02-13 - High-Impact Cycle 2 Execution

### Run 20: Official CE-Bench + direct HUSAI custom CE-Bench (matched baseline)
- Command (remote B200):
```bash
env KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_official_external_benchmarks.py \
  --skip-saebench \
  --cebench-repo /workspace/CE-Bench \
  --cebench-use-compat-runner \
  --cebench-sae-regex-pattern pythia-70m-deduped-res-sm \
  --cebench-sae-block-pattern blocks.0.hook_resid_pre \
  --cebench-artifacts-path /tmp/ce_bench_artifacts \
  --cebench-force-rerun \
  --husai-cebench-checkpoint results/saes/husai_pythia70m_topk_seed42/sae_final.pt \
  --husai-cebench-release husai_pythia70m_topk_seed42 \
  --husai-cebench-model-name pythia-70m-deduped \
  --husai-cebench-hook-layer 0 \
  --husai-cebench-hook-name blocks.0.hook_resid_pre \
  --husai-cebench-match-baseline \
  --execute
```
- Launch log:
  - `results/experiments/phase4e_external_benchmark_official/launch_husai_cebench_latest.log`
- Run directory:
  - `results/experiments/phase4e_external_benchmark_official/run_20260213T152344Z/`
- Outcome:
  - in progress at time of this log append

### Run 21: Direct HUSAI custom CE-Bench adapter pilot (remote B200, 500 rows)
- Command:
```bash
env KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_husai_cebench_custom_eval.py \
  --cebench-repo /workspace/CE-Bench \
  --checkpoint results/saes/husai_pythia70m_topk_seed42/sae_final.pt \
  --architecture topk \
  --sae-release husai_pythia70m_topk_seed42 \
  --model-name pythia-70m-deduped \
  --hook-layer 0 \
  --hook-name blocks.0.hook_resid_pre \
  --device cuda \
  --sae-dtype float32 \
  --max-rows 500 \
  --output-folder results/experiments/phase4e_external_benchmark_official/husai_custom_cebench_pilot_20260213 \
  --artifacts-path /tmp/ce_bench_artifacts \
  --matched-baseline-summary results/experiments/phase4e_external_benchmark_official/run_20260213T103218Z/cebench/cebench_metrics_summary.json
```
- Outcome: success
- Key outputs:
  - `results/experiments/phase4e_external_benchmark_official/husai_custom_cebench_pilot_20260213/husai_custom_cebench_summary.json`
  - `results/experiments/phase4e_external_benchmark_official/husai_custom_cebench_pilot_20260213/cebench_metrics_summary.json`
- Result summary:
  - custom contrastive/independent/interpretability max: `10.736 / 11.450 / 10.734`
  - delta vs official CE-Bench baseline summary: `-38.379 / -42.248 / -36.747`
  - note: baseline artifact uses 5000 rows while this pilot uses 500 rows.

### Run 22: Architecture frontier pilot launch (remote B200)
- Command:
```bash
env KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_architecture_frontier_external.py \
  ... \
  --cebench-max-rows 200
```
- Outcome: failed fast (argument parser)
- Failure point:
  - `unrecognized arguments: --cebench-max-rows 200`
- Resolution:
  - Added missing parser arg in `scripts/experiments/run_architecture_frontier_external.py`.

## 2026-02-13 - Cycle 2 Closure: Matched CE-Bench, Frontier, Scaling, Gates

### Run 42: Matched-200 CE-Bench baseline closure + local evidence capture
- Baseline artifact generated from completed remote outputs:
  - `docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json`
  - `docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.md`
- Baseline metric snapshot (`total_rows=200`):
  - contrastive max: `50.51130743026734`
  - independent max: `50.99926634788513`
  - interpretability max: `47.951611585617066`

### Run 43: Architecture frontier reliability fixes and validated rerun
- Key code fixes landed:
  - BatchTopK threshold calibration/train-inference handling,
  - non-degenerate custom SAE init,
  - explicit SAEBench dataset passing,
  - path normalization,
  - SAEBench dataset controls.
- Commits:
  - `1e3d94e`, `38a00db`, `bed457c`, `dda27d8`, `b5aca3a`
- Final run artifact:
  - `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_results.json`

### Run 44: Frontier matched-baseline delta synthesis
- Generated artifacts:
  - `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_cebench_deltas_vs_matched200.md`
  - `docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_cebench_deltas_vs_matched200.json`
- CE-Bench interpretability deltas vs matched baseline:
  - topk: `-40.3662`
  - relu: `-43.7235`
  - batchtopk: `-41.4848`
  - jumprelu: `-43.6006`

### Run 45: External scaling study hardening + full 8-condition execution
- Scaling runner hardening commit:
  - `884a039` (`run_external_metric_scaling_study.py` dataset fail-fast + explicit dataset passing)
- Remote preparatory cache generation:
  - layer-1 SAEBench cache generation completed for 16 matched datasets
  - artifact: `results/experiments/cachegen_layer1_husai_saebench/husai_custom_sae_summary.json`
- Full scaling run completed:
  - `results/experiments/phase4e_external_scaling_study/run_20260213T203923Z/`
  - tracked copies:
    - `docs/evidence/phase4e_external_scaling_study/run_20260213T203923Z_results.json`
    - `docs/evidence/phase4e_external_scaling_study/run_20260213T203923Z_summary.md`
    - `docs/evidence/phase4e_external_scaling_study/run_20260213T203923Z_summary_table.md`
- Aggregate findings:
  - by token budget:
    - `10000`: SAEBench delta mean `-0.07854`, CE-Bench interpretability mean `8.0104`
    - `30000`: SAEBench delta mean `-0.08497`, CE-Bench interpretability mean `8.1203`
  - by hook layer:
    - layer `0`: SAEBench delta mean `-0.06873`, CE-Bench interpretability mean `6.8769`
    - layer `1`: SAEBench delta mean `-0.09479`, CE-Bench interpretability mean `9.2538`
  - by d_sae:
    - `1024`: SAEBench delta mean `-0.07996`, CE-Bench interpretability mean `7.1746`
    - `2048`: SAEBench delta mean `-0.08355`, CE-Bench interpretability mean `8.9561`

### Run 46: Assignment-aware consistency v2 with external acceptance
- Command:
```bash
python scripts/experiments/run_assignment_consistency_v2.py \
  --external-summary docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_topk_saebench_summary.json \
  --min-delta-pwmcc 0.0 \
  --min-delta-lcb 0.0 \
  --max-ev-drop 0.05 \
  --min-external-delta 0.0
```
- Artifact:
  - `docs/evidence/phase4d_assignment_consistency_v2/run_20260213T203957Z_results.json`
- Result summary:
  - best lambda: `0.2`
  - delta PWMCC: `+0.070804`
  - conservative delta LCB: `+0.054419`
  - EV drop: `0.000878`
  - external delta: `-0.132836`
  - `pass_all`: `False` (external gate fail)

### Run 47: Stress-gated release policy strict gating + fail-fast support
- Baseline strict-eval command:
```bash
python scripts/experiments/run_stress_gated_release_policy.py \
  --phase4a-results results/experiments/phase4a_trained_vs_random/results.json \
  --external-summary docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_topk_saebench_summary.json \
  --require-transcoder --require-ood --require-external --min-external-delta 0.0
```
- Strict fail-fast command:
```bash
python scripts/experiments/run_stress_gated_release_policy.py \
  --phase4a-results results/experiments/phase4a_trained_vs_random/results.json \
  --external-summary docs/evidence/phase4b_architecture_frontier_external/run_20260213T173707Z_topk_saebench_summary.json \
  --require-transcoder --require-ood --require-external --min-external-delta 0.0 \
  --fail-on-gate-fail
```
- Artifacts:
  - `docs/evidence/phase4e_stress_gated_release/run_20260213T204120Z_release_policy.json`
  - strict run exit verification: `EXIT_CODE=2`
- Gate status:
  - random_model: pass
  - transcoder: fail (missing)
  - ood: fail (missing)
  - external: fail (negative delta)
  - pass_all: fail

### Run 48: CI-enforceable gate flags landed
- Commit:
  - `e2f5e8e`
- Changes:
  - `scripts/experiments/run_assignment_consistency_v2.py`:
    - new `--fail-on-acceptance-fail`
  - `scripts/experiments/run_stress_gated_release_policy.py`:
    - new `--fail-on-gate-fail`
- Validation:
  - `python -m py_compile ...` passed
  - `pytest -q` passed (`83 passed`)

## 2026-02-14 - Stress Runner Implementation Validation

### Run 49: Transcoder stress runner smoke validation
- Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE TMPDIR=/tmp MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_transcoder_stress_eval.py \
  --device cpu \
  --epochs 1 \
  --seeds 42,123 \
  --batch-size 256 \
  --max-samples 512 \
  --output-dir /tmp/husai_transcoder_smoke
```
- Outcome: success
- Key outputs:
  - `/tmp/husai_transcoder_smoke/run_20260214T041827Z/transcoder_stress_summary.json`
  - `/tmp/husai_transcoder_smoke/run_20260214T041827Z/transcoder_stress_summary.md`
- Notes:
  - Validates new `run_transcoder_stress_eval.py` execution path and artifact schema.
  - Fixed path portability issue for non-repo output directories in the same cycle.

### Run 50: OOD stress runner interface validation
- Command:
```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/experiments/run_ood_stress_eval.py --help
```
- Outcome: success
- Notes:
  - CLI parser and argument surface validated.
  - Full end-to-end execution deferred in this local workspace due missing SAEBench activation cache.

## 2026-02-14 - High-Impact External Baseline Calibration

### Run 51: CE-Bench adapter check with matched public baseline
- Remote command (B200):
```bash
python scripts/experiments/run_husai_cebench_custom_eval.py \
  --cebench-repo /workspace/CE-Bench \
  --checkpoint results/saes/husai_pythia70m_topk_seed42/sae_final.pt \
  --architecture topk \
  --sae-release husai_topk_seed42_adaptercheck \
  --model-name pythia-70m-deduped \
  --hook-layer 0 \
  --hook-name blocks.0.hook_resid_pre \
  --device cuda \
  --sae-dtype float32 \
  --max-rows 200 \
  --output-folder results/experiments/high_impact_adapter_check/run_20260214T202232Z \
  --artifacts-path /tmp/ce_bench_artifacts_adapter_parity \
  --matched-baseline-summary docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json
```
- Outcome: success
- Key outputs:
  - `results/experiments/high_impact_adapter_check/run_20260214T202232Z/husai_custom_cebench_summary.json`
  - `results/experiments/high_impact_adapter_check/run_20260214T202232Z/cebench_metrics_summary.json`
  - tracked copy: `docs/evidence/high_impact_adapter_check/run_20260214T202232Z_husai_custom_cebench_summary.json`
- Result summary:
  - CE-Bench interpretability max: `10.8933`
  - Delta vs matched public baseline interpretability max: `-37.0583`
  - Matched-baseline delta fields are now populated and reproducible in the adapter summary schema.

### Run 52: Matched-budget architecture frontier external (multi-seed, matched-baseline enabled)
- Remote launch command (B200):
```bash
python scripts/experiments/run_architecture_frontier_external.py \
  --activation-cache-dir /tmp/sae_bench_model_cache/model_activations_pythia-70m-deduped \
  --activation-glob '*_blocks.0.hook_resid_pre.pt' \
  --architectures topk,relu,batchtopk,jumprelu \
  --seeds 42,123,456,789,1011 \
  --d-sae 1024 \
  --k 32 \
  --epochs 6 \
  --batch-size 4096 \
  --learning-rate 0.001 \
  --device cuda \
  --run-saebench --run-cebench \
  --cebench-repo /workspace/CE-Bench \
  --cebench-max-rows 200 \
  --cebench-matched-baseline-summary docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json \
  --saebench-results-path /tmp/husai_saebench_probe_results_frontier_multiseed \
  --saebench-model-cache-path /tmp/sae_bench_model_cache \
  --saebench-dataset-limit 8 \
  --cebench-artifacts-path /tmp/ce_bench_artifacts_frontier_multiseed \
  --output-dir results/experiments/phase4b_architecture_frontier_external_multiseed
```
- Launch log:
  - `results/experiments/phase4b_architecture_frontier_external_multiseed/launch_frontier_multiseed_20260214T202700Z.log`
- Current status:
  - in progress on B200 (run id: `run_20260214T202538Z`)
  - first condition (`topk_seed42`) completed through CE-Bench with matched-baseline delta capture
  - progress counters at `2026-02-14T20:30:05Z`:
    - checkpoints: `2`
    - SAEBench summaries: `2`
    - CE-Bench summaries: `1`

## 2026-02-14 - Reflective Integrity Update and Ongoing Frontier Multiseed

### Run 53: Consistency-audit refresh (assignment-v2 + stress gates included)
- Command:
```bash
python scripts/analysis/verify_experiment_consistency.py
```
- Outcome: success
- Key outputs:
  - `results/analysis/experiment_consistency_report.json`
  - `results/analysis/experiment_consistency_report.md`
- Result summary:
  - `overall_pass=False`
  - failing checks:
    - `assignment_v2_external_gate_pass=False`
    - `stress_release_policy_pass_all=False`
- Notes:
  - This prevents false-green status from legacy-only consistency checks.

### Run 54: Architecture frontier multiseed external (B200, in progress)
- Remote run:
  - `results/experiments/phase4b_architecture_frontier_external_multiseed/run_20260214T202538Z/`
- Current monitored snapshot (`~20:59 UTC`):
  - checkpoints: `13`
  - SAEBench summaries: `13`
  - CE-Bench summaries: `12`
- Notes:
  - run continues through `topk,relu,batchtopk,jumprelu` x seeds `42,123,456,789,1011`.

### Run 55: Remote high-impact queue orchestration launch (B200)
- Remote launcher command (from local controller):
```bash
nohup bash scripts/experiments/run_b200_high_impact_queue.sh > results/experiments/cycle3_queue/queue_launcher_<timestamp>.log 2>&1 &
```
- Queue script path:
  - `scripts/experiments/run_b200_high_impact_queue.sh`
- Queue behavior:
  1. wait for active frontier multiseed run completion,
  2. run multiseed scaling study,
  3. select best frontier external candidate,
  4. run transcoder and OOD stress,
  5. run strict release gate.
- Launch status:
  - active on remote (`PID 43530` at launch check)
  - queue log: `results/experiments/cycle3_queue/queue_launcher_20260214T210734Z.log`

### Run 56: Frontier multiseed interim metric snapshot
- Snapshot artifact:
  - `docs/evidence/phase4b_architecture_frontier_external_multiseed/run_20260214T202538Z_interim_snapshot_20260214T211200Z.md`
- Snapshot counts (UTC ~21:12):
  - checkpoints `15`, SAEBench summaries `15`, CE-Bench summaries `14`
- Partial means:
  - `topk`: SAEBench delta `-0.04059`, CE-Bench interp `7.72677`
  - `relu`: SAEBench delta `-0.02469`, CE-Bench interp `4.25769`
  - `batchtopk`: SAEBench delta `-0.04336`, CE-Bench interp `6.58815`

### Run 57: Cycle 3 queue completion synthesis (final tables + gate outcome)
- Synthesized artifact set copied from remote B200:
  - `docs/evidence/cycle3_queue_final/queue_manifest_run_20260214T210734Z.json`
  - `docs/evidence/cycle3_queue_final/frontier_multiseed_results_run_20260214T202538Z.json`
  - `docs/evidence/cycle3_queue_final/scaling_multiseed_results_run_20260214T212435Z.json`
  - `docs/evidence/cycle3_queue_final/transcoder_stress_summary_run_20260214T224242Z.json`
  - `docs/evidence/cycle3_queue_final/ood_stress_summary_run_20260214T224309Z.json`
  - `docs/evidence/cycle3_queue_final/release_policy_run_20260214T225029Z.json`
- Final synthesis markdown:
  - `docs/evidence/cycle3_queue_final/cycle3_final_synthesis_run_20260214T210734Z.md`
- Outcome summary:
  - queue status: complete
  - strict release gate exit: `2`
  - release `pass_all=False` (transcoder/external gate fails)
  - OOD gate: pass
  - W&B: no active logging artifacts for this queue cycle (file-based logging only)

## 2026-02-15 - Release Policy Refactor Validation

### Run 58: Multi-objective candidate selector + joint release-gate smoke
- Local validation commands:
```bash
python scripts/experiments/select_release_candidate.py \
  --frontier-results docs/evidence/cycle3_queue_final/frontier_multiseed_results_run_20260214T202538Z.json \
  --scaling-results docs/evidence/cycle3_queue_final/scaling_multiseed_results_run_20260214T212435Z.json \
  --require-both-external \
  --output-dir results/experiments/release_candidate_selection_smoke

python scripts/experiments/run_stress_gated_release_policy.py \
  --phase4a-results results/experiments/phase4a_trained_vs_random/results.json \
  --transcoder-results docs/evidence/cycle3_queue_final/transcoder_stress_summary_run_20260214T224242Z.json \
  --ood-results docs/evidence/cycle3_queue_final/ood_stress_summary_run_20260214T224309Z.json \
  --external-candidate-json results/experiments/release_candidate_selection_smoke/run_20260215T052937Z/selected_candidate.json \
  --external-mode joint \
  --min-saebench-delta 0.0 \
  --min-cebench-delta 0.0 \
  --require-transcoder --require-ood --require-external \
  --fail-on-gate-fail \
  --output-dir results/experiments/release_stress_gates_smoke
```
- Outcome: expected gate failure (`exit code 2`) with explicit joint external diagnostics.
- Key outputs:
  - `results/experiments/release_candidate_selection_smoke/run_20260215T052937Z/selected_candidate.json`
  - `results/experiments/release_stress_gates_smoke/run_20260215T052953Z/release_policy.json`
- Result summary:
  - selector chosen candidate: `topk_seed123` (frontier)
  - selected metrics: SAEBench delta `-0.032604`, CE-Bench delta `-39.910494`
  - joint release gate fields populated from candidate JSON (`saebench_delta`, `cebench_interp_delta_vs_baseline`)
  - strict gate remained false due transcoder and external thresholds.
- Additional validation:
  - `pytest -q tests/unit/test_release_policy_selector.py` -> `2 passed`
  - `bash -n scripts/experiments/run_b200_high_impact_queue.sh` -> pass
  - `python -m py_compile scripts/experiments/select_release_candidate.py scripts/experiments/run_stress_gated_release_policy.py` -> pass

### Run 59: Determinism fix + layer-aware CE-Bench baseline mapping (pre-cycle rerun)
- Local code changes:
  - `scripts/experiments/run_transcoder_stress_eval.py`
    - seed before model construction (transcoder + SAE) to make initialization deterministic per seed.
    - set/log `CUBLAS_WORKSPACE_CONFIG` for stronger CUDA determinism.
  - `scripts/experiments/run_external_metric_scaling_study.py`
    - add `--cebench-matched-baseline-map` for hook-aware CE-Bench matched baselines.
    - per-condition baseline selection priority: `hook_name` -> `hook_layer` -> `default`.
    - record resolved baseline path per condition in `records[*].cebench_matched_baseline_summary`.
  - `scripts/experiments/run_b200_high_impact_queue.sh`
    - queue now accepts baseline map via `CEBENCH_BASELINE_MAP` and forwards it to scaling.
    - queue manifest now records baseline default/map paths.
  - added baseline map artifact scaffold:
    - `docs/evidence/phase4e_cebench_matched200/cebench_baseline_map.json`
  - added unit tests:
    - `tests/unit/test_external_scaling_baseline_map.py`

- Validation commands:
```bash
python -m py_compile scripts/experiments/run_transcoder_stress_eval.py scripts/experiments/run_external_metric_scaling_study.py
pytest -q tests/unit/test_external_scaling_baseline_map.py tests/unit/test_release_policy_selector.py
pytest -q tests/unit
bash -n scripts/experiments/run_b200_high_impact_queue.sh
make smoke
```

- Outcome: success
  - `tests/unit`: `80 passed`
  - smoke pipeline: pass
  - queue shell syntax: pass

- Scientific reason for change:
  - removes a real reproducibility bug in transcoder stress (seed not controlling init).
  - removes layer-mismatch risk for CE-Bench deltas in scaling studies, enabling fairer external comparisons before the next B200 cycle.

### Run 60: CE-Bench layer-baseline integrity check (official SAE availability)
- Remote probe command (B200):
```bash
python scripts/experiments/run_cebench_compat.py \
  --cebench-repo /workspace/CE-Bench \
  --sae-regex-pattern pythia-70m-deduped-res-sm \
  --sae-block-pattern blocks.1.hook_resid_pre \
  --output-folder results/experiments/phase4e_external_benchmark_official/cebench_matched200_layer1 \
  --artifacts-path /tmp/ce_bench_artifacts_matched200_layer1 \
  --max-rows 200 \
  --force-rerun
```
- Outcome: expected failure (`AssertionError: No SAEs selected`).
- Evidence:
  - `results/experiments/phase4e_external_benchmark_official/cebench_matched200_layer1_launch.log`
  - `results/experiments/phase4e_external_benchmark_official/cebench_matched200_layer1/cebench_metrics_summary.json`
- Follow-up integrity check:
  - enumerated available `pythia-70m-deduped-res-sm` SAE IDs from SAE-Lens directory.
  - confirmed `blocks.1.hook_resid_pre` is not available (only `blocks.0.hook_resid_pre` plus `resid_post` layers).
- Code change from this finding:
  - `cebench_baseline_map.json` now sets layer-1 pre hook keys to `null`.
  - scaling harness treats `null` as explicit “no matched baseline” (delta disabled) instead of incorrectly using layer-0 baseline.

### Run 61: Cycle-4 implementation pass (uncertainty-aware gates + new experiment harnesses)
- Local code changes:
  - `scripts/experiments/select_release_candidate.py`
    - added grouped-across-seeds selection mode (`--group-by-condition`).
    - added uncertainty mode (`--uncertainty-mode point|lcb`) and per-group CI metrics (`*_ci95_low/high`).
  - `scripts/experiments/run_stress_gated_release_policy.py`
    - added CI-aware external gating:
      - `--use-external-lcb`
      - `--min-saebench-delta-lcb`
      - `--min-cebench-delta-lcb`
    - records gate basis (`point` vs `lcb`) and gated values used.
  - `scripts/experiments/run_b200_high_impact_queue.sh`
    - added selector/gate configuration for uncertainty-aware mode via env vars:
      - `SELECTOR_GROUP_BY_CONDITION`, `SELECTOR_UNCERTAINTY_MODE`, `MIN_SEEDS_PER_GROUP`
      - `USE_EXTERNAL_LCB_GATES`, `MIN_SAEBENCH_DELTA_LCB`, `MIN_CEBENCH_DELTA_LCB`
  - `scripts/experiments/husai_custom_sae_adapter.py`
    - added `matryoshka` architecture alias support and TopK eval compatibility mapping.
  - new experiment scripts:
    - `scripts/experiments/run_transcoder_stress_sweep.py`
    - `scripts/experiments/run_assignment_consistency_v3.py`
    - `scripts/experiments/run_matryoshka_frontier_external.py`
    - `scripts/experiments/run_known_circuit_recovery_closure.py`
  - new/updated tests:
    - `tests/unit/test_release_policy_selector.py` (grouped LCB selector + LCB gate behavior)
    - `tests/unit/test_husai_custom_sae_adapter.py`

- Validation commands:
```bash
pytest -q tests/unit/test_release_policy_selector.py tests/unit/test_husai_custom_sae_adapter.py
pytest -q tests/unit
python -m py_compile scripts/experiments/select_release_candidate.py scripts/experiments/run_stress_gated_release_policy.py scripts/experiments/husai_custom_sae_adapter.py scripts/experiments/run_transcoder_stress_sweep.py scripts/experiments/run_assignment_consistency_v3.py scripts/experiments/run_matryoshka_frontier_external.py scripts/experiments/run_known_circuit_recovery_closure.py
bash -n scripts/experiments/run_b200_high_impact_queue.sh
make smoke
```
- Outcome: success
  - unit tests: `84 passed`
  - smoke: pass
  - shell syntax + py compile: pass

### Run 62: Live B200 monitoring checkpoint (cycle queue still active)
- Remote monitoring command summary:
```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@198.13.252.68 -p 34426 -i ~/.ssh/runpod_key '<status/log checks>'
```
- Observed status:
  - active queue: `results/experiments/cycle3_queue/run_20260215T153651Z`
  - active stage: scaling multiseed (`run_external_metric_scaling_study.py`)
  - progress snapshot:
    - checkpoints: `22`
    - SAEBench summaries: `22`
    - CE-Bench summaries: `21`
    - pending condition: `tok30000_layer1_dsae2048_seed42`
  - sample CE-Bench log shows healthy completion (200/200 rows, no runtime error).
- W&B status:
  - `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE` not set on remote process env snapshot.
  - no active W&B run artifacts observed for this queue stage (file-based logs/artifacts only).

### Run 63: Null-baseline aggregation crash fix (scaling/frontier)
- Trigger:
  - Remote queue `results/experiments/cycle3_queue/run_20260215T153651Z` crashed after completing all condition evals.
  - Exception in scaling aggregation:
    - `AttributeError: 'NoneType' object has no attribute 'get'`
    - Source: `scripts/experiments/run_external_metric_scaling_study.py` when `delta_vs_matched_baseline` is explicitly `null`.

- Root cause:
  - Nested `.get(...).get(...)` chains assumed intermediate dicts.
  - With explicit baseline disable (`null`), intermediate value is `None`.

- Code fix:
  - `scripts/experiments/run_external_metric_scaling_study.py`
    - Added defensive nested metric extraction for aggregation metrics.
  - `scripts/experiments/run_architecture_frontier_external.py`
    - Applied the same defensive nested metric extraction pattern to avoid mirrored failure mode.

- Validation:
```bash
python -m py_compile scripts/experiments/run_external_metric_scaling_study.py scripts/experiments/run_architecture_frontier_external.py
pytest -q tests/unit/test_external_scaling_baseline_map.py tests/unit/test_release_policy_selector.py
pytest -q tests/unit
```
- Outcome: success (`84 passed` unit tests).
- Next action:
  - Pull fixes on B200 and rerun failed stage/queue to regenerate complete selector+gate outputs.

### Run 64: Known-circuit closure run on B200 (trained vs random Fourier checks)
- Remote command:
```bash
python scripts/experiments/run_known_circuit_recovery_closure.py \
  --transformer-checkpoint results/transformer_5000ep/transformer_best.pt \
  --sae-checkpoint-glob 'results/experiments/phase4d_assignment_consistency_v2/run_20260213T203957Z/checkpoints/lambda_0.2/sae_seed*.pt' \
  --output-dir results/experiments/known_circuit_recovery_closure
```
- Outcome: success (closure artifact produced)
- Key outputs:
  - `docs/evidence/known_circuit_recovery_closure/run_20260215T165907Z_closure_summary.json`
  - `docs/evidence/known_circuit_recovery_closure/run_20260215T165907Z_closure_summary.md`
- Result summary:
  - transformer Fourier R²: `0.020268`
  - random-transformer mean R²: `0.020717`
  - transformer delta vs random mean: `-0.000449`
  - transformer delta vs random LCB: `-0.000881`
  - SAE checkpoint matches on remote for requested glob: `0` (path absent on pod), so SAE overlap gate unresolved/fail in this run.
  - overall `pass_all=False`

### Run 65: B200 queue relaunch with uncertainty-aware selector + LCB gate settings
- Relaunch configuration:
```bash
MIN_SAEBENCH_DELTA=0.0 MIN_CEBENCH_DELTA=0.0 \
MIN_SAEBENCH_DELTA_LCB=0.0 MIN_CEBENCH_DELTA_LCB=0.0 \
USE_EXTERNAL_LCB_GATES=1 \
SELECTOR_GROUP_BY_CONDITION=1 SELECTOR_UNCERTAINTY_MODE=lcb MIN_SEEDS_PER_GROUP=3 \
CEBENCH_BASELINE_MAP=docs/evidence/phase4e_cebench_matched200/cebench_baseline_map.json \
bash scripts/experiments/run_b200_high_impact_queue.sh
```
- Active queue run:
  - `results/experiments/cycle3_queue/run_20260215T165724Z`
- Monitoring snapshot (in progress):
  - queue stage: scaling multiseed
  - latest scaling run: `results/experiments/phase4e_external_scaling_study_multiseed/run_20260215T165725Z`

### Run 66: Automatic post-queue follow-up chain scheduled on B200
- Script deployed and launched:
  - `scripts/experiments/run_cycle4_followups_after_queue.sh`
  - launcher log: `results/experiments/cycle4_followups/launcher_20260215T170149Z.log`
- Behavior:
  1. wait for active queue completion,
  2. pull latest `main`,
  3. run `run_transcoder_stress_sweep.py` (pilot grid),
  4. run `run_matryoshka_frontier_external.py` (pilot),
  5. run `run_assignment_consistency_v3.py` (internal sweep).
- Current status:
  - waiting loop confirmed in launcher log.

### Run 67: SAEBench cache-contamination hardening + live queue quality audit
- Why:
  - Active scaling queue logs repeatedly showed `Skipping dataset ...` in SAEBench probes.
  - This can be benign cache reuse, but it is a reproducibility risk when release IDs are reused across reruns.

- Code changes:
  - `scripts/experiments/run_external_metric_scaling_study.py`
    - changed SAE release ids from `husai_scaling_{cond_id}` to `husai_scaling_{run_id}_{cond_id}`
      for both SAEBench and CE-Bench custom eval invocations.
  - `scripts/experiments/run_architecture_frontier_external.py`
    - changed SAE release ids from `husai_{condition_id}` to `husai_{run_id}_{condition_id}`
      for both SAEBench and CE-Bench custom eval invocations.

- Validation:
```bash
python -m py_compile scripts/experiments/run_external_metric_scaling_study.py scripts/experiments/run_architecture_frontier_external.py
pytest -q tests/unit/test_release_policy_selector.py tests/unit/test_husai_custom_sae_adapter.py
make smoke
```
- Outcome: success
  - compile: pass
  - targeted unit tests: `6 passed`
  - smoke pipeline: pass

- Live B200 audit snapshot:
  - queue process still active: `run_b200_high_impact_queue.sh` -> scaling stage
  - latest scaling run: `results/experiments/phase4e_external_scaling_study_multiseed/run_20260215T165725Z`
  - completed artifacts observed for early conditions (train + SAEBench + CE-Bench summaries)
  - W&B env still unset on remote (`WANDB_API_KEY/PROJECT/ENTITY/MODE`), so current run remains file-artifact logged only.

### Run 68: Cycle-4 automation hardening for requested five-step program
- Goal:
  - Execute the post-queue high-impact program with stricter default selection/gating and cache-safe external reruns.

- Code changes:
  - `scripts/experiments/select_release_candidate.py`
    - default selection mode changed to grouped condition-level selection.
    - default uncertainty mode changed to `lcb`.
    - default `min_seeds_per_group` changed to `3`.
    - added explicit opt-out flag `--seed-level-selection`.
  - `scripts/experiments/run_transcoder_stress_sweep.py`
    - added `--fail-on-gate-fail` and non-zero exit when CI-LCB gate fails.
  - `scripts/experiments/run_matryoshka_frontier_external.py`
    - made SAE release IDs run-unique (`husai_{run_id}_{condition_id}`) for cache-safe external reruns.
  - `scripts/experiments/run_assignment_consistency_v3.py`
    - made external eval SAE release IDs run-unique (`husai_assignv3_{run_dir.name}_lambda...`).
  - `scripts/experiments/run_cycle4_followups_after_queue.sh`
    - expanded to execute all requested tracks after queue completion:
      1. transcoder stress hyper-sweep with CI-LCB threshold,
      2. grouped LCB candidate selection,
      3. matryoshka matched-budget external run,
      4. assignment-v3 with external-aware Pareto stage,
      5. known-circuit closure run with trained-vs-random CI,
      plus OOD stress and strict release-gate evaluation.
  - `RUNBOOK.md`
    - updated defaults/commands for grouped-LCB selection and CI-LCB transcoder gating.
  - `tests/unit/test_release_policy_selector.py`
    - explicit `--seed-level-selection` in seed-level test to match new selector defaults.

- Validation commands:
```bash
python -m py_compile scripts/experiments/select_release_candidate.py scripts/experiments/run_transcoder_stress_sweep.py scripts/experiments/run_matryoshka_frontier_external.py scripts/experiments/run_assignment_consistency_v3.py
bash -n scripts/experiments/run_cycle4_followups_after_queue.sh
pytest -q tests/unit/test_release_policy_selector.py tests/unit/test_husai_custom_sae_adapter.py
pytest -q tests/unit
make smoke
```
- Outcome: success
  - targeted unit tests: `6 passed`
  - full unit suite: `84 passed`
  - smoke pipeline: pass

- Live B200 status snapshot during this run:
  - queue still active: `results/experiments/cycle3_queue/queue_launcher_20260215T165724Z.log`
  - scaling progress snapshot: `train=6/24`, `sae=6/24`, `ce=5/24`

### Run 69: Queue-complete transition + selector metadata preservation
- Queue outcome observed on B200:
  - `results/experiments/cycle3_queue/run_20260215T165724Z/manifest.json` produced.
  - scaling run completed fully: `24/24` train, `24/24` SAEBench, `24/24` CE-Bench.
  - strict release gate returned non-zero (`release_policy_rc=2`).

- Code refinements after queue completion:
  - `scripts/experiments/select_release_candidate.py`
    - preserve grouped metadata (e.g., `uncertainty_mode`, grouped seed count fields) when writing selection scores.
    - fixes metadata overwrite where `annotate_scores` previously replaced the entire `selection` object.
  - `scripts/experiments/run_cycle4_followups_after_queue.sh`
    - hardened queue wait matcher to target only the actual queue process command.

- Validation:
```bash
pytest -q tests/unit/test_release_policy_selector.py
bash -n scripts/experiments/run_cycle4_followups_after_queue.sh
```
- Outcome: success (`4 passed`, shell syntax clean).

- Cycle-4 status at log time:
  - active run: `results/experiments/cycle4_followups/run_20260215T184508Z`
  - current stage: step 1 transcoder stress hyper-sweep in progress.

### Run 70: Cycle-4 step3 blocker fix and resume-capable followup script
- Observed failure on remote cycle-4 run (`run_20260215T184508Z`):
  - step3 (matryoshka frontier) crashed with missing activation cache path:
    - `FileNotFoundError: No activation files matched *_blocks.0.hook_resid_pre.pt under /workspace/HUSAI/results/cache/external_benchmarks/sae_bench_model_cache/model_activations_pythia-70m-deduped`

- Confirmed completed step1 before failure:
  - `results/experiments/phase4e_transcoder_stress_sweep_b200/run_20260215T184609Z/summary.md`
  - pass_all: `True`
  - best condition: `dsae128_k32_ep20_lr0.001`
  - best delta_lcb: `0.0010100603103637695`

- Fixes applied:
  - `scripts/experiments/run_cycle4_followups_after_queue.sh`
    - added explicit RunPod cache paths for matryoshka and assignment-v3 external eval stages:
      - `--activation-cache-dir /tmp/sae_bench_model_cache/model_activations_pythia-70m-deduped`
      - `--saebench-model-cache-path /tmp/sae_bench_model_cache`
      - tmp-backed results/artifacts paths for SAEBench/CE-Bench.
    - added `RESUME_FROM_STEP` to continue without rerunning finished stages.
    - preserved strict gate pipeline after step5 (selector + OOD + LCB release gate).

- Validation:
```bash
bash -n scripts/experiments/run_cycle4_followups_after_queue.sh
```
- Outcome: syntax check pass.

### Run 71: Cycle-4 followups completion evidence sync and gate verification
- Evidence synced locally from B200 run root:
  - `docs/evidence/cycle4_followups_run_20260215T190004Z/followups/manifest.json`
  - `docs/evidence/cycle4_followups_run_20260215T190004Z/selector/selection_summary.json`
  - `docs/evidence/cycle4_followups_run_20260215T190004Z/release_gate/release_policy.json`
  - `docs/evidence/cycle4_followups_run_20260215T190004Z/transcoder_sweep/results.json`
  - `docs/evidence/cycle4_followups_run_20260215T190004Z/matryoshka/results.json`
  - `docs/evidence/cycle4_followups_run_20260215T190004Z/assignment_v3/results.json`
  - `docs/evidence/cycle4_followups_run_20260215T190004Z/known_circuit/closure_summary.json`

- Outcome summary:
  - strict gate `pass_all=False`
  - random/transcoder/OOD pass
  - external SAEBench + CE-Bench fail
  - matryoshka cycle4 run had adapter normalization failures with dead-feature collapse (`l0=0`)
  - assignment-v3 external eval skipped due `d_model` mismatch

### Run 72: Reflective hardening pass (adapter + known-circuit + matryoshka path)
- Code updates:
  - `scripts/experiments/husai_custom_sae_adapter.py`
    - dead-decoder-row repair + encoder masking before decoder norm checks.
  - `scripts/experiments/run_known_circuit_recovery_closure.py`
    - switched SAE overlap basis to model-space projection from token Fourier basis.
    - added skipped-checkpoint reason accounting.
  - `scripts/experiments/run_matryoshka_frontier_external.py`
    - switched Matryoshka training to HUSAI `TopKSAE` with auxiliary dead-feature recovery.

- Tests added:
  - `tests/unit/test_husai_custom_sae_adapter.py`
  - `tests/unit/test_known_circuit_recovery_closure.py`

- Validation commands:
```bash
python -m py_compile scripts/experiments/husai_custom_sae_adapter.py scripts/experiments/run_known_circuit_recovery_closure.py scripts/experiments/run_matryoshka_frontier_external.py
pytest -q tests/unit/test_husai_custom_sae_adapter.py tests/unit/test_known_circuit_recovery_closure.py
pytest -q tests/unit/test_release_policy_selector.py tests/unit/test_external_scaling_baseline_map.py
python scripts/analysis/verify_experiment_consistency.py
```

- Outcome:
  - compile checks: pass
  - targeted unit tests: `6 passed`
  - selector/baseline tests: `6 passed`
  - consistency report regenerated:
    - `results/analysis/experiment_consistency_report.json`
    - `results/analysis/experiment_consistency_report.md`

### Run 73: Post-fix reruns on B200 (known-circuit + matryoshka)
- Remote code baseline:
  - pulled `main` at commit `de8d70b` on RunPod.

#### 73a) Known-circuit closure rerun
- Command:
```bash
python scripts/experiments/run_known_circuit_recovery_closure.py \
  --transformer-checkpoint results/transformer_5000ep/transformer_best.pt \
  --sae-checkpoint-glob 'results/experiments/phase4d_assignment_consistency_v3/run_*/checkpoints/lambda_*/sae_seed*.pt' \
  --output-dir results/experiments/known_circuit_recovery_closure
```
- Output:
  - `results/experiments/known_circuit_recovery_closure/run_20260215T203809Z/closure_summary.json`
- Synced evidence:
  - `docs/evidence/cycle4_postfix_reruns/known_circuit_run_20260215T203809Z_summary.json`
- Key result:
  - `checkpoints_evaluated=20` (previous artifact run had 0)
  - gate still failing, but now scientifically valid.

#### 73b) Matryoshka frontier rerun after fixes
- Command:
```bash
python scripts/experiments/run_matryoshka_frontier_external.py \
  --activation-cache-dir /tmp/sae_bench_model_cache/model_activations_pythia-70m-deduped \
  --activation-glob '*_blocks.0.hook_resid_pre.pt' \
  --max-files 80 --max-rows-per-file 2048 --max-total-rows 150000 \
  --seeds 42,123,456 --d-sae 1024 --k 32 --epochs 6 --batch-size 4096 \
  --learning-rate 0.001 --device cuda --dtype float32 \
  --run-saebench --run-cebench --cebench-repo /workspace/CE-Bench \
  --cebench-max-rows 200 \
  --cebench-matched-baseline-summary docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json \
  --saebench-model-cache-path /tmp/sae_bench_model_cache \
  --saebench-results-path /tmp/husai_saebench_probe_results_frontier_matryoshka_v2 \
  --cebench-artifacts-path /tmp/ce_bench_artifacts_frontier_matryoshka_v2 \
  --saebench-dataset-limit 8 \
  --output-dir results/experiments/phase4b_matryoshka_frontier_external
```
- Output:
  - `results/experiments/phase4b_matryoshka_frontier_external/run_20260215T203710Z/results.json`
- Synced evidence:
  - `docs/evidence/cycle4_postfix_reruns/matryoshka/run_20260215T203710Z_results.json`
- Key result:
  - external eval succeeds for all 3 seeds (no adapter normalization crash)
  - `train_l0_mean=32.0`, `train_ev_mean=0.6166`
  - `saebench_best_minus_llm_auc_mean=-0.03444`
  - `cebench_interp_delta_vs_baseline_mean=-40.16739`

### Run 74: Cycle-4 canonical sync + artifact hygiene hardening
- Scope:
  - synchronized canonical docs to latest followup evidence run:
    - `docs/evidence/cycle4_followups_run_20260215T220728Z/`
  - removed stale statements about unresolved assignment-v3 external compatibility.
  - updated final writeups and navigation docs to latest gate truth.

- Code hygiene fix:
  - `scripts/experiments/run_assignment_consistency_v3.py`
    - added JSON sanitization for non-finite floats (`NaN` -> `null`) before writing `results.json`.
    - switched JSON write to `allow_nan=False` for standards-compliant artifacts.

- Updated docs:
  - `README.md`
  - `START_HERE.md`
  - `EXECUTIVE_SUMMARY.md`
  - `PROJECT_STUDY_GUIDE.md`
  - `CYCLE4_FINAL_REFLECTIVE_REVIEW.md`
  - `ADVISOR_BRIEF.md`
  - `PROPOSAL_COMPLETENESS_REVIEW.md`
  - `REPO_NAVIGATION.md`
  - `FINAL_READINESS_REVIEW.md`
  - `HIGH_IMPACT_FOLLOWUPS_REPORT.md`
  - `FINAL_BLOG.md`
  - `FINAL_PAPER.md`

- Validation:
```bash
python -m py_compile scripts/experiments/run_assignment_consistency_v3.py
pytest -q tests/unit/test_assignment_consistency_v3.py
pytest -q tests/unit/test_husai_custom_sae_adapter.py tests/unit/test_known_circuit_recovery_closure.py
```

- Outcome:
  - py_compile: pass
  - `test_assignment_consistency_v3`: `2 passed`
  - adapter/known-circuit tests: `7 passed`

### Run 75: Cycle-5 external push (routed hyper-sweep + assignment sweep + reselection)
- New code and orchestration:
  - `scripts/experiments/run_routed_frontier_external.py`
    - added `--route-topk-mode` with new `expert_topk` routing mode to keep top-k within routed expert groups.
  - `scripts/experiments/select_release_candidate.py`
    - added `--assignment-results` ingestion so assignment-v3 external outputs can compete in global selection.
  - `scripts/experiments/run_cycle5_external_push.sh`
    - new high-impact queue covering routed sweep, assignment sweep, selector, OOD, strict gate.

- Remote cycle-5 queue run:
  - queue id: `results/experiments/cycle5_external_push/run_20260215T232351Z`
  - stage outputs:
    - routed sweep runs:
      - `run_20260215T232359Z`
      - `run_20260215T233353Z`
      - `run_20260215T234257Z`
      - `run_20260215T235219Z`
      - `run_20260216T000156Z`
    - assignment sweep runs:
      - `run_20260216T001059Z`
      - `run_20260216T005618Z`
    - selector run:
      - `results/experiments/release_candidate_selection_cycle5/run_20260216T014101Z`
    - OOD run:
      - `results/experiments/phase4e_ood_stress_b200/run_20260216T014105Z`
    - strict gate run:
      - `results/experiments/release_stress_gates/run_20260216T014820Z`

- Key numerical findings:
  - Routed best CE-Bench delta improved to `-37.260996` (`run_20260215T234257Z`) with restored effective sparsity (`l0=32`).
  - Assignment best CE-Bench delta improved to `-34.345572` (`run_20260216T005618Z`, `d_sae=2048`), but SAEBench delta remained negative (`-0.049864`).
  - Default selector (`min_seeds_per_group=3`) still selected baseline topk candidate.
  - Strict gate still failed (`pass_all=False`, external gate fail).

- Evidence synced locally:
  - `docs/evidence/cycle5_external_push_run_20260215T232351Z/`
  - synthesis summary:
    - `docs/evidence/cycle5_external_push_run_20260215T232351Z/cycle5_synthesis.md`

### Run 76: Post-cycle5 corrective analysis (selector threshold sensitivity)
- Ran corrected reselection with assignment integration and relaxed grouping threshold:
  - `results/experiments/release_candidate_selection_cycle5_min2/run_20260216T040024Z`
- Outcome:
  - selector switched from baseline `topk` to grouped assignment candidate `assignv3_lambda0.05`.
- Interpretation:
  - seed-group threshold can hide promising groups; this is a selector-policy sensitivity, not a model win.
- Additional note:
  - Assignment candidate still has negative SAEBench/CE-Bench deltas, so strict external-positive gate would remain failing.

### Run 77: Cycle-5 manifest bugfix
- Identified and fixed queue manifest serialization bug in `run_cycle5_external_push.sh`:
  - prior script wrote null metadata because non-exported shell vars were read from `os.environ`.
  - fixed by passing values as explicit Python argv to manifest writer.
- Commit:
  - `246dc4b` `Fix cycle5 manifest serialization and argument passing`

### Run 78: Post-cycle5 polish pass (selector diagnostics + determinism hardening + doc sync)
- Code changes:
  - `scripts/experiments/select_release_candidate.py`
    - added grouped-selection diagnostics (`grouping_diagnostics`) to output payload.
    - added warning when condition groups are dropped by `--min-seeds-per-group`.
    - added grouped diagnostics to markdown summary.
  - `scripts/experiments/run_cycle5_external_push.sh`
    - added deterministic env export: `CUBLAS_WORKSPACE_CONFIG`.
  - `scripts/experiments/run_b200_high_impact_queue.sh`
    - added deterministic env export: `CUBLAS_WORKSPACE_CONFIG`.

- Tests:
  - `tests/unit/test_release_policy_selector.py`
    - added `test_selector_grouped_reports_dropped_groups`.

- Documentation and organization sync:
  - updated cycle-5 canonical docs: `RUNBOOK.md`, `FINAL_BLOG.md`, `FINAL_PAPER.md`, `FINAL_READINESS_REVIEW.md`, `ADVISOR_BRIEF.md`, `HIGH_IMPACT_FOLLOWUPS_REPORT.md`, `PROPOSAL_COMPLETENESS_REVIEW.md`, `NOVEL_CONTRIBUTIONS.md`, `LIT_REVIEW.md`.
  - added learning guide: `LEARNING_PATH.md`.
  - refreshed navigation references in `README.md`, `START_HERE.md`, `EXECUTIVE_SUMMARY.md`, `PROJECT_STUDY_GUIDE.md`, `REPO_NAVIGATION.md`.

- Monitoring verification:
  - confirmed no active B200 jobs.
  - confirmed latest cycle-5 run remains canonical:
    - `results/experiments/cycle5_external_push/run_20260215T232351Z`
  - confirmed strict gate status unchanged:
    - `random=true`, `transcoder=true`, `ood=true`, `external=false`, `pass_all=false`.

- W&B check:
  - no active `WANDB_*` env in remote queue session.
  - no remote `wandb/run-*` directories for latest queue path; telemetry currently artifact-file based.

### Run 79: Cycle-6 SAE-aware push launch and live monitoring (in progress)
- Commit launched:
  - `fa7d0fc` `Add cycle6 SAE-aware queue and external checkpoint policy`
- Queue launch:
  - command: `bash scripts/experiments/run_cycle6_saeaware_push.sh`
  - queue run id: `results/experiments/cycle6_saeaware_push/run_20260216T054943Z`
  - launch log: `results/experiments/cycle6_saeaware_push/launch_20260216T054940Z.log`

- Stage status at `2026-02-16T05:59:11Z` (UTC):
  - active stage: cycle-6 stage1 routed sweep (`r1`) still running.
  - active process:
    - `scripts/experiments/run_routed_frontier_external.py`
    - seed-level external eval subprocesses (`run_husai_saebench_custom_eval.py` / `run_husai_cebench_custom_eval.py`).

- Partial metrics snapshot (`r1`, run `run_20260216T054951Z`):
  - completed seeds (SAEBench): 3
  - completed seeds (CE-Bench): 2
  - seeded checkpoints produced: 3
  - seed `42`:
    - SAEBench `best_minus_llm_auc = -0.066756772`
    - CE-Bench `interpretability_score_mean_max = 10.307504203`
    - CE-Bench delta vs matched baseline `interpretability_score_mean_max = -37.644107382`
  - seed `123`:
    - SAEBench `best_minus_llm_auc = -0.071979987`
    - CE-Bench delta vs matched baseline `interpretability_score_mean_max = -34.298578978`
  - seed `456`:
    - SAEBench complete (`-0.068519380`), CE-Bench pending at snapshot time.

- Monitoring checks:
  - GPU state: B200 memory allocated (python processes present), utilization fluctuating near external-eval boundaries.
  - queue logs: no hard runtime errors; only non-fatal pydantic warnings observed.
  - W&B telemetry:
    - remote environment has no `WANDB_*` variables.
    - no remote `wandb/run-*` directories present for this run.
    - telemetry remains artifact-file based (JSON/MD/log files under run directories).

- Notes:
  - cycle-6 launcher was hardened before launch to avoid self-wait deadlock and to fail fast on missing selector/inputs.
  - full synthesis pending completion of stage1-4 artifacts and strict release gate output.

### Run 80: Cycle-7 Pareto push design + deferred launch (queued)
- New queue script:
  - `scripts/experiments/run_cycle7_pareto_push.sh`
- Goal:
  - Target the routed SAEBench/CE-Bench trade-off directly using a Pareto-zone condition set, then run SAEBench-prioritized assignment-v3 + grouped LCB selection + strict gate.
- Key design choices:
  - Routed stage includes both cycle-5 anchors and interpolation settings (`d_sae`, `k`, `num_experts`).
  - Assignment stage uses external checkpoint policy `external_score` with SAEBench-heavy candidate weights.
  - Selector enforces grouped uncertainty-aware policy (`min_seeds_per_group=4`, plus min3 sensitivity run).
- Remote launch:
  - launch log: `results/experiments/cycle7_pareto_push/launch_20260216T062212Z.log`
  - run dir: `results/experiments/cycle7_pareto_push/run_20260216T062213Z`
  - current behavior: waiting behind active cycle-6 runner (intentional).
- Additional planning artifact:
  - `CYCLE7_PARETO_PLAN.md`

### Run 81: Cycle-6 stage transition snapshot + cycle-7 queue verification (in progress)
- Snapshot time (UTC): `2026-02-16T06:48:14Z`
- Active cycle-6 run:
  - `results/experiments/cycle6_saeaware_push/run_20260216T054943Z`
- Stage transitions confirmed:
  - stage1 routed sweep complete (`r1..r4`)
  - stage2 assignment-v3 started (`a1` active)

- Cycle-6 routed aggregate means (from `phase4b_routed_frontier_external_sweep_cycle6`):
  - `run_20260216T054951Z` (`d_sae=2048,k=32,experts=8`):
    - `ce_mean=-36.5004`, `sae_mean=-0.07083`
  - `run_20260216T060326Z` (`d_sae=2048,k=32,experts=12`):
    - `ce_mean=-36.1916`, `sae_mean=-0.07193`
  - `run_20260216T061625Z` (`d_sae=2048,k=48,experts=8`):
    - `ce_mean=-37.3348`, `sae_mean=-0.07448`
  - `run_20260216T062924Z` (`d_sae=3072,k=48,experts=12`):
    - `ce_mean=-38.4639`, `sae_mean=-0.07603`

- Interim interpretation:
  - CE improved modestly versus earlier cycle-5 routed means in some settings.
  - SAEBench remained negative and generally worsened in this routed cycle.
  - This validates the cycle-7 Pareto-zone redesign emphasis on recovering SAEBench without CE collapse.

- Assignment-stage health check (`run_20260216T064311Z`):
  - checkpoint count increased to `8` (`lambda_0.0` complete across seeds + `lambda_0.03` started)
  - no external-eval summaries yet (expected at this stage of training)

- Deferred cycle-7 queue verification:
  - launch log: `results/experiments/cycle7_pareto_push/launch_20260216T062212Z.log`
  - run dir: `results/experiments/cycle7_pareto_push/run_20260216T062213Z`
  - status: waiting behind active cycle-6 process (expected behavior)

### Run 82: Routed robustness/diversity upgrade + cycle-8 queue preparation
- Goal:
  - Add high-impact routed regularization controls to improve external generalization (especially SAEBench) and queue a strict grouped-LCB follow-up cycle.

- Code changes:
  - `scripts/experiments/run_routed_frontier_external.py`
    - added decoder diversity penalty helper (`decoder_diversity_penalty`).
    - added routed robustness controls:
      - `--robust-noise-std`
      - `--route-consistency-coef`
      - `--decoder-diversity-coef`
      - `--decoder-diversity-sample`
    - integrated new losses into training objective and logged aggregates:
      - `route_consistency_loss`
      - `decoder_diversity_loss`
  - `tests/unit/test_routed_frontier_modes.py`
    - added tests for diversity penalty behavior (orthogonal vs duplicate columns).
  - `scripts/experiments/run_cycle8_robust_pareto_push.sh`
    - added new cycle-8 queue with robust routed sweep + assignment-v3 external-aware sweep + grouped-LCB selector + strict release gate.
  - `CYCLE8_ROBUST_PLAN.md`
    - added cycle rationale, stage design, success criteria, and literature anchors.

- Validation:
```bash
python -m py_compile scripts/experiments/run_routed_frontier_external.py
bash -n scripts/experiments/run_cycle8_robust_pareto_push.sh
pytest -q tests/unit/test_routed_frontier_modes.py
pytest -q tests/unit/test_release_policy_selector.py tests/unit/test_assignment_consistency_v3.py
```
- Outcome: success
  - routed compile: pass
  - cycle-8 shell syntax: pass
  - routed test suite: pass (`4 passed`)
  - selector/assignment tests: pass (`10 passed`)

- Live remote monitoring snapshot during this run:
  - cycle-7 queue active on B200:
    - `results/experiments/cycle7_pareto_push/run_20260216T062213Z/cycle7.log`
  - routed stage completed `p1..p3`, `p4` in progress at snapshot time.

### Run 83: Live cycle-7 monitoring snapshot + cycle-8 launch
- Remote status confirmed on RunPod B200:
  - cycle-7 run: `results/experiments/cycle7_pareto_push/run_20260216T062213Z`
  - stage-1 routed sweep complete (`p1..p5`)
  - stage-2 assignment run active:
    - `results/experiments/phase4d_assignment_consistency_v3_cycle7_pareto/run_20260216T142558Z`
  - cycle-8 queue launched and waiting behind cycle-7:
    - `results/experiments/cycle8_robust_pareto_push/run_20260216T163502Z`

- Routed stage aggregate snapshot (cycle-7):
  - best SAEBench delta among p1..p5: `-0.0638067108` (`run_20260216T133659Z`)
  - best CE-Bench delta among p1..p5: `-36.1834613471` (`run_20260216T140951Z`)

- Assignment stage progress snapshot (`run_20260216T142558Z`):
  - checkpoints: `56`
  - completed SAEBench summaries: `>=5` and increasing
  - completed CE-Bench summaries: `>=4` and increasing

- W&B telemetry check:
  - no remote `WANDB_*` env vars detected
  - no active remote `wandb/run-*` outputs for current queue
  - telemetry remains artifact/log-file based

- Evidence synced locally:
  - `docs/evidence/cycle7_live_snapshot_20260216T165714Z/monitoring_summary.md`
  - `docs/evidence/cycle7_live_snapshot_20260216T165714Z/assignment/progress_snapshot.json`
  - `docs/evidence/cycle7_live_snapshot_20260216T165714Z/cycle7.log`
  - `docs/evidence/cycle7_live_snapshot_20260216T165714Z/cycle8/cycle8.log`

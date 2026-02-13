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

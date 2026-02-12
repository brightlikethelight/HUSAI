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

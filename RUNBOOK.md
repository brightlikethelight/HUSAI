# Runbook

Updated: 2026-03-05

## 1) Environment Setup

```bash
conda env create -f environment.yml
conda activate husai
pip install -r requirements-dev.txt
pre-commit install
```

Recommended runtime defaults:

```bash
export HUSAI_ROOT="$(pwd)"
export HUSAI_TMP_ROOT="${HUSAI_TMP_ROOT:-$HUSAI_ROOT/tmp}"
mkdir -p "$HUSAI_TMP_ROOT" "$HUSAI_TMP_ROOT/mpl"
export TMPDIR="$HUSAI_TMP_ROOT"
export MPLCONFIGDIR="$HUSAI_TMP_ROOT/mpl"
export KMP_DUPLICATE_LIB_OK=TRUE
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CEBENCH_REPO="${CEBENCH_REPO:-$HUSAI_ROOT/../CE-Bench}"
```

## 2) Quality Checks

```bash
pytest tests -q
make smoke
python scripts/analysis/verify_experiment_consistency.py
```

## 3) One-Click Core Reproduction

```bash
make reproduce-phase4a
make ablate-core
python scripts/experiments/run_assignment_consistency_v3.py --device cpu
```

## 4) External Evaluation Commands

```bash
python scripts/experiments/run_husai_saebench_custom_eval.py \
  --checkpoint <ckpt.pt> \
  --model-name pythia-70m-deduped \
  --hook-layer 0 \
  --hook-name blocks.0.hook_resid_pre \
  --device cuda

python scripts/experiments/run_husai_cebench_custom_eval.py \
  --cebench-repo "$CEBENCH_REPO" \
  --checkpoint <ckpt.pt> \
  --model-name pythia-70m-deduped \
  --hook-layer 0 \
  --hook-name blocks.0.hook_resid_pre \
  --device cuda
```

## 5) Strict Gate Evaluation

```bash
python scripts/experiments/run_stress_gated_release_policy.py \
  --phase4a-results results/experiments/phase4a_trained_vs_random/results.json \
  --transcoder-results <transcoder_summary.json> \
  --ood-results <ood_summary.json> \
  --external-candidate-json <selected_candidate.json> \
  --external-mode joint \
  --use-external-lcb \
  --min-saebench-delta-lcb 0.0 \
  --min-cebench-delta-lcb 0.0 \
  --require-transcoder --require-ood --require-external
```

## 6) Experiment Logging Requirements

Every run must capture:
- command
- git commit
- config hash
- seeds
- dataset slice/version
- artifact root

Then append a run entry to `EXPERIMENT_LOG.md`.

## 7) Claim Integrity Rule

Before writing final numbers, verify claim tier in `EVIDENCE_STATUS.md`.

# Quick Start

Updated: 2026-03-05

Use this file for the fastest local verification path.

## 1) Setup

```bash
conda env create -f environment.yml
conda activate husai
pip install -r requirements-dev.txt
```

## 2) Deterministic Runtime Defaults

```bash
export HUSAI_ROOT="$(pwd)"
export HUSAI_TMP_ROOT="${HUSAI_TMP_ROOT:-$HUSAI_ROOT/tmp}"
mkdir -p "$HUSAI_TMP_ROOT" "$HUSAI_TMP_ROOT/mpl"
export TMPDIR="$HUSAI_TMP_ROOT"
export MPLCONFIGDIR="$HUSAI_TMP_ROOT/mpl"
export KMP_DUPLICATE_LIB_OK=TRUE
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## 3) Verify Health

```bash
pytest tests -q
make smoke
```

## 4) Minimal Reproduction Path

```bash
python scripts/experiments/run_phase4a_reproduction.py
python scripts/experiments/run_core_ablations.py
python scripts/experiments/run_assignment_consistency_v3.py --device cpu
```

## 5) External Evaluation Path

```bash
python scripts/experiments/run_husai_saebench_custom_eval.py \
  --checkpoint <ckpt.pt> \
  --model-name pythia-70m-deduped \
  --hook-layer 0 \
  --hook-name blocks.0.hook_resid_pre \
  --device cuda

python scripts/experiments/run_husai_cebench_custom_eval.py \
  --cebench-repo <path/to/CE-Bench> \
  --checkpoint <ckpt.pt> \
  --model-name pythia-70m-deduped \
  --hook-layer 0 \
  --hook-name blocks.0.hook_resid_pre \
  --device cuda
```

## 6) Claim Discipline

Before citing final-cycle metrics, read `EVIDENCE_STATUS.md` to separate local verified artifacts from remote-reported package references.

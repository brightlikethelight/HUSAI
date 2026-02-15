#!/usr/bin/env bash
set -euo pipefail

cd /workspace/HUSAI
mkdir -p results/experiments/cycle4_followups

echo "[followups] waiting for active queue to finish..."
while pgrep -f "run_b200_high_impact_queue.sh" >/dev/null 2>&1; do
  date -u +"[followups] %Y-%m-%dT%H:%M:%SZ queue still running"
  sleep 60
done

echo "[followups] queue finished; pulling latest main"
git pull origin main

echo "[followups] run transcoder stress sweep"
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_transcoder_stress_sweep.py \
  --d-sae-values 128,256 \
  --k-values 16,32 \
  --epochs-values 12 \
  --learning-rate-values 0.0003,0.001 \
  --device cuda \
  --output-dir results/experiments/phase4e_transcoder_stress_sweep_b200

echo "[followups] run matryoshka frontier external pilot"
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_matryoshka_frontier_external.py \
  --seeds 42,123 \
  --d-sae 1024 \
  --k 32 \
  --epochs 6 \
  --batch-size 4096 \
  --learning-rate 0.001 \
  --device cuda \
  --run-saebench \
  --run-cebench \
  --cebench-repo /workspace/CE-Bench \
  --cebench-max-rows 200 \
  --cebench-matched-baseline-summary docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json \
  --output-dir results/experiments/phase4b_matryoshka_frontier_external

echo "[followups] run assignment v3 internal sweep"
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_assignment_consistency_v3.py \
  --device cpu \
  --epochs 20 \
  --output-dir results/experiments/phase4d_assignment_consistency_v3

echo "[followups] done"

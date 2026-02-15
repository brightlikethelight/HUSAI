#!/usr/bin/env bash
set -euo pipefail

cd /workspace/HUSAI

WAIT_SECONDS="${WAIT_SECONDS:-60}"
CEBENCH_REPO="${CEBENCH_REPO:-/workspace/CE-Bench}"
MIN_TRANSCODER_DELTA_LCB="${MIN_TRANSCODER_DELTA_LCB:-0.0}"
FAIL_ON_TRANSCODER_FAIL="${FAIL_ON_TRANSCODER_FAIL:-1}"
FAIL_ON_RELEASE_GATE_FAIL="${FAIL_ON_RELEASE_GATE_FAIL:-0}"
MIN_SAEBENCH_DELTA_LCB="${MIN_SAEBENCH_DELTA_LCB:-0.0}"
MIN_CEBENCH_DELTA_LCB="${MIN_CEBENCH_DELTA_LCB:-0.0}"

SAEBENCH_DATASETS="${SAEBENCH_DATASETS:-100_news_fake,105_click_bait,106_hate_hate,107_hate_offensive,110_aimade_humangpt3,113_movie_sent,114_nyc_borough_Manhattan,115_nyc_borough_Brooklyn,116_nyc_borough_Bronx,117_us_state_FL,118_us_state_CA,119_us_state_TX,120_us_timezone_Chicago,121_us_timezone_New_York,122_us_timezone_Los_Angeles,123_world_country_United_Kingdom}"

RUN_ID="run_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="results/experiments/cycle4_followups/${RUN_ID}"
mkdir -p "$RUN_DIR"
LOG_PATH="$RUN_DIR/followups.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[followups] run_id=$RUN_ID"
echo "[followups] waiting for active queue to finish..."
while pgrep -f "run_b200_high_impact_queue.sh" >/dev/null 2>&1; do
  date -u +"[followups] %Y-%m-%dT%H:%M:%SZ queue still running"
  sleep "$WAIT_SECONDS"
done

echo "[followups] queue finished; pulling latest main"
git pull origin main

echo "[followups] step1: transcoder stress hyper-sweep (CI-LCB gate)"
TRANS_ARGS=(
  --d-sae-values 128,256
  --k-values 16,32
  --epochs-values 12,20
  --learning-rate-values 0.0003,0.001
  --device cuda
  --min-delta-lcb "$MIN_TRANSCODER_DELTA_LCB"
  --output-dir results/experiments/phase4e_transcoder_stress_sweep_b200
)
if [[ "$FAIL_ON_TRANSCODER_FAIL" == "1" ]]; then
  TRANS_ARGS+=(--fail-on-gate-fail)
fi
set +e
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_transcoder_stress_sweep.py "${TRANS_ARGS[@]}"
TRANS_RC=$?
set -e

echo "[followups] step1_rc=$TRANS_RC"
TRANS_SWEEP_RUN="$(ls -1dt results/experiments/phase4e_transcoder_stress_sweep_b200/run_* 2>/dev/null | head -n1 || true)"
TRANS_SWEEP_RESULTS="${TRANS_SWEEP_RUN}/results.json"
BEST_TRANSCODER_SUMMARY=""
if [[ -f "$TRANS_SWEEP_RESULTS" ]]; then
  BEST_TRANSCODER_SUMMARY="$(python - "$TRANS_SWEEP_RESULTS" <<'PY'
import json,sys
obj=json.load(open(sys.argv[1]))
best=obj.get("best_condition") or {}
print(best.get("summary_path") or "")
PY
)"
fi
if [[ -n "$BEST_TRANSCODER_SUMMARY" && "$BEST_TRANSCODER_SUMMARY" != /* ]]; then
  BEST_TRANSCODER_SUMMARY="/workspace/HUSAI/${BEST_TRANSCODER_SUMMARY}"
fi


echo "[followups] step3: matryoshka frontier under matched budget"
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_matryoshka_frontier_external.py \
  --seeds 42,123,456 \
  --d-sae 1024 \
  --k 32 \
  --epochs 6 \
  --batch-size 4096 \
  --learning-rate 0.001 \
  --device cuda \
  --run-saebench \
  --run-cebench \
  --cebench-repo "$CEBENCH_REPO" \
  --cebench-max-rows 200 \
  --cebench-matched-baseline-summary docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json \
  --saebench-datasets "$SAEBENCH_DATASETS" \
  --output-dir results/experiments/phase4b_matryoshka_frontier_external


echo "[followups] step4: assignment-aware v3 with external-aware Pareto selection"
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_assignment_consistency_v3.py \
  --device cuda \
  --epochs 20 \
  --train-seeds 123,456,789 \
  --lambdas 0.0,0.05,0.1,0.2,0.3 \
  --run-saebench \
  --run-cebench \
  --cebench-repo "$CEBENCH_REPO" \
  --cebench-max-rows 200 \
  --cebench-matched-baseline-summary docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json \
  --saebench-datasets "$SAEBENCH_DATASETS" \
  --force-rerun-external \
  --require-external \
  --min-saebench-delta 0.0 \
  --min-cebench-delta 0.0 \
  --output-dir results/experiments/phase4d_assignment_consistency_v3


echo "[followups] step5: known-circuit closure with trained-vs-random confidence bounds"
python scripts/experiments/run_known_circuit_recovery_closure.py \
  --transformer-checkpoint results/transformer_5000ep/transformer_best.pt \
  --sae-checkpoint-glob 'results/experiments/phase4d_assignment_consistency_v3/run_*/checkpoints/lambda_*/sae_seed*.pt' \
  --output-dir results/experiments/known_circuit_recovery_closure


echo "[followups] step2: default grouped uncertainty-aware (LCB) candidate selection"
FRONTIER_BASE="$(ls -1dt results/experiments/phase4b_architecture_frontier_external_multiseed/run_* 2>/dev/null | head -n1 || true)"
FRONTIER_MATRY="$(ls -1dt results/experiments/phase4b_matryoshka_frontier_external/run_* 2>/dev/null | head -n1 || true)"
SCALING_RUN="$(ls -1dt results/experiments/phase4e_external_scaling_study_multiseed/run_* 2>/dev/null | head -n1 || true)"

SELECTOR_CMD=(
  python scripts/experiments/select_release_candidate.py
  --frontier-results "$FRONTIER_BASE/results.json"
  --scaling-results "$SCALING_RUN/results.json"
  --require-both-external
  --output-dir results/experiments/release_candidate_selection
)
if [[ -n "$FRONTIER_MATRY" && -f "$FRONTIER_MATRY/results.json" ]]; then
  SELECTOR_CMD+=(--frontier-results "$FRONTIER_MATRY/results.json")
fi
"${SELECTOR_CMD[@]}"

SELECTOR_RUN="$(ls -1dt results/experiments/release_candidate_selection/run_* 2>/dev/null | head -n1 || true)"
SELECTED_JSON="$SELECTOR_RUN/selected_candidate.json"

read -r BEST_CHECKPOINT BEST_ARCH BEST_HOOK_LAYER BEST_HOOK_NAME < <(
  python - "$SELECTED_JSON" <<'PY'
import json,sys
obj=json.load(open(sys.argv[1]))
print(
  obj.get("checkpoint", ""),
  obj.get("architecture", "topk"),
  obj.get("hook_layer", 0),
  obj.get("hook_name", "blocks.0.hook_resid_pre"),
)
PY
)

if [[ -n "$BEST_CHECKPOINT" && "$BEST_CHECKPOINT" != /* ]]; then
  BEST_CHECKPOINT="/workspace/HUSAI/$BEST_CHECKPOINT"
fi


echo "[followups] run OOD stress on selected candidate"
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_ood_stress_eval.py \
  --checkpoint "$BEST_CHECKPOINT" \
  --architecture "$BEST_ARCH" \
  --sae-release husai_cycle4_ood \
  --model-name pythia-70m-deduped \
  --hook-layer "$BEST_HOOK_LAYER" \
  --hook-name "$BEST_HOOK_NAME" \
  --device cuda \
  --dtype float32 \
  --results-path /tmp/husai_saebench_probe_results_ood_cycle4 \
  --model-cache-path /tmp/sae_bench_model_cache \
  --output-dir results/experiments/phase4e_ood_stress_b200 \
  --force-rerun

OOD_SUMMARY="$(ls -1dt results/experiments/phase4e_ood_stress_b200/run_*/ood_stress_summary.json 2>/dev/null | head -n1 || true)"


echo "[followups] strict release gate with external LCB thresholds"
GATE_ARGS=(
  --phase4a-results results/experiments/phase4a_trained_vs_random/results.json
  --transcoder-results "$BEST_TRANSCODER_SUMMARY"
  --ood-results "$OOD_SUMMARY"
  --external-candidate-json "$SELECTED_JSON"
  --external-mode joint
  --use-external-lcb
  --min-saebench-delta-lcb "$MIN_SAEBENCH_DELTA_LCB"
  --min-cebench-delta-lcb "$MIN_CEBENCH_DELTA_LCB"
  --require-transcoder
  --require-ood
  --require-external
)
if [[ "$FAIL_ON_RELEASE_GATE_FAIL" == "1" ]]; then
  GATE_ARGS+=(--fail-on-gate-fail)
fi
set +e
python scripts/experiments/run_stress_gated_release_policy.py "${GATE_ARGS[@]}"
RELEASE_RC=$?
set -e

RELEASE_RUN="$(ls -1dt results/experiments/release_stress_gates/run_* 2>/dev/null | head -n1 || true)"
RELEASE_POLICY_JSON="$RELEASE_RUN/release_policy.json"

cat > "$RUN_DIR/manifest.json" <<MANIFEST
{
  "run_id": "$RUN_ID",
  "transcoder_sweep_run": "$TRANS_SWEEP_RUN",
  "transcoder_sweep_results": "$TRANS_SWEEP_RESULTS",
  "best_transcoder_summary": "$BEST_TRANSCODER_SUMMARY",
  "transcoder_sweep_rc": $TRANS_RC,
  "frontier_base_run": "$FRONTIER_BASE",
  "frontier_matry_run": "$FRONTIER_MATRY",
  "scaling_run": "$SCALING_RUN",
  "selector_run": "$SELECTOR_RUN",
  "selected_candidate_json": "$SELECTED_JSON",
  "best_checkpoint": "$BEST_CHECKPOINT",
  "best_architecture": "$BEST_ARCH",
  "best_hook_layer": "$BEST_HOOK_LAYER",
  "best_hook_name": "$BEST_HOOK_NAME",
  "ood_summary": "$OOD_SUMMARY",
  "release_run": "$RELEASE_RUN",
  "release_policy_json": "$RELEASE_POLICY_JSON",
  "release_rc": $RELEASE_RC
}
MANIFEST

echo "[followups] complete"
echo "[followups] manifest=$RUN_DIR/manifest.json"

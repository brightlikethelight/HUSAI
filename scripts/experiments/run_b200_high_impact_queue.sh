#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/experiments/cycle3_queue
QUEUE_ID="run_$(date -u +%Y%m%dT%H%M%SZ)"
QUEUE_DIR="results/experiments/cycle3_queue/${QUEUE_ID}"
mkdir -p "$QUEUE_DIR"
LOG_PATH="$QUEUE_DIR/queue.log"
exec > >(tee -a "$LOG_PATH") 2>&1

WAIT_SECONDS="${WAIT_SECONDS:-60}"
CEBENCH_REPO="${CEBENCH_REPO:-/workspace/CE-Bench}"
MIN_SAEBENCH_DELTA="${MIN_SAEBENCH_DELTA:-0.0}"
MIN_CEBENCH_DELTA="${MIN_CEBENCH_DELTA:-0.0}"
MIN_SAEBENCH_DELTA_LCB="${MIN_SAEBENCH_DELTA_LCB:-0.0}"
MIN_CEBENCH_DELTA_LCB="${MIN_CEBENCH_DELTA_LCB:-0.0}"
USE_EXTERNAL_LCB_GATES="${USE_EXTERNAL_LCB_GATES:-1}"
DEFAULT_CEBENCH_BASELINE="docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json"
CEBENCH_BASELINE_MAP="${CEBENCH_BASELINE_MAP:-docs/evidence/phase4e_cebench_matched200/cebench_baseline_map.json}"
SELECTOR_GROUP_BY_CONDITION="${SELECTOR_GROUP_BY_CONDITION:-1}"
SELECTOR_UNCERTAINTY_MODE="${SELECTOR_UNCERTAINTY_MODE:-lcb}"
MIN_SEEDS_PER_GROUP="${MIN_SEEDS_PER_GROUP:-3}"

SAEBENCH_DATASETS="${SAEBENCH_DATASETS:-100_news_fake,105_click_bait,106_hate_hate,107_hate_offensive,110_aimade_humangpt3,113_movie_sent,114_nyc_borough_Manhattan,115_nyc_borough_Brooklyn,116_nyc_borough_Bronx,117_us_state_FL,118_us_state_CA,119_us_state_TX,120_us_timezone_Chicago,121_us_timezone_New_York,122_us_timezone_Los_Angeles,123_world_country_United_Kingdom}"

echo "[queue] id=${QUEUE_ID}"
echo "[queue] root=${ROOT_DIR}"

echo "[queue] waiting for active frontier process to finish..."
while pgrep -f "scripts/experiments/run_architecture_frontier_external.py" >/dev/null 2>&1; do
  date -u +"[queue] %Y-%m-%dT%H:%M:%SZ frontier still running"
  sleep "$WAIT_SECONDS"
done

echo "[queue] frontier process no longer active"

FRONTIER_RUN="${FRONTIER_RUN:-$(ls -1dt results/experiments/phase4b_architecture_frontier_external_multiseed/run_* 2>/dev/null | head -n1 || true)}"
if [[ -z "$FRONTIER_RUN" ]]; then
  echo "[queue] ERROR: could not locate a frontier multiseed run directory"
  exit 1
fi

echo "[queue] frontier_run=${FRONTIER_RUN}"

if [[ ! -f "$FRONTIER_RUN/results.json" ]]; then
  echo "[queue] ERROR: frontier run has no results.json: $FRONTIER_RUN/results.json"
  exit 1
fi

echo "[queue] launching multiseed external scaling study"
SCALING_CEBENCH_ARGS=(
  --cebench-matched-baseline-summary "$DEFAULT_CEBENCH_BASELINE"
)
if [[ -f "$CEBENCH_BASELINE_MAP" ]]; then
  SCALING_CEBENCH_ARGS+=(--cebench-matched-baseline-map "$CEBENCH_BASELINE_MAP")
  echo "[queue] using CE-Bench baseline map: $CEBENCH_BASELINE_MAP"
else
  echo "[queue] CE-Bench baseline map not found, using default baseline only: $DEFAULT_CEBENCH_BASELINE"
fi

KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_external_metric_scaling_study.py \
  --activation-cache-dir /tmp/sae_bench_model_cache/model_activations_pythia-70m-deduped \
  --activation-glob-template '*_blocks.{layer}.hook_resid_pre.pt' \
  --hook-name-template 'blocks.{layer}.hook_resid_pre' \
  --token-budgets 10000,30000 \
  --hook-layers 0,1 \
  --d-sae-values 1024,2048 \
  --seeds 42,123,456 \
  --k 32 \
  --epochs 6 \
  --batch-size 4096 \
  --learning-rate 0.001 \
  --max-files 16 \
  --max-rows-per-file 2048 \
  --run-saebench \
  --run-cebench \
  --cebench-repo "$CEBENCH_REPO" \
  --cebench-max-rows 200 \
  "${SCALING_CEBENCH_ARGS[@]}" \
  --saebench-datasets "$SAEBENCH_DATASETS" \
  --saebench-results-path /tmp/husai_saebench_probe_results_scaling_multiseed \
  --saebench-model-cache-path /tmp/sae_bench_model_cache \
  --cebench-artifacts-path /tmp/ce_bench_artifacts_scaling_multiseed \
  --device cuda \
  --dtype float32 \
  --output-dir results/experiments/phase4e_external_scaling_study_multiseed

SCALING_RUN="$(ls -1dt results/experiments/phase4e_external_scaling_study_multiseed/run_* | head -n1)"
echo "[queue] scaling_run=${SCALING_RUN}"

if [[ ! -f "$SCALING_RUN/results.json" ]]; then
  echo "[queue] ERROR: scaling run has no results.json: $SCALING_RUN/results.json"
  exit 1
fi

echo "[queue] selecting release candidate using multi-objective selector"
SELECTOR_ARGS=(
  --frontier-results "$FRONTIER_RUN/results.json"
  --scaling-results "$SCALING_RUN/results.json"
  --require-both-external
  --output-dir results/experiments/release_candidate_selection
)
if [[ "$SELECTOR_GROUP_BY_CONDITION" == "1" ]]; then
  SELECTOR_ARGS+=(
    --group-by-condition
    --uncertainty-mode "$SELECTOR_UNCERTAINTY_MODE"
    --min-seeds-per-group "$MIN_SEEDS_PER_GROUP"
  )
fi
python scripts/experiments/select_release_candidate.py "${SELECTOR_ARGS[@]}"

SELECTOR_RUN="$(ls -1dt results/experiments/release_candidate_selection/run_* | head -n1)"
SELECTED_JSON="${SELECTOR_RUN}/selected_candidate.json"
CANDIDATE_JSON="$SELECTED_JSON"

if [[ ! -f "$SELECTED_JSON" ]]; then
  echo "[queue] ERROR: selector output not found: $SELECTED_JSON"
  exit 1
fi

read -r BEST_CHECKPOINT BEST_ARCH BEST_HOOK_LAYER BEST_HOOK_NAME BEST_SAEBENCH_SUMMARY BEST_CEBENCH_SUMMARY < <(
  SELECTED_JSON="$SELECTED_JSON" python - <<'PY'
import json
import os

path = os.environ["SELECTED_JSON"]
d = json.load(open(path))
print(
    d.get("checkpoint", ""),
    d.get("architecture", "topk"),
    d.get("hook_layer", 0),
    d.get("hook_name", "blocks.0.hook_resid_pre"),
    d.get("saebench_summary_path", ""),
    d.get("cebench_summary_path", ""),
)
PY
)

echo "[queue] best_checkpoint=${BEST_CHECKPOINT}"
echo "[queue] best_architecture=${BEST_ARCH}"
echo "[queue] best_hook_layer=${BEST_HOOK_LAYER}"
echo "[queue] best_hook_name=${BEST_HOOK_NAME}"
echo "[queue] selected_candidate_json=${SELECTED_JSON}"

echo "[queue] launching transcoder stress eval"
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_transcoder_stress_eval.py \
  --transformer-checkpoint results/transformer_5000ep/transformer_best.pt \
  --device cuda \
  --epochs 12 \
  --seeds 42,123,456 \
  --batch-size 512 \
  --output-dir results/experiments/phase4e_transcoder_stress_b200

TRANSCODER_SUMMARY="$(ls -1dt results/experiments/phase4e_transcoder_stress_b200/run_*/transcoder_stress_summary.json | head -n1)"
echo "[queue] transcoder_summary=${TRANSCODER_SUMMARY}"

echo "[queue] launching OOD stress eval on selected candidate"
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_ood_stress_eval.py \
  --checkpoint "$BEST_CHECKPOINT" \
  --architecture "$BEST_ARCH" \
  --sae-release husai_cycle3_ood \
  --model-name pythia-70m-deduped \
  --hook-layer "$BEST_HOOK_LAYER" \
  --hook-name "$BEST_HOOK_NAME" \
  --device cuda \
  --dtype float32 \
  --results-path /tmp/husai_saebench_probe_results_ood_cycle3 \
  --model-cache-path /tmp/sae_bench_model_cache \
  --output-dir results/experiments/phase4e_ood_stress_b200 \
  --force-rerun

OOD_SUMMARY="$(ls -1dt results/experiments/phase4e_ood_stress_b200/run_*/ood_stress_summary.json | head -n1)"
echo "[queue] ood_summary=${OOD_SUMMARY}"

echo "[queue] evaluating strict release policy (joint external mode)"
GATE_ARGS=(
  --phase4a-results results/experiments/phase4a_trained_vs_random/results.json
  --transcoder-results "$TRANSCODER_SUMMARY"
  --ood-results "$OOD_SUMMARY"
  --external-candidate-json "$SELECTED_JSON"
  --external-mode joint
  --min-saebench-delta "$MIN_SAEBENCH_DELTA"
  --min-cebench-delta "$MIN_CEBENCH_DELTA"
  --require-transcoder
  --require-ood
  --require-external
  --fail-on-gate-fail
)
if [[ "$USE_EXTERNAL_LCB_GATES" == "1" ]]; then
  GATE_ARGS+=(
    --use-external-lcb
    --min-saebench-delta-lcb "$MIN_SAEBENCH_DELTA_LCB"
    --min-cebench-delta-lcb "$MIN_CEBENCH_DELTA_LCB"
  )
fi

set +e
python scripts/experiments/run_stress_gated_release_policy.py "${GATE_ARGS[@]}"
RELEASE_RC=$?
set -e

echo "[queue] release_policy_rc=${RELEASE_RC}"

cat > "$QUEUE_DIR/manifest.json" <<MANIFEST
{
  "queue_id": "${QUEUE_ID}",
  "frontier_run": "${FRONTIER_RUN}",
  "scaling_run": "${SCALING_RUN}",
  "selector_run": "${SELECTOR_RUN}",
  "selected_candidate_json": "${SELECTED_JSON}",
  "candidate_json": "${CANDIDATE_JSON}",
  "best_saebench_summary": "${BEST_SAEBENCH_SUMMARY}",
  "best_cebench_summary": "${BEST_CEBENCH_SUMMARY}",
  "transcoder_summary": "${TRANSCODER_SUMMARY}",
  "ood_summary": "${OOD_SUMMARY}",
  "release_policy_rc": ${RELEASE_RC},
  "cebench_baseline_default": "${DEFAULT_CEBENCH_BASELINE}",
  "cebench_baseline_map": "${CEBENCH_BASELINE_MAP}",
  "selector_group_by_condition": "${SELECTOR_GROUP_BY_CONDITION}",
  "selector_uncertainty_mode": "${SELECTOR_UNCERTAINTY_MODE}",
  "min_seeds_per_group": "${MIN_SEEDS_PER_GROUP}",
  "use_external_lcb_gates": "${USE_EXTERNAL_LCB_GATES}",
  "min_saebench_delta_lcb": "${MIN_SAEBENCH_DELTA_LCB}",
  "min_cebench_delta_lcb": "${MIN_CEBENCH_DELTA_LCB}"
}
MANIFEST

echo "[queue] complete"
echo "[queue] manifest=${QUEUE_DIR}/manifest.json"

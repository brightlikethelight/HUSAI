#!/usr/bin/env bash
set -euo pipefail

cd /workspace/HUSAI

WAIT_SECONDS="${WAIT_SECONDS:-60}"
CEBENCH_REPO="${CEBENCH_REPO:-/workspace/CE-Bench}"
ACTIVATION_CACHE_DIR="${ACTIVATION_CACHE_DIR:-/tmp/sae_bench_model_cache/model_activations_pythia-70m-deduped}"
SAEBENCH_MODEL_CACHE_PATH="${SAEBENCH_MODEL_CACHE_PATH:-/tmp/sae_bench_model_cache}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
SAEBENCH_DATASETS="${SAEBENCH_DATASETS:-100_news_fake,105_click_bait,106_hate_hate,107_hate_offensive,110_aimade_humangpt3,113_movie_sent,114_nyc_borough_Manhattan,115_nyc_borough_Brooklyn,116_nyc_borough_Bronx,117_us_state_FL,118_us_state_CA,119_us_state_TX,120_us_timezone_Chicago,121_us_timezone_New_York,122_us_timezone_Los_Angeles,123_world_country_United_Kingdom}"
ASSIGN_UPDATE_INTERVAL="${ASSIGN_UPDATE_INTERVAL:-4}"
SUPERVISED_PROXY_MODE="${SUPERVISED_PROXY_MODE:-file_id}"
SUPERVISED_PROXY_NUM_CLASSES="${SUPERVISED_PROXY_NUM_CLASSES:-0}"

RUN_ID="run_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="results/experiments/cycle10_external_recovery/${RUN_ID}"
mkdir -p "$RUN_DIR"
LOG_PATH="$RUN_DIR/cycle10.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[cycle10] run_id=$RUN_ID"
echo "[cycle10] assignment_update_interval=$ASSIGN_UPDATE_INTERVAL"
echo "[cycle10] supervised_proxy_mode=$SUPERVISED_PROXY_MODE supervised_proxy_num_classes=$SUPERVISED_PROXY_NUM_CLASSES"

echo "[cycle10] waiting for conflicting experiment runners to finish"
has_conflicting_runner() {
  pgrep -f "^bash scripts/experiments/run_cycle9_novelty_push.sh" >/dev/null 2>&1 || \
  pgrep -f "^bash scripts/experiments/run_cycle8_robust_pareto_push.sh" >/dev/null 2>&1 || \
  pgrep -f "^bash scripts/experiments/run_cycle7_pareto_push.sh" >/dev/null 2>&1 || \
  pgrep -f "^bash scripts/experiments/run_cycle6_saeaware_push.sh" >/dev/null 2>&1 || \
  pgrep -f "^bash scripts/experiments/run_cycle5_external_push.sh" >/dev/null 2>&1 || \
  pgrep -f "^bash scripts/experiments/run_cycle4_followups_after_queue.sh" >/dev/null 2>&1 || \
  pgrep -f "^python scripts/experiments/run_architecture_frontier_external.py" >/dev/null 2>&1 || \
  pgrep -f "^python scripts/experiments/run_routed_frontier_external.py" >/dev/null 2>&1 || \
  pgrep -f "^python scripts/experiments/run_assignment_consistency_v3.py" >/dev/null 2>&1 || \
  pgrep -f "^python scripts/experiments/run_external_metric_scaling_study.py" >/dev/null 2>&1
}

while has_conflicting_runner; do
  date -u +"[cycle10] %Y-%m-%dT%H:%M:%SZ another runner active"
  sleep "$WAIT_SECONDS"
done

echo "[cycle10] git pull"
git pull origin main

default_baseline="docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json"

echo "[cycle10] stage1: routed external recovery sweep"
ROUTED_SWEEP_ROOT="results/experiments/phase4b_routed_frontier_external_sweep_cycle10_recovery"
mkdir -p "$ROUTED_SWEEP_ROOT"

declare -a ROUTED_RUNS=()
run_routed_condition() {
  local tag="$1"
  local dsae="$2"
  local k="$3"
  local experts="$4"
  local noise_std="$5"
  local consistency_coef="$6"
  local diversity_coef="$7"
  local lr="$8"
  local epochs="$9"

  echo "[cycle10][routed] condition=${tag} dsae=${dsae} k=${k} experts=${experts} noise=${noise_std} cons=${consistency_coef} div=${diversity_coef} lr=${lr} epochs=${epochs}"
  KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
  python scripts/experiments/run_routed_frontier_external.py \
    --activation-cache-dir "$ACTIVATION_CACHE_DIR" \
    --activation-glob '*_blocks.0.hook_resid_pre.pt' \
    --max-files 80 \
    --max-rows-per-file 2048 \
    --max-total-rows 150000 \
    --seeds 42,123,456,789 \
    --d-sae "$dsae" \
    --k "$k" \
    --epochs "$epochs" \
    --batch-size 4096 \
    --learning-rate "$lr" \
    --num-experts "$experts" \
    --route-balance-coef 0.02 \
    --route-entropy-coef 0.02 \
    --route-topk-mode expert_topk \
    --robust-noise-std "$noise_std" \
    --route-consistency-coef "$consistency_coef" \
    --decoder-diversity-coef "$diversity_coef" \
    --decoder-diversity-sample 1024 \
    --device cuda \
    --dtype float32 \
    --run-saebench \
    --run-cebench \
    --cebench-repo "$CEBENCH_REPO" \
    --cebench-max-rows 200 \
    --cebench-matched-baseline-summary "$default_baseline" \
    --saebench-datasets "$SAEBENCH_DATASETS" \
    --saebench-results-path /tmp/husai_saebench_probe_results_cycle10_routed \
    --saebench-model-cache-path "$SAEBENCH_MODEL_CACHE_PATH" \
    --cebench-artifacts-path /tmp/ce_bench_artifacts_cycle10_routed \
    --output-dir "$ROUTED_SWEEP_ROOT"

  local run_dir
  run_dir="$(ls -1dt "$ROUTED_SWEEP_ROOT"/run_* 2>/dev/null | head -n1 || true)"
  if [[ -n "$run_dir" ]]; then
    ROUTED_RUNS+=("$run_dir")
    echo "[cycle10][routed] done ${tag} -> $run_dir"
  fi
}

# Center around current best routed condition (cycle8 r4) with nearby variants.
run_routed_condition "c1" 1024 48 4 0.03 0.12 0.015 0.0008 12
run_routed_condition "c2" 1024 48 4 0.02 0.15 0.020 0.0008 12
run_routed_condition "c3" 1280 48 4 0.025 0.12 0.020 0.00075 12
run_routed_condition "c4" 1024 56 4 0.03 0.12 0.015 0.00075 12

echo "[cycle10] stage2: assignment-v3 supervised-proxy recovery sweep"
ASSIGN_SWEEP_ROOT="results/experiments/phase4d_assignment_consistency_v3_cycle10_recovery"
mkdir -p "$ASSIGN_SWEEP_ROOT"

declare -a ASSIGN_RUNS=()
run_assignment_condition() {
  local tag="$1"
  local dsae="$2"
  local k="$3"
  local epochs="$4"
  local lr="$5"
  local sup_weight="$6"

  echo "[cycle10][assignment] condition=${tag} dsae=${dsae} k=${k} epochs=${epochs} lr=${lr} sup_weight=${sup_weight}"
  KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
  python scripts/experiments/run_assignment_consistency_v3.py \
    --activation-cache-dir "$ACTIVATION_CACHE_DIR" \
    --activation-glob '*_blocks.0.hook_resid_pre.pt' \
    --max-files 80 \
    --max-rows-per-file 2048 \
    --max-total-rows 150000 \
    --d-sae "$dsae" \
    --k "$k" \
    --device cuda \
    --epochs "$epochs" \
    --batch-size 4096 \
    --learning-rate "$lr" \
    --assignment-update-interval "$ASSIGN_UPDATE_INTERVAL" \
    --supervised-proxy-mode "$SUPERVISED_PROXY_MODE" \
    --supervised-proxy-weight "$sup_weight" \
    --supervised-proxy-num-classes "$SUPERVISED_PROXY_NUM_CLASSES" \
    --train-seeds 123,456,789,1011 \
    --lambdas 0.0,0.02,0.04,0.06,0.08,0.1,0.15 \
    --run-saebench \
    --run-cebench \
    --cebench-repo "$CEBENCH_REPO" \
    --cebench-max-rows 200 \
    --cebench-matched-baseline-summary "$default_baseline" \
    --saebench-datasets "$SAEBENCH_DATASETS" \
    --saebench-results-path /tmp/husai_saebench_probe_results_cycle10_assignment \
    --saebench-model-cache-path "$SAEBENCH_MODEL_CACHE_PATH" \
    --cebench-artifacts-path /tmp/ce_bench_artifacts_cycle10_assignment \
    --external-checkpoint-policy external_score \
    --external-checkpoint-candidates-per-lambda 4 \
    --external-candidate-require-both \
    --external-candidate-min-saebench-delta -0.03 \
    --external-candidate-min-cebench-delta -35.5 \
    --external-candidate-weight-saebench 0.82 \
    --external-candidate-weight-cebench 0.10 \
    --external-candidate-weight-alignment 0.04 \
    --external-candidate-weight-ev 0.04 \
    --weight-internal-lcb 0.20 \
    --weight-ev 0.05 \
    --weight-saebench 0.60 \
    --weight-cebench 0.15 \
    --force-rerun-external \
    --require-external \
    --min-saebench-delta -0.005 \
    --min-cebench-delta -35.0 \
    --output-dir "$ASSIGN_SWEEP_ROOT"

  local run_dir
  run_dir="$(ls -1dt "$ASSIGN_SWEEP_ROOT"/run_* 2>/dev/null | head -n1 || true)"
  if [[ -n "$run_dir" ]]; then
    ASSIGN_RUNS+=("$run_dir")
    echo "[cycle10][assignment] done ${tag} -> $run_dir"
  fi
}

run_assignment_condition "s1" 2048 48 24 0.0005 0.05
run_assignment_condition "s2" 2048 48 24 0.0005 0.10
run_assignment_condition "s3" 3072 48 24 0.0004 0.10

echo "[cycle10] stage3: grouped-LCB selector"
FRONTIER_BASE="$(ls -1dt results/experiments/phase4b_architecture_frontier_external_multiseed/run_* 2>/dev/null | head -n1 || true)"
SCALING_RUN="$(ls -1dt results/experiments/phase4e_external_scaling_study_multiseed/run_* 2>/dev/null | head -n1 || true)"
if [[ -z "$FRONTIER_BASE" || ! -f "$FRONTIER_BASE/results.json" ]]; then
  echo "[cycle10][error] missing frontier base results"
  exit 2
fi
if [[ -z "$SCALING_RUN" || ! -f "$SCALING_RUN/results.json" ]]; then
  echo "[cycle10][error] missing scaling results"
  exit 2
fi

SELECTOR_ARGS=(
  --frontier-results "$FRONTIER_BASE/results.json"
  --scaling-results "$SCALING_RUN/results.json"
  --require-both-external
  --group-by-condition
  --uncertainty-mode lcb
  --min-seeds-per-group 4
  --weight-saebench 0.80
  --weight-cebench 0.15
  --weight-train-ev 0.05
  --output-dir results/experiments/release_candidate_selection_cycle10_recovery
)
for rr in "${ROUTED_RUNS[@]}"; do
  [[ -f "$rr/results.json" ]] && SELECTOR_ARGS+=(--frontier-results "$rr/results.json")
done
for ar in "${ASSIGN_RUNS[@]}"; do
  [[ -f "$ar/results.json" ]] && SELECTOR_ARGS+=(--assignment-results "$ar/results.json")
done
python scripts/experiments/select_release_candidate.py "${SELECTOR_ARGS[@]}"

SELECTOR_RUN="$(ls -1dt results/experiments/release_candidate_selection_cycle10_recovery/run_* 2>/dev/null | head -n1 || true)"
SELECTED_JSON="$SELECTOR_RUN/selected_candidate.json"
if [[ -z "$SELECTOR_RUN" || ! -f "$SELECTED_JSON" ]]; then
  echo "[cycle10][error] selector output missing selected_candidate.json"
  exit 2
fi

read -r BEST_CHECKPOINT BEST_ARCH BEST_HOOK_LAYER BEST_HOOK_NAME < <(
  python - "$SELECTED_JSON" <<'PY'
import json
import sys
obj = json.load(open(sys.argv[1]))
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
if [[ -z "$BEST_CHECKPOINT" || ! -f "$BEST_CHECKPOINT" ]]; then
  echo "[cycle10][error] selected checkpoint missing: $BEST_CHECKPOINT"
  exit 2
fi

echo "[cycle10] stage4: OOD + strict gate"
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_ood_stress_eval.py \
  --checkpoint "$BEST_CHECKPOINT" \
  --architecture "$BEST_ARCH" \
  --sae-release husai_cycle10_ood \
  --model-name pythia-70m-deduped \
  --hook-layer "$BEST_HOOK_LAYER" \
  --hook-name "$BEST_HOOK_NAME" \
  --device cuda \
  --dtype float32 \
  --results-path /tmp/husai_saebench_probe_results_ood_cycle10 \
  --model-cache-path "$SAEBENCH_MODEL_CACHE_PATH" \
  --output-dir results/experiments/phase4e_ood_stress_b200 \
  --force-rerun

OOD_SUMMARY="$(ls -1dt results/experiments/phase4e_ood_stress_b200/run_*/ood_stress_summary.json 2>/dev/null | head -n1 || true)"
TRANS_SWEEP_RUN="$(ls -1dt results/experiments/phase4e_transcoder_stress_sweep_b200/run_* 2>/dev/null | head -n1 || true)"
BEST_TRANSCODER_SUMMARY=""
if [[ -n "$TRANS_SWEEP_RUN" && -f "$TRANS_SWEEP_RUN/results.json" ]]; then
  BEST_TRANSCODER_SUMMARY="$(python - "$TRANS_SWEEP_RUN/results.json" <<'PY'
import json
import sys
obj = json.load(open(sys.argv[1]))
print((obj.get("best_condition") or {}).get("summary_path") or "")
PY
)"
  if [[ -n "$BEST_TRANSCODER_SUMMARY" && "$BEST_TRANSCODER_SUMMARY" != /* ]]; then
    BEST_TRANSCODER_SUMMARY="/workspace/HUSAI/$BEST_TRANSCODER_SUMMARY"
  fi
fi

set +e
python scripts/experiments/run_stress_gated_release_policy.py \
  --phase4a-results results/experiments/phase4a_trained_vs_random/results.json \
  --transcoder-results "$BEST_TRANSCODER_SUMMARY" \
  --ood-results "$OOD_SUMMARY" \
  --external-candidate-json "$SELECTED_JSON" \
  --external-mode joint \
  --use-external-lcb \
  --min-saebench-delta-lcb 0.0 \
  --min-cebench-delta-lcb 0.0 \
  --require-transcoder \
  --require-ood \
  --require-external
RELEASE_RC=$?
set -e

RELEASE_RUN="$(ls -1dt results/experiments/release_stress_gates/run_* 2>/dev/null | head -n1 || true)"

printf "%s\n" "${ROUTED_RUNS[@]}" > "$RUN_DIR/routed_runs.txt"
printf "%s\n" "${ASSIGN_RUNS[@]}" > "$RUN_DIR/assignment_runs.txt"

python - "$RUN_DIR/manifest.json" \
  "$RUN_ID" \
  "$FRONTIER_BASE" \
  "$SCALING_RUN" \
  "$SELECTOR_RUN" \
  "$SELECTED_JSON" \
  "$BEST_CHECKPOINT" \
  "$BEST_ARCH" \
  "$BEST_HOOK_LAYER" \
  "$BEST_HOOK_NAME" \
  "$OOD_SUMMARY" \
  "$BEST_TRANSCODER_SUMMARY" \
  "$RELEASE_RUN" \
  "$RELEASE_RC" <<'PY'
import json
import sys
from pathlib import Path

(
    manifest_path,
    run_id,
    frontier_base,
    scaling_run,
    selector_run,
    selected_candidate_json,
    best_checkpoint,
    best_arch,
    best_hook_layer,
    best_hook_name,
    ood_summary,
    best_transcoder_summary,
    release_run,
    release_rc,
) = sys.argv[1:]

manifest_path = Path(manifest_path)
run_dir = manifest_path.parent

def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]

payload = {
    "run_id": run_id,
    "routed_runs": read_lines(run_dir / "routed_runs.txt"),
    "assignment_runs": read_lines(run_dir / "assignment_runs.txt"),
    "frontier_base": frontier_base,
    "scaling_run": scaling_run,
    "selector_run": selector_run,
    "selected_candidate_json": selected_candidate_json,
    "best_checkpoint": best_checkpoint,
    "best_architecture": best_arch,
    "best_hook_layer": best_hook_layer,
    "best_hook_name": best_hook_name,
    "ood_summary": ood_summary,
    "transcoder_summary": best_transcoder_summary,
    "release_run": release_run,
    "release_rc": int(release_rc),
}
manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "[cycle10] complete"
echo "[cycle10] manifest=$RUN_DIR/manifest.json"

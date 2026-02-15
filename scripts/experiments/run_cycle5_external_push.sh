#!/usr/bin/env bash
set -euo pipefail

cd /workspace/HUSAI

WAIT_SECONDS="${WAIT_SECONDS:-60}"
CEBENCH_REPO="${CEBENCH_REPO:-/workspace/CE-Bench}"
ACTIVATION_CACHE_DIR="${ACTIVATION_CACHE_DIR:-/tmp/sae_bench_model_cache/model_activations_pythia-70m-deduped}"
SAEBENCH_MODEL_CACHE_PATH="${SAEBENCH_MODEL_CACHE_PATH:-/tmp/sae_bench_model_cache}"
SAEBENCH_DATASETS="${SAEBENCH_DATASETS:-100_news_fake,105_click_bait,106_hate_hate,107_hate_offensive,110_aimade_humangpt3,113_movie_sent,114_nyc_borough_Manhattan,115_nyc_borough_Brooklyn,116_nyc_borough_Bronx,117_us_state_FL,118_us_state_CA,119_us_state_TX,120_us_timezone_Chicago,121_us_timezone_New_York,122_us_timezone_Los_Angeles,123_world_country_United_Kingdom}"

RUN_ID="run_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="results/experiments/cycle5_external_push/${RUN_ID}"
mkdir -p "$RUN_DIR"
LOG_PATH="$RUN_DIR/cycle5.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[cycle5] run_id=$RUN_ID"
echo "[cycle5] waiting for conflicting experiment runners to finish"
while pgrep -f "run_cycle4_followups_after_queue.sh|run_architecture_frontier_external.py|run_routed_frontier_external.py|run_assignment_consistency_v3.py|run_external_metric_scaling_study.py" >/dev/null 2>&1; do
  date -u +"[cycle5] %Y-%m-%dT%H:%M:%SZ another runner active"
  sleep "$WAIT_SECONDS"
done

echo "[cycle5] git pull"
git pull origin main

echo "[cycle5] stage1: routed hyper-sweep with expert-topk mode"
ROUTED_SWEEP_ROOT="results/experiments/phase4b_routed_frontier_external_sweep"
mkdir -p "$ROUTED_SWEEP_ROOT"
default_baseline="docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json"

declare -a ROUTED_RUNS=()
run_routed_condition() {
  local tag="$1"
  local dsae="$2"
  local k="$3"
  local experts="$4"
  local bal="$5"
  local ent="$6"
  local lr="$7"
  local epochs="$8"
  local mode="$9"

  echo "[cycle5][routed] condition=${tag} dsae=${dsae} k=${k} experts=${experts} bal=${bal} ent=${ent} lr=${lr} epochs=${epochs} mode=${mode}"
  KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
  python scripts/experiments/run_routed_frontier_external.py \
    --activation-cache-dir "$ACTIVATION_CACHE_DIR" \
    --activation-glob '*_blocks.0.hook_resid_pre.pt' \
    --max-files 80 \
    --max-rows-per-file 2048 \
    --max-total-rows 150000 \
    --seeds 42,123,456 \
    --d-sae "$dsae" \
    --k "$k" \
    --epochs "$epochs" \
    --batch-size 4096 \
    --learning-rate "$lr" \
    --num-experts "$experts" \
    --route-balance-coef "$bal" \
    --route-entropy-coef "$ent" \
    --route-topk-mode "$mode" \
    --device cuda \
    --dtype float32 \
    --run-saebench \
    --run-cebench \
    --cebench-repo "$CEBENCH_REPO" \
    --cebench-max-rows 200 \
    --cebench-matched-baseline-summary "$default_baseline" \
    --saebench-datasets "$SAEBENCH_DATASETS" \
    --saebench-results-path /tmp/husai_saebench_probe_results_cycle5_routed \
    --saebench-model-cache-path "$SAEBENCH_MODEL_CACHE_PATH" \
    --cebench-artifacts-path /tmp/ce_bench_artifacts_cycle5_routed \
    --output-dir "$ROUTED_SWEEP_ROOT"

  local run_dir
  run_dir="$(ls -1dt "$ROUTED_SWEEP_ROOT"/run_* 2>/dev/null | head -n1 || true)"
  if [[ -n "$run_dir" ]]; then
    ROUTED_RUNS+=("$run_dir")
    echo "[cycle5][routed] done ${tag} -> $run_dir"
  fi
}

run_routed_condition "r1" 1024 32 8 0.05 0.01 0.001 8 expert_topk
run_routed_condition "r2" 1024 32 4 0.05 0.01 0.001 8 expert_topk
run_routed_condition "r3" 2048 32 8 0.05 0.01 0.001 8 expert_topk
run_routed_condition "r4" 2048 48 8 0.05 0.02 0.0007 10 expert_topk
run_routed_condition "r5" 1024 32 8 0.2 0.01 0.001 6 global_mask

echo "[cycle5] stage2: assignment-v3 external-aware sweep"
ASSIGN_SWEEP_ROOT="results/experiments/phase4d_assignment_consistency_v3_external_sweep"
mkdir -p "$ASSIGN_SWEEP_ROOT"

declare -a ASSIGN_RUNS=()
run_assignment_condition() {
  local tag="$1"
  local dsae="$2"
  local k="$3"
  local epochs="$4"
  local lr="$5"

  echo "[cycle5][assignment] condition=${tag} dsae=${dsae} k=${k} epochs=${epochs} lr=${lr}"
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
    --train-seeds 123,456,789,1011 \
    --lambdas 0.0,0.05,0.1,0.2,0.3,0.4 \
    --run-saebench \
    --run-cebench \
    --cebench-repo "$CEBENCH_REPO" \
    --cebench-max-rows 200 \
    --cebench-matched-baseline-summary "$default_baseline" \
    --saebench-datasets "$SAEBENCH_DATASETS" \
    --saebench-results-path /tmp/husai_saebench_probe_results_cycle5_assignment \
    --saebench-model-cache-path "$SAEBENCH_MODEL_CACHE_PATH" \
    --cebench-artifacts-path /tmp/ce_bench_artifacts_cycle5_assignment \
    --force-rerun-external \
    --require-external \
    --min-saebench-delta 0.0 \
    --min-cebench-delta 0.0 \
    --output-dir "$ASSIGN_SWEEP_ROOT"

  local run_dir
  run_dir="$(ls -1dt "$ASSIGN_SWEEP_ROOT"/run_* 2>/dev/null | head -n1 || true)"
  if [[ -n "$run_dir" ]]; then
    ASSIGN_RUNS+=("$run_dir")
    echo "[cycle5][assignment] done ${tag} -> $run_dir"
  fi
}

run_assignment_condition "a1" 1024 32 16 0.001
run_assignment_condition "a2" 2048 32 16 0.0007

echo "[cycle5] stage3: grouped LCB reselection with assignment integration"
FRONTIER_BASE="$(ls -1dt results/experiments/phase4b_architecture_frontier_external_multiseed/run_* 2>/dev/null | head -n1 || true)"
SCALING_RUN="$(ls -1dt results/experiments/phase4e_external_scaling_study_multiseed/run_* 2>/dev/null | head -n1 || true)"

SELECTOR_ARGS=(
  --frontier-results "$FRONTIER_BASE/results.json"
  --scaling-results "$SCALING_RUN/results.json"
  --require-both-external
  --group-by-condition
  --uncertainty-mode lcb
  --min-seeds-per-group 3
  --output-dir results/experiments/release_candidate_selection_cycle5
)
for rr in "${ROUTED_RUNS[@]}"; do
  [[ -f "$rr/results.json" ]] && SELECTOR_ARGS+=(--frontier-results "$rr/results.json")
done
for ar in "${ASSIGN_RUNS[@]}"; do
  [[ -f "$ar/results.json" ]] && SELECTOR_ARGS+=(--assignment-results "$ar/results.json")
done
python scripts/experiments/select_release_candidate.py "${SELECTOR_ARGS[@]}"

SELECTOR_RUN="$(ls -1dt results/experiments/release_candidate_selection_cycle5/run_* 2>/dev/null | head -n1 || true)"
SELECTED_JSON="$SELECTOR_RUN/selected_candidate.json"

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

echo "[cycle5] stage4: OOD stress + strict release gate"
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_ood_stress_eval.py \
  --checkpoint "$BEST_CHECKPOINT" \
  --architecture "$BEST_ARCH" \
  --sae-release husai_cycle5_ood \
  --model-name pythia-70m-deduped \
  --hook-layer "$BEST_HOOK_LAYER" \
  --hook-name "$BEST_HOOK_NAME" \
  --device cuda \
  --dtype float32 \
  --results-path /tmp/husai_saebench_probe_results_ood_cycle5 \
  --model-cache-path "$SAEBENCH_MODEL_CACHE_PATH" \
  --output-dir results/experiments/phase4e_ood_stress_b200 \
  --force-rerun

OOD_SUMMARY="$(ls -1dt results/experiments/phase4e_ood_stress_b200/run_*/ood_stress_summary.json 2>/dev/null | head -n1 || true)"
TRANS_SWEEP_RUN="$(ls -1dt results/experiments/phase4e_transcoder_stress_sweep_b200/run_* 2>/dev/null | head -n1 || true)"
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

python - "$RUN_DIR/manifest.json" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
run_dir = manifest_path.parent


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]

payload = {
    "run_id": os.environ.get("RUN_ID"),
    "routed_runs": read_lines(run_dir / "routed_runs.txt"),
    "assignment_runs": read_lines(run_dir / "assignment_runs.txt"),
    "frontier_base": os.environ.get("FRONTIER_BASE"),
    "scaling_run": os.environ.get("SCALING_RUN"),
    "selector_run": os.environ.get("SELECTOR_RUN"),
    "selected_candidate_json": os.environ.get("SELECTED_JSON"),
    "best_checkpoint": os.environ.get("BEST_CHECKPOINT"),
    "best_architecture": os.environ.get("BEST_ARCH"),
    "best_hook_layer": os.environ.get("BEST_HOOK_LAYER"),
    "best_hook_name": os.environ.get("BEST_HOOK_NAME"),
    "ood_summary": os.environ.get("OOD_SUMMARY"),
    "transcoder_summary": os.environ.get("BEST_TRANSCODER_SUMMARY"),
    "release_run": os.environ.get("RELEASE_RUN"),
    "release_rc": int(os.environ.get("RELEASE_RC", "1")),
}
manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "[cycle5] complete"
echo "[cycle5] manifest=$RUN_DIR/manifest.json"

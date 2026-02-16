#!/usr/bin/env bash
set -euo pipefail

cd /workspace/HUSAI

WAIT_SECONDS="${WAIT_SECONDS:-60}"
CEBENCH_REPO="${CEBENCH_REPO:-/workspace/CE-Bench}"
ACTIVATION_CACHE_DIR="${ACTIVATION_CACHE_DIR:-/tmp/sae_bench_model_cache/model_activations_pythia-70m-deduped}"
SAEBENCH_MODEL_CACHE_PATH="${SAEBENCH_MODEL_CACHE_PATH:-/tmp/sae_bench_model_cache}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
SAEBENCH_DATASETS="${SAEBENCH_DATASETS:-100_news_fake,105_click_bait,106_hate_hate,107_hate_offensive,110_aimade_humangpt3,113_movie_sent,114_nyc_borough_Manhattan,115_nyc_borough_Brooklyn,116_nyc_borough_Bronx,117_us_state_FL,118_us_state_CA,119_us_state_TX,120_us_timezone_Chicago,121_us_timezone_New_York,122_us_timezone_Los_Angeles,123_world_country_United_Kingdom}"

RUN_ID="run_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="results/experiments/cycle7_pareto_push/${RUN_ID}"
mkdir -p "$RUN_DIR"
LOG_PATH="$RUN_DIR/cycle7.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[cycle7] run_id=$RUN_ID"
echo "[cycle7] waiting for conflicting experiment runners to finish"
while pgrep -f "run_cycle6_saeaware_push.sh|run_cycle5_external_push.sh|run_cycle4_followups_after_queue.sh|run_architecture_frontier_external.py|run_routed_frontier_external.py|run_assignment_consistency_v3.py|run_external_metric_scaling_study.py" >/dev/null 2>&1; do
  date -u +"[cycle7] %Y-%m-%dT%H:%M:%SZ another runner active"
  sleep "$WAIT_SECONDS"
done

echo "[cycle7] git pull"
git pull origin main

echo "[cycle7] stage1: routed Pareto-zone sweep (SAEBench/CE tradeoff)"
ROUTED_SWEEP_ROOT="results/experiments/phase4b_routed_frontier_external_sweep_cycle7_pareto"
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

  echo "[cycle7][routed] condition=${tag} dsae=${dsae} k=${k} experts=${experts} bal=${bal} ent=${ent} lr=${lr} epochs=${epochs}"
  KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
  python scripts/experiments/run_routed_frontier_external.py \
    --activation-cache-dir "$ACTIVATION_CACHE_DIR" \
    --activation-glob '*_blocks.0.hook_resid_pre.pt' \
    --max-files 80 \
    --max-rows-per-file 2048 \
    --max-total-rows 150000 \
    --seeds 42,123,456,789,1011 \
    --d-sae "$dsae" \
    --k "$k" \
    --epochs "$epochs" \
    --batch-size 4096 \
    --learning-rate "$lr" \
    --num-experts "$experts" \
    --route-balance-coef "$bal" \
    --route-entropy-coef "$ent" \
    --route-topk-mode expert_topk \
    --device cuda \
    --dtype float32 \
    --run-saebench \
    --run-cebench \
    --cebench-repo "$CEBENCH_REPO" \
    --cebench-max-rows 200 \
    --cebench-matched-baseline-summary "$default_baseline" \
    --saebench-datasets "$SAEBENCH_DATASETS" \
    --saebench-results-path /tmp/husai_saebench_probe_results_cycle7_routed \
    --saebench-model-cache-path "$SAEBENCH_MODEL_CACHE_PATH" \
    --cebench-artifacts-path /tmp/ce_bench_artifacts_cycle7_routed \
    --output-dir "$ROUTED_SWEEP_ROOT"

  local run_dir
  run_dir="$(ls -1dt "$ROUTED_SWEEP_ROOT"/run_* 2>/dev/null | head -n1 || true)"
  if [[ -n "$run_dir" ]]; then
    ROUTED_RUNS+=("$run_dir")
    echo "[cycle7][routed] done ${tag} -> $run_dir"
  fi
}

# Best CE anchor from cycle5
a="p1"; run_routed_condition "$a" 1024 32 4 0.02 0.02 0.0010 10
# Best SAEBench anchor from cycle5
a="p2"; run_routed_condition "$a" 2048 48 8 0.02 0.02 0.0007 12
# Pareto interpolation points
a="p3"; run_routed_condition "$a" 1536 40 6 0.02 0.02 0.0008 12
a="p4"; run_routed_condition "$a" 2048 40 6 0.02 0.02 0.0007 12
a="p5"; run_routed_condition "$a" 1024 48 4 0.02 0.02 0.0008 12

echo "[cycle7] stage2: assignment-v3 SAEBench-prioritized sweep"
ASSIGN_SWEEP_ROOT="results/experiments/phase4d_assignment_consistency_v3_cycle7_pareto"
mkdir -p "$ASSIGN_SWEEP_ROOT"

declare -a ASSIGN_RUNS=()
run_assignment_condition() {
  local tag="$1"
  local dsae="$2"
  local k="$3"
  local epochs="$4"
  local lr="$5"

  echo "[cycle7][assignment] condition=${tag} dsae=${dsae} k=${k} epochs=${epochs} lr=${lr}"
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
    --train-seeds 123,456,789,1011,1213,1415 \
    --lambdas 0.0,0.03,0.05,0.08,0.1,0.15,0.2,0.3 \
    --run-saebench \
    --run-cebench \
    --cebench-repo "$CEBENCH_REPO" \
    --cebench-max-rows 200 \
    --cebench-matched-baseline-summary "$default_baseline" \
    --saebench-datasets "$SAEBENCH_DATASETS" \
    --saebench-results-path /tmp/husai_saebench_probe_results_cycle7_assignment \
    --saebench-model-cache-path "$SAEBENCH_MODEL_CACHE_PATH" \
    --cebench-artifacts-path /tmp/ce_bench_artifacts_cycle7_assignment \
    --external-checkpoint-policy external_score \
    --external-checkpoint-candidates-per-lambda 4 \
    --external-candidate-require-both \
    --external-candidate-min-saebench-delta -0.05 \
    --external-candidate-min-cebench-delta -36.0 \
    --external-candidate-weight-saebench 0.70 \
    --external-candidate-weight-cebench 0.20 \
    --external-candidate-weight-alignment 0.05 \
    --external-candidate-weight-ev 0.05 \
    --weight-internal-lcb 0.25 \
    --weight-ev 0.10 \
    --weight-saebench 0.45 \
    --weight-cebench 0.20 \
    --force-rerun-external \
    --require-external \
    --min-saebench-delta -0.03 \
    --min-cebench-delta -36.0 \
    --output-dir "$ASSIGN_SWEEP_ROOT"

  local run_dir
  run_dir="$(ls -1dt "$ASSIGN_SWEEP_ROOT"/run_* 2>/dev/null | head -n1 || true)"
  if [[ -n "$run_dir" ]]; then
    ASSIGN_RUNS+=("$run_dir")
    echo "[cycle7][assignment] done ${tag} -> $run_dir"
  fi
}

run_assignment_condition "a1" 2048 32 20 0.0007
run_assignment_condition "a2" 2048 48 20 0.0006
run_assignment_condition "a3" 3072 48 24 0.0005

echo "[cycle7] stage3: grouped LCB reselection"
FRONTIER_BASE="$(ls -1dt results/experiments/phase4b_architecture_frontier_external_multiseed/run_* 2>/dev/null | head -n1 || true)"
SCALING_RUN="$(ls -1dt results/experiments/phase4e_external_scaling_study_multiseed/run_* 2>/dev/null | head -n1 || true)"
if [[ -z "$FRONTIER_BASE" || ! -f "$FRONTIER_BASE/results.json" ]]; then
  echo "[cycle7][error] missing frontier base results (phase4b_architecture_frontier_external_multiseed)"
  exit 2
fi
if [[ -z "$SCALING_RUN" || ! -f "$SCALING_RUN/results.json" ]]; then
  echo "[cycle7][error] missing scaling results (phase4e_external_scaling_study_multiseed)"
  exit 2
fi

SELECTOR_ARGS=(
  --frontier-results "$FRONTIER_BASE/results.json"
  --scaling-results "$SCALING_RUN/results.json"
  --require-both-external
  --group-by-condition
  --uncertainty-mode lcb
  --min-seeds-per-group 4
  --weight-saebench 0.70
  --weight-cebench 0.25
  --weight-train-ev 0.05
  --output-dir results/experiments/release_candidate_selection_cycle7
)
for rr in "${ROUTED_RUNS[@]}"; do
  [[ -f "$rr/results.json" ]] && SELECTOR_ARGS+=(--frontier-results "$rr/results.json")
done
for ar in "${ASSIGN_RUNS[@]}"; do
  [[ -f "$ar/results.json" ]] && SELECTOR_ARGS+=(--assignment-results "$ar/results.json")
done
python scripts/experiments/select_release_candidate.py "${SELECTOR_ARGS[@]}"

SELECTOR_RUN="$(ls -1dt results/experiments/release_candidate_selection_cycle7/run_* 2>/dev/null | head -n1 || true)"
SELECTED_JSON="$SELECTOR_RUN/selected_candidate.json"
if [[ -z "$SELECTOR_RUN" || ! -f "$SELECTED_JSON" ]]; then
  echo "[cycle7][error] selector output missing selected_candidate.json"
  exit 2
fi

SELECTOR_ARGS_MIN3=(
  --frontier-results "$FRONTIER_BASE/results.json"
  --scaling-results "$SCALING_RUN/results.json"
  --require-both-external
  --group-by-condition
  --uncertainty-mode lcb
  --min-seeds-per-group 3
  --weight-saebench 0.70
  --weight-cebench 0.25
  --weight-train-ev 0.05
  --output-dir results/experiments/release_candidate_selection_cycle7_min3
)
for rr in "${ROUTED_RUNS[@]}"; do
  [[ -f "$rr/results.json" ]] && SELECTOR_ARGS_MIN3+=(--frontier-results "$rr/results.json")
done
for ar in "${ASSIGN_RUNS[@]}"; do
  [[ -f "$ar/results.json" ]] && SELECTOR_ARGS_MIN3+=(--assignment-results "$ar/results.json")
done
python scripts/experiments/select_release_candidate.py "${SELECTOR_ARGS_MIN3[@]}"

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
  echo "[cycle7][error] selected checkpoint missing: $BEST_CHECKPOINT"
  exit 2
fi

echo "[cycle7] stage4: OOD + strict release gate"
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_ood_stress_eval.py \
  --checkpoint "$BEST_CHECKPOINT" \
  --architecture "$BEST_ARCH" \
  --sae-release husai_cycle7_ood \
  --model-name pythia-70m-deduped \
  --hook-layer "$BEST_HOOK_LAYER" \
  --hook-name "$BEST_HOOK_NAME" \
  --device cuda \
  --dtype float32 \
  --results-path /tmp/husai_saebench_probe_results_ood_cycle7 \
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

echo "[cycle7] complete"
echo "[cycle7] manifest=$RUN_DIR/manifest.json"

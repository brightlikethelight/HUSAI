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
  echo "[queue] WARNING: frontier run has no results.json yet; proceeding with available external summaries"
fi

echo "[queue] launching multiseed external scaling study"
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
  --cebench-matched-baseline-summary docs/evidence/phase4e_cebench_matched200/cebench_matched200_summary.json \
  --saebench-datasets "$SAEBENCH_DATASETS" \
  --saebench-results-path /tmp/husai_saebench_probe_results_scaling_multiseed \
  --saebench-model-cache-path /tmp/sae_bench_model_cache \
  --cebench-artifacts-path /tmp/ce_bench_artifacts_scaling_multiseed \
  --device cuda \
  --dtype float32 \
  --output-dir results/experiments/phase4e_external_scaling_study_multiseed

SCALING_RUN="$(ls -1dt results/experiments/phase4e_external_scaling_study_multiseed/run_* | head -n1)"
echo "[queue] scaling_run=${SCALING_RUN}"

echo "[queue] selecting best external condition from frontier summaries"
python - <<'PY'
import glob
import json
from pathlib import Path

frontier_run = Path(glob.glob("results/experiments/phase4b_architecture_frontier_external_multiseed/run_*")[-1])
summaries = sorted(frontier_run.glob("external_eval/*/saebench/husai_custom_sae_summary.json"))
if not summaries:
    raise SystemExit("No frontier SAEBench summaries found")

best = None
for p in summaries:
    payload = json.loads(p.read_text())
    score = (payload.get("summary") or {}).get("best_minus_llm_auc")
    if score is None:
        continue
    condition = p.parts[-3]
    arch = condition.split("_seed")[0]
    ckpt = frontier_run / "checkpoints" / condition / "sae_final.pt"
    row = {
        "condition": condition,
        "architecture": arch,
        "score": float(score),
        "summary_path": str(p),
        "checkpoint_path": str(ckpt),
        "frontier_run": str(frontier_run),
    }
    if best is None or row["score"] > best["score"]:
        best = row

if best is None:
    raise SystemExit("No valid best_minus_llm_auc found in frontier summaries")

out_path = Path("results/experiments/cycle3_queue") / "latest_external_candidate.json"
out_path.write_text(json.dumps(best, indent=2) + "\n")
print(f"[queue] best_condition={best['condition']} score={best['score']}")
print(f"[queue] candidate_json={out_path}")
PY

CANDIDATE_JSON="results/experiments/cycle3_queue/latest_external_candidate.json"
BEST_CHECKPOINT="$(python -c 'import json;print(json.load(open("results/experiments/cycle3_queue/latest_external_candidate.json"))["checkpoint_path"])')"
BEST_ARCH="$(python -c 'import json;print(json.load(open("results/experiments/cycle3_queue/latest_external_candidate.json"))["architecture"])')"
BEST_SUMMARY="$(python -c 'import json;print(json.load(open("results/experiments/cycle3_queue/latest_external_candidate.json"))["summary_path"])')"

echo "[queue] best_checkpoint=${BEST_CHECKPOINT}"
echo "[queue] best_architecture=${BEST_ARCH}"
echo "[queue] best_summary=${BEST_SUMMARY}"

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

echo "[queue] launching OOD stress eval on best frontier candidate"
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/tmp/mpl \
python scripts/experiments/run_ood_stress_eval.py \
  --checkpoint "$BEST_CHECKPOINT" \
  --architecture "$BEST_ARCH" \
  --sae-release husai_cycle3_ood \
  --model-name pythia-70m-deduped \
  --hook-layer 0 \
  --hook-name blocks.0.hook_resid_pre \
  --device cuda \
  --dtype float32 \
  --results-path /tmp/husai_saebench_probe_results_ood_cycle3 \
  --model-cache-path /tmp/sae_bench_model_cache \
  --output-dir results/experiments/phase4e_ood_stress_b200 \
  --force-rerun

OOD_SUMMARY="$(ls -1dt results/experiments/phase4e_ood_stress_b200/run_*/ood_stress_summary.json | head -n1)"
echo "[queue] ood_summary=${OOD_SUMMARY}"

echo "[queue] evaluating strict release policy"
set +e
python scripts/experiments/run_stress_gated_release_policy.py \
  --phase4a-results results/experiments/phase4a_trained_vs_random/results.json \
  --transcoder-results "$TRANSCODER_SUMMARY" \
  --ood-results "$OOD_SUMMARY" \
  --external-summary "$BEST_SUMMARY" \
  --require-transcoder \
  --require-ood \
  --require-external \
  --fail-on-gate-fail
RELEASE_RC=$?
set -e

echo "[queue] release_policy_rc=${RELEASE_RC}"

cat > "$QUEUE_DIR/manifest.json" <<MANIFEST
{
  "queue_id": "${QUEUE_ID}",
  "frontier_run": "${FRONTIER_RUN}",
  "scaling_run": "${SCALING_RUN}",
  "candidate_json": "${CANDIDATE_JSON}",
  "transcoder_summary": "${TRANSCODER_SUMMARY}",
  "ood_summary": "${OOD_SUMMARY}",
  "best_external_summary": "${BEST_SUMMARY}",
  "release_policy_rc": ${RELEASE_RC}
}
MANIFEST

echo "[queue] complete"
echo "[queue] manifest=${QUEUE_DIR}/manifest.json"

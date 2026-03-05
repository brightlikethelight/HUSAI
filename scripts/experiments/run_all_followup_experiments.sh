#!/usr/bin/env bash
# Run all follow-up experiments for the HUSAI stability paper.
#
# Priority order (from the research plan):
#   1. exp_1layer_ground_truth   (1.2) -- Low effort, high payoff
#   2. exp_subspace_stability    (2.2) -- Low effort, novel angle
#   3. exp_effective_rank_predictor (2.3) -- Needs 1.2 output
#   4. exp_contrastive_stability (1.3) -- Medium effort
#   5. exp_intervention_stability (2.1) -- Medium effort
#   6. exp_dictionary_pinning    (2.4) -- Low effort
#   7. exp_pythia70m_stability   (1.1) -- Needs GPU / patience
#
# Usage:
#   bash scripts/experiments/run_all_followup_experiments.sh [--skip-pythia]
#
# Each experiment saves results to results/experiments/<name>/run_<UTC>/
# and figures to figures/

set -euo pipefail
export KMP_DUPLICATE_LIB_OK=TRUE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

SKIP_PYTHIA=false
for arg in "$@"; do
    case "$arg" in
        --skip-pythia) SKIP_PYTHIA=true ;;
    esac
done

echo "================================================================="
echo "HUSAI Follow-Up Experiments Runner"
echo "================================================================="
echo "Project root: $PROJECT_ROOT"
echo "Skip Pythia:  $SKIP_PYTHIA"
echo "Started:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

run_experiment() {
    local name="$1"
    local script="$2"
    echo "-----------------------------------------------------------------"
    echo "Running: $name"
    echo "Script:  $script"
    echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "-----------------------------------------------------------------"

    if python "$script"; then
        echo "PASSED: $name"
    else
        echo "FAILED: $name (exit code $?)"
    fi
    echo ""
}

# 1. 1-Layer Ground Truth (Experiment 1.2)
run_experiment "1-Layer Ground Truth (1.2)" \
    "scripts/experiments/exp_1layer_ground_truth.py"

# 2. Subspace Stability (Experiment 2.2)
run_experiment "Subspace Stability (2.2)" \
    "scripts/experiments/exp_subspace_stability.py"

# 3. Effective Rank Predictor (Experiment 2.3)
run_experiment "Effective Rank Predictor (2.3)" \
    "scripts/experiments/exp_effective_rank_predictor.py"

# 4. Contrastive Stability (Experiment 1.3)
run_experiment "Contrastive Stability (1.3)" \
    "scripts/experiments/exp_contrastive_stability.py"

# 5. Intervention Stability (Experiment 2.1)
run_experiment "Intervention Stability (2.1)" \
    "scripts/experiments/exp_intervention_stability.py"

# 6. Dictionary Pinning (Experiment 2.4)
run_experiment "Dictionary Pinning (2.4)" \
    "scripts/experiments/exp_dictionary_pinning.py"

# 7. Pythia-70M Stability (Experiment 1.1) -- optional, GPU-intensive
if [ "$SKIP_PYTHIA" = false ]; then
    run_experiment "Pythia-70M Stability (1.1)" \
        "scripts/experiments/exp_pythia70m_stability.py"
else
    echo "SKIPPED: Pythia-70M Stability (1.1) [--skip-pythia flag]"
fi

echo ""
echo "================================================================="
echo "All experiments complete!"
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "Results:  results/experiments/"
echo "Figures:  figures/"
echo "================================================================="

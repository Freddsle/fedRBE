#!/usr/bin/env bash
# Runs the random forest evaluations and Figure 4 in a detached screen session,
# one step after the other:
#   1. 70:30 train/test split (run_classification_train_test_split.py)
#   2. leave-one-cohort-out (run_classification_leave_one_cohort_out.py)
#   3. Figure 4, S7 and S8 for both correction arms (evaluation/aggregate_figure4.R)
# The run stops at the first failing step. Start it from anywhere:
#   bash evaluation_classification_after_correction/run_rf_evaluations_screen.sh
# Extra arguments go to both evaluations, e.g. --datasets ecoli --seeds 42.
# All output goes to evaluation_classification_after_correction/rf_evaluations_<timestamp>.log.

# Units (dataset and seed, or dataset and held-out cohort) run at the same time
N_JOBS=8
# Recompute results that are already in the results file. Needed after changing
# the correction or the forest: cached rows do not record these settings.
FORCE_RUN=true

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

step() {
    echo ""
    echo "[$(date '+%F %T')] $1"
}

if [[ "${1:-}" != "--in-screen" ]]; then
    if ! command -v screen > /dev/null; then
        echo "Error: screen is not installed" >&2
        exit 1
    fi
    timestamp="$(date +%Y%m%d_%H%M%S)"
    session="rf_evaluations_${timestamp}"
    log="${SCRIPT_DIR}/rf_evaluations_${timestamp}.log"
    screen -dmS "$session" bash -c \
        "bash $(printf '%q' "${BASH_SOURCE[0]}") --in-screen $(printf '%q ' "$@") > $(printf '%q' "$log") 2>&1"
    echo "Started screen session ${session}"
    echo "Log:    ${log}"
    echo "Follow: tail -f ${log}"
    echo "Attach: screen -r ${session} (detach with Ctrl-A, D)"
    exit 0
fi
shift

export PYTHONUNBUFFERED=1
force=()
if [[ "$FORCE_RUN" == true ]]; then
    force=(--force-run)
fi

step "Settings: N_JOBS=${N_JOBS}, FORCE_RUN=${FORCE_RUN}, extra arguments: ${*:-none}"

step "1/3 Train/test split evaluation"
pixi run python -u evaluation_classification_after_correction/run_classification_train_test_split.py \
    --jobs "$N_JOBS" "${force[@]}" "$@"
step "1/3 done"

step "2/3 Leave-one-cohort-out evaluation"
pixi run python -u evaluation_classification_after_correction/run_classification_leave_one_cohort_out.py \
    --jobs "$N_JOBS" "${force[@]}" "$@"
step "2/3 done"

step "3/3 Figure 4, S7 and S8 for both correction arms"
env -u CORRECTION_ARM pixi run Rscript evaluation/aggregate_figure4.R
step "3/3 done, all steps completed"

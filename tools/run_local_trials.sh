#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/k3d_guard.sh"

VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
TRIALS=5
DURATION_MIN=30
SKIP_BUILD=false
OUTPUT_ROOT="$ROOT_DIR/docs/Final Results/trials"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --trials N        Number of repeated trials (default: 5)
  --duration N      Duration in minutes for each run (default: 30)
  --output-root DIR Root output directory (default: docs/Final Results/trials)
  --skip-build      Skip docker image rebuild in run scripts
  -h, --help        Show this help

Runs baseline, QMIX, and HPA sequentially for each trial and then generates
summary statistics from all generated JSON files.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --duration)
            DURATION_MIN="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if ! [[ "$TRIALS" =~ ^[0-9]+$ ]] || (( TRIALS < 1 )); then
    echo "[ERROR] --trials must be an integer >= 1"
    exit 1
fi

if ! [[ "$DURATION_MIN" =~ ^[0-9]+$ ]] || (( DURATION_MIN < 2 )); then
    echo "[ERROR] --duration must be an integer >= 2"
    exit 1
fi

if [[ "$OUTPUT_ROOT" != /* ]]; then
    OUTPUT_ROOT="$ROOT_DIR/$OUTPUT_ROOT"
fi

assert_k3d_context

RUN_ID="local_trials_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

echo "============================================================"
echo "AURA local multi-trial benchmark"
echo "============================================================"
echo "Run ID:           $RUN_ID"
echo "Trials:           $TRIALS"
echo "Duration/run:     ${DURATION_MIN} minutes"
echo "Output directory: $RUN_DIR"
echo "Build strategy:   $( [[ "$SKIP_BUILD" == "true" ]] && echo "skip all rebuilds" || echo "build first trial, skip after" )"
echo ""

for trial_idx in $(seq 1 "$TRIALS"); do
    trial_name=$(printf "trial_%02d" "$trial_idx")
    trial_dir="$RUN_DIR/$trial_name"
    mkdir -p "$trial_dir"

    echo "------------------------------------------------------------"
    echo "Starting $trial_name"
    echo "------------------------------------------------------------"

    BUILD_ARG=()
    if [[ "$SKIP_BUILD" == "true" || "$trial_idx" -gt 1 ]]; then
        BUILD_ARG+=(--skip-build)
    fi

    echo "[1/3] Baseline"
    bash "$SCRIPT_DIR/run_baseline_test.sh" \
        --duration "$DURATION_MIN" \
        --output-dir "$trial_dir/baseline" \
        "${BUILD_ARG[@]}"

    echo "[2/3] QMIX"
    bash "$SCRIPT_DIR/run_qmix_test.sh" \
        --duration "$DURATION_MIN" \
        --output-dir "$trial_dir/qmix" \
        "${BUILD_ARG[@]}"

    echo "[3/3] HPA"
    bash "$SCRIPT_DIR/run_hpa_test.sh" \
        --duration "$DURATION_MIN" \
        --output-dir "$trial_dir/hpa" \
        "${BUILD_ARG[@]}"
done

SUMMARY_TXT="$RUN_DIR/summary.txt"

echo ""
echo "Generating statistics summary..."
"$VENV_PYTHON" "$SCRIPT_DIR/summarize_trials.py" --root "$RUN_DIR" | tee "$SUMMARY_TXT"

echo ""
echo "Completed all trials."
echo "Summary: $SUMMARY_TXT"

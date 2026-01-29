#!/usr/bin/env bash

################################################################################
# APEX test launcher (Pong) - PBER + RASPBERry
#
# Runs 2 APEX variants per GPU:
#   1) APEX-PBER
#   2) APEX-RASPBERry
#
# Usage:
#   ./run_apex_test_pong.sh                # default GPU 0
#   ./run_apex_test_pong.sh -n 0,1         # comma-separated GPU list
#   ./run_apex_test_pong.sh -m pber        # run PBER only
#   ./run_apex_test_pong.sh -m raspberry   # run RASPBERry only
#
################################################################################

set -euo pipefail

GPU_LIST_ARG="0"
MODE="both"
LAUNCH_DELAY_BETWEEN_GPUS=10
LAUNCH_DELAY_SAME_GPU=30

while getopts "n:m:h" opt; do
    case $opt in
        n) GPU_LIST_ARG="$OPTARG" ;;
        m) MODE="$OPTARG" ;;
        h)
            echo "Usage: $0 [-n GPU_IDS] [-m MODE] [-h]"
            echo "Options: -n GPU_IDS (default: 0)"
            echo "         -m MODE (both|pber|raspberry; default: both)"
            echo "Example: -n 0,1"
            exit 0
            ;;
        \?) echo "Invalid option" >&2; exit 1 ;;
    esac
done

if ! [[ "${GPU_LIST_ARG}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Error: -n only supports comma-separated GPU IDs (e.g., 0,1,2)" >&2
    exit 1
fi

IFS=',' read -ra GPU_IDS <<< "${GPU_LIST_ARG}"

if [ ${#GPU_IDS[@]} -eq 0 ]; then
    echo "Error: at least one GPU ID is required" >&2
    exit 1
fi

for gpu_id in "${GPU_IDS[@]}"; do
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
        echo "Error: GPU ID must be a non-negative integer (got: ${gpu_id})" >&2
        exit 1
    fi
done

case "${MODE}" in
    both|pber|raspberry) ;;
    *)
        echo "Error: -m must be one of: both, pber, raspberry (got: ${MODE})" >&2
        exit 1
        ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_LOG_DIR="/home/seventheli/research/RASPBERry/logs/test"
mkdir -p "${SCRIPT_LOG_DIR}"

ENV_NAME="Atari-PongNoFrameskip-v4"
PBER_CONFIG="configs/apex_pber_atari.yml"
RASP_CONFIG="configs/apex_raspberry_atari.yml"

echo "================================================================================"
echo "APEX test (Pong) | env: ${ENV_NAME}"
echo "Target GPUs: ${GPU_IDS[*]}"
case "${MODE}" in
    both) MODE_DESC="PBER -> RASPBERry sequential per GPU" ;;
    pber) MODE_DESC="PBER only" ;;
    raspberry) MODE_DESC="RASPBERry only" ;;
esac
echo "Mode: ${MODE_DESC}"
echo "Log dir: ${SCRIPT_LOG_DIR}"
if [ "${MODE}" = "both" ]; then
    TASKS_PER_GPU=2
else
    TASKS_PER_GPU=1
fi
echo "Total tasks: $(( ${#GPU_IDS[@]} * TASKS_PER_GPU ))"
echo "================================================================================"

declare -a ALL_PIDS
declare -a ALL_NAMES

for idx in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[$idx]}"
    echo "GPU ${gpu}: Pong APEX test group $((idx + 1))"
    log_suffix="pong_gpu${gpu}_${TIMESTAMP}"
    task_idx=0
    task_total=${TASKS_PER_GPU}

    if [ "${MODE}" = "both" ] || [ "${MODE}" = "pber" ]; then
        task_idx=$((task_idx + 1))
        echo "  [${task_idx}/${task_total}] APEX-PBER (log: ${SCRIPT_LOG_DIR}/apex_pber_${log_suffix}.log)"
        python runner/run_apex_pber_algo.py --config ${PBER_CONFIG} --env ${ENV_NAME} --gpu ${gpu} \
            > ${SCRIPT_LOG_DIR}/apex_pber_${log_suffix}.log 2>&1 &
        ALL_PIDS+=($!)
        ALL_NAMES+=("GPU${gpu}-PBER")
        if [ "${MODE}" = "both" ]; then
            sleep ${LAUNCH_DELAY_SAME_GPU}
        fi
    fi

    if [ "${MODE}" = "both" ] || [ "${MODE}" = "raspberry" ]; then
        task_idx=$((task_idx + 1))
        echo "  [${task_idx}/${task_total}] APEX-RASPBERry (log: ${SCRIPT_LOG_DIR}/apex_raspberry_${log_suffix}.log)"
        python runner/run_apex_raspberry_algo.py --config ${RASP_CONFIG} --env ${ENV_NAME} --gpu ${gpu} \
            > ${SCRIPT_LOG_DIR}/apex_raspberry_${log_suffix}.log 2>&1 &
        ALL_PIDS+=($!)
        ALL_NAMES+=("GPU${gpu}-RASPBERry")
    fi

    if [ ${idx} -lt $(( ${#GPU_IDS[@]} - 1 )) ]; then
        sleep ${LAUNCH_DELAY_BETWEEN_GPUS}
    fi
done

echo ""
echo "Submitted $(( ${#GPU_IDS[@]} * TASKS_PER_GPU )) APEX test tasks"
for idx in "${!ALL_PIDS[@]}"; do
    printf "  %-20s -> PID:%s\n" "${ALL_NAMES[$idx]}" "${ALL_PIDS[$idx]}"
done
echo ""
echo "Log dir: ${SCRIPT_LOG_DIR}"
echo "Terminate all: kill ${ALL_PIDS[@]}"
echo ""

echo "Waiting for all tasks to finish..."
for pid in "${ALL_PIDS[@]}"; do
    wait $pid 2>/dev/null || true
done

echo "Done."

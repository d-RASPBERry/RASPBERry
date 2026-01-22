#!/usr/bin/env bash

################################################################################
# APEX ablation launcher (Atlantis)
#
# Runs 3 APEX variants per GPU:
#   1) APEX-PER
#   2) APEX-PBER
#   3) APEX-RASPBERry
#
# Usage:
#   ./run_apex_abalation_atlantis.sh                # default GPU 0
#   ./run_apex_abalation_atlantis.sh -n 0,1,2       # comma-separated GPU list
#   ./run_apex_abalation_atlantis.sh -m exclusive   # exclusive mode (3x GPUs)
#
################################################################################

set -euo pipefail

GPU_LIST_ARG="0"
GPU_ASSIGNMENT_MODE="shared"
LAUNCH_DELAY_BETWEEN_GPUS=10
LAUNCH_DELAY_SAME_GPU=30

while getopts "n:m:h" opt; do
    case $opt in
        n) GPU_LIST_ARG="$OPTARG" ;;
        m) GPU_ASSIGNMENT_MODE="$OPTARG" ;;
        h)
            echo "Usage: $0 [-n GPU_IDS] [-h]"
            echo "Options: -n GPU_IDS (default: 0)"
            echo "         -m shared|exclusive (default: shared)"
            echo "Example: -n 0,1,2 -m exclusive"
            exit 0
            ;;
        \?) echo "Invalid option" >&2; exit 1 ;;
    esac
done

if ! [[ "${GPU_LIST_ARG}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Error: -n only supports comma-separated GPU IDs (e.g., 0,1,2)" >&2
    exit 1
fi

case "${GPU_ASSIGNMENT_MODE}" in
    shared|exclusive) ;;
    *)
        echo "Error: -m only supports shared or exclusive (got: ${GPU_ASSIGNMENT_MODE})" >&2
        exit 1
        ;;
esac

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

NUM_GPUS=${#GPU_IDS[@]}
TOTAL_TASKS=$((NUM_GPUS * 3))
GROUP_COUNT=0

if [ "${GPU_ASSIGNMENT_MODE}" = "exclusive" ]; then
    if (( NUM_GPUS % 3 != 0 )); then
        echo "Error: exclusive mode requires GPU count to be a multiple of 3 (got ${NUM_GPUS})" >&2
        exit 1
    fi
    GROUP_COUNT=$((NUM_GPUS / 3))
    TOTAL_TASKS=$((GROUP_COUNT * 3))
else
    GROUP_COUNT=${NUM_GPUS}
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_LOG_DIR="/home/seventheli/research/RASPBERry/logs/experiments/scripts"
mkdir -p ${SCRIPT_LOG_DIR}

ENV_NAME="Atari-AtlantisNoFrameskip-v4"
PER_CONFIG="configs/apex_per_atari.yml"
PBER_CONFIG="configs/apex_pber_atari.yml"
RASP_CONFIG="configs/apex_raspberry_atari.yml"

echo "================================================================================"
echo "APEX ablation (Atlantis) | env: ${ENV_NAME}"
echo "Target GPUs: ${GPU_IDS[*]}"
if [ "${GPU_ASSIGNMENT_MODE}" = "exclusive" ]; then
    echo "Mode: exclusive (groups: ${GROUP_COUNT})"
else
    echo "Mode: shared (PER -> PBER -> RASPBERry on each GPU)"
fi
echo "Log dir: ${SCRIPT_LOG_DIR}"
echo "Total tasks: ${TOTAL_TASKS}"
echo "================================================================================"

declare -a ALL_PIDS
declare -a ALL_NAMES

if [ "${GPU_ASSIGNMENT_MODE}" = "shared" ]; then
    for idx in "${!GPU_IDS[@]}"; do
        gpu="${GPU_IDS[$idx]}"
        echo "GPU ${gpu}: Atlantis APEX ablation group $((idx+1))"
        log_suffix="atlantis_gpu${gpu}_${TIMESTAMP}"

        echo "  [1/3] APEX-PER (log: ${SCRIPT_LOG_DIR}/apex_per_${log_suffix}.log)"
        python runner/run_apex_per_algo.py --config ${PER_CONFIG} --env ${ENV_NAME} --gpu ${gpu} \
            > ${SCRIPT_LOG_DIR}/apex_per_${log_suffix}.log 2>&1 &
        ALL_PIDS+=($!)
        ALL_NAMES+=("GPU${gpu}-PER")
        sleep ${LAUNCH_DELAY_SAME_GPU}

        echo "  [2/3] APEX-PBER (log: ${SCRIPT_LOG_DIR}/apex_pber_${log_suffix}.log)"
        python runner/run_apex_pber_algo.py --config ${PBER_CONFIG} --env ${ENV_NAME} --gpu ${gpu} \
            > ${SCRIPT_LOG_DIR}/apex_pber_${log_suffix}.log 2>&1 &
        ALL_PIDS+=($!)
        ALL_NAMES+=("GPU${gpu}-PBER")
        sleep ${LAUNCH_DELAY_SAME_GPU}

        echo "  [3/3] APEX-RASPBERry (log: ${SCRIPT_LOG_DIR}/apex_raspberry_${log_suffix}.log)"
        python runner/run_apex_raspberry_algo.py --config ${RASP_CONFIG} --env ${ENV_NAME} --gpu ${gpu} \
            > ${SCRIPT_LOG_DIR}/apex_raspberry_${log_suffix}.log 2>&1 &
        ALL_PIDS+=($!)
        ALL_NAMES+=("GPU${gpu}-RASPBERry")

        if [ ${idx} -lt $((NUM_GPUS - 1)) ]; then
            sleep ${LAUNCH_DELAY_BETWEEN_GPUS}
        fi
    done
else
    for ((group_idx=0; group_idx<GROUP_COUNT; group_idx++)); do
        base=$((group_idx * 3))
        gpu_per=${GPU_IDS[$base]}
        gpu_pber=${GPU_IDS[$((base + 1))]}
        gpu_rasp=${GPU_IDS[$((base + 2))]}
        log_suffix="atlantis_group$((group_idx + 1))_${TIMESTAMP}"

        echo "Group $((group_idx + 1)) Atlantis APEX ablation (exclusive)"

        echo "  [PER] GPU ${gpu_per} (log: ${SCRIPT_LOG_DIR}/apex_per_${log_suffix}.log)"
        python runner/run_apex_per_algo.py --config ${PER_CONFIG} --env ${ENV_NAME} --gpu ${gpu_per} \
            > ${SCRIPT_LOG_DIR}/apex_per_${log_suffix}.log 2>&1 &
        ALL_PIDS+=($!)
        ALL_NAMES+=("GPU${gpu_per}-PER(G$((group_idx + 1)))")

        echo "  [PBER] GPU ${gpu_pber} (log: ${SCRIPT_LOG_DIR}/apex_pber_${log_suffix}.log)"
        python runner/run_apex_pber_algo.py --config ${PBER_CONFIG} --env ${ENV_NAME} --gpu ${gpu_pber} \
            > ${SCRIPT_LOG_DIR}/apex_pber_${log_suffix}.log 2>&1 &
        ALL_PIDS+=($!)
        ALL_NAMES+=("GPU${gpu_pber}-PBER(G$((group_idx + 1)))")

        echo "  [RASPBERry] GPU ${gpu_rasp} (log: ${SCRIPT_LOG_DIR}/apex_raspberry_${log_suffix}.log)"
        python runner/run_apex_raspberry_algo.py --config ${RASP_CONFIG} --env ${ENV_NAME} --gpu ${gpu_rasp} \
            > ${SCRIPT_LOG_DIR}/apex_raspberry_${log_suffix}.log 2>&1 &
        ALL_PIDS+=($!)
        ALL_NAMES+=("GPU${gpu_rasp}-RASPBERry(G$((group_idx + 1)))")

        if [ ${group_idx} -lt $((GROUP_COUNT - 1)) ]; then
            sleep ${LAUNCH_DELAY_BETWEEN_GPUS}
        fi
    done
fi

echo ""
echo "Submitted ${TOTAL_TASKS} APEX tasks"
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

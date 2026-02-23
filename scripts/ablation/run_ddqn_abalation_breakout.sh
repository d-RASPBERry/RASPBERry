#!/usr/bin/env bash

################################################################################
# DDQN ablation launcher (Breakout)
#
# Runs 3 DDQN variants per GPU:
#   1) DDQN-PER
#   2) DDQN-PBER
#   3) DDQN-RASPBERry
#
# Usage:
#   ./run_ddqn_abalation_breakout.sh                # default GPU 0
#   ./run_ddqn_abalation_breakout.sh -n 0,1,2       # comma-separated GPU list
#   ./run_ddqn_abalation_breakout.sh -m exclusive   # exclusive mode (3x GPUs)
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
            echo "Usage: $0 [-n GPU_IDS] [-m shared|exclusive]"
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
    *) echo "Error: -m only supports shared or exclusive" >&2; exit 1 ;;
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
        echo "Error: exclusive mode requires GPU count to be a multiple of 3" >&2
        exit 1
    fi
    GROUP_COUNT=$((NUM_GPUS / 3))
    TOTAL_TASKS=$((GROUP_COUNT * 3))
else
    GROUP_COUNT=${NUM_GPUS}
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

PER_CONFIG="configs/experiments/ddqn/per/breakout.yml"
PBER_CONFIG="configs/experiments/ddqn/pber/breakout.yml"
RASP_CONFIG="configs/experiments/ddqn/raspberry/breakout.yml"

for cfg in "${PER_CONFIG}" "${PBER_CONFIG}" "${RASP_CONFIG}"; do
    if [ ! -f "${cfg}" ]; then
        echo "Error: missing config ${cfg}" >&2
        exit 1
    fi
done

echo "================================================================================"
echo "DDQN ablation (Breakout)"
echo "GPUs: ${GPU_IDS[*]} | Mode: ${GPU_ASSIGNMENT_MODE} | Tasks: ${TOTAL_TASKS}"
echo "================================================================================"

declare -a ALL_PIDS
declare -a ALL_NAMES

if [ "${GPU_ASSIGNMENT_MODE}" = "shared" ]; then
    for idx in "${!GPU_IDS[@]}"; do
        gpu="${GPU_IDS[$idx]}"

        python runner/run_ddqn_per_algo.py --config "${PER_CONFIG}" --gpu "${gpu}" &
        ALL_PIDS+=($!)
        ALL_NAMES+=("GPU${gpu}-PER")
        echo "  [1/3] DDQN-PER       (GPU ${gpu}) -> PID $!"
        sleep ${LAUNCH_DELAY_SAME_GPU}

        python runner/run_ddqn_pber_algo.py --config "${PBER_CONFIG}" --gpu "${gpu}" &
        ALL_PIDS+=($!)
        ALL_NAMES+=("GPU${gpu}-PBER")
        echo "  [2/3] DDQN-PBER      (GPU ${gpu}) -> PID $!"
        sleep ${LAUNCH_DELAY_SAME_GPU}

        python runner/run_ddqn_raspberry_algo.py --config "${RASP_CONFIG}" --gpu "${gpu}" &
        ALL_PIDS+=($!)
        ALL_NAMES+=("GPU${gpu}-RASPBERry")
        echo "  [3/3] DDQN-RASPBERry (GPU ${gpu}) -> PID $!"

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

        python runner/run_ddqn_per_algo.py --config "${PER_CONFIG}" --gpu "${gpu_per}" &
        ALL_PIDS+=($!)
        ALL_NAMES+=("GPU${gpu_per}-PER(G$((group_idx + 1)))")
        echo "  [PER]       GPU ${gpu_per} -> PID $!"

        python runner/run_ddqn_pber_algo.py --config "${PBER_CONFIG}" --gpu "${gpu_pber}" &
        ALL_PIDS+=($!)
        ALL_NAMES+=("GPU${gpu_pber}-PBER(G$((group_idx + 1)))")
        echo "  [PBER]      GPU ${gpu_pber} -> PID $!"

        python runner/run_ddqn_raspberry_algo.py --config "${RASP_CONFIG}" --gpu "${gpu_rasp}" &
        ALL_PIDS+=($!)
        ALL_NAMES+=("GPU${gpu_rasp}-RASPBERry(G$((group_idx + 1)))")
        echo "  [RASPBERry] GPU ${gpu_rasp} -> PID $!"

        if [ ${group_idx} -lt $((GROUP_COUNT - 1)) ]; then
            sleep ${LAUNCH_DELAY_BETWEEN_GPUS}
        fi
    done
fi

echo ""
echo "Submitted ${TOTAL_TASKS} DDQN tasks"
for idx in "${!ALL_PIDS[@]}"; do
    printf "  %-24s -> PID:%s\n" "${ALL_NAMES[$idx]}" "${ALL_PIDS[$idx]}"
done
echo "Terminate all: kill ${ALL_PIDS[@]}"

echo ""
echo "Waiting for all tasks to finish..."
for pid in "${ALL_PIDS[@]}"; do
    wait $pid 2>/dev/null || true
done
echo "Done."

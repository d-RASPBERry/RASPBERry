#!/usr/bin/env bash

################################################################################
# APEX ablation launcher (MiniGrid-DoorKey-8x8)
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
if [ ${#GPU_IDS[@]} -eq 0 ]; then echo "Error: at least one GPU ID is required" >&2; exit 1; fi
for gpu_id in "${GPU_IDS[@]}"; do
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then echo "Error: invalid GPU ID ${gpu_id}" >&2; exit 1; fi
done

NUM_GPUS=${#GPU_IDS[@]}
TOTAL_TASKS=$((NUM_GPUS * 3))
GROUP_COUNT=0
if [ "${GPU_ASSIGNMENT_MODE}" = "exclusive" ]; then
    if (( NUM_GPUS % 3 != 0 )); then echo "Error: exclusive mode requires GPU count multiple of 3" >&2; exit 1; fi
    GROUP_COUNT=$((NUM_GPUS / 3))
    TOTAL_TASKS=$((GROUP_COUNT * 3))
else
    GROUP_COUNT=${NUM_GPUS}
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

PER_CONFIG="configs/experiments/apex/per/minigrid_doorkey_8x8.yml"
PBER_CONFIG="configs/experiments/apex/pber/minigrid_doorkey_8x8.yml"
RASP_CONFIG="configs/experiments/apex/raspberry/minigrid_doorkey_8x8.yml"
for cfg in "${PER_CONFIG}" "${PBER_CONFIG}" "${RASP_CONFIG}"; do
    [ -f "${cfg}" ] || { echo "Error: missing config ${cfg}" >&2; exit 1; }
done

echo "================================================================================"
echo "APEX ablation (MiniGrid-DoorKey-8x8)"
echo "GPUs: ${GPU_IDS[*]} | Mode: ${GPU_ASSIGNMENT_MODE} | Tasks: ${TOTAL_TASKS}"
echo "================================================================================"

declare -a ALL_PIDS
declare -a ALL_NAMES

if [ "${GPU_ASSIGNMENT_MODE}" = "shared" ]; then
    for idx in "${!GPU_IDS[@]}"; do
        gpu="${GPU_IDS[$idx]}"

        python runner/run_apex_per_algo.py --config "${PER_CONFIG}" --gpu "${gpu}" &
        ALL_PIDS+=($!)
        echo "  [1/3] APEX-PER       (GPU ${gpu}) -> PID $!"
        ALL_NAMES+=("GPU${gpu}-PER")
        sleep ${LAUNCH_DELAY_SAME_GPU}

        python runner/run_apex_pber_algo.py --config "${PBER_CONFIG}" --gpu "${gpu}" &
        ALL_PIDS+=($!)
        echo "  [2/3] APEX-PBER      (GPU ${gpu}) -> PID $!"
        ALL_NAMES+=("GPU${gpu}-PBER")
        sleep ${LAUNCH_DELAY_SAME_GPU}

        python runner/run_apex_raspberry_algo.py --config "${RASP_CONFIG}" --gpu "${gpu}" &
        ALL_PIDS+=($!)
        echo "  [3/3] APEX-RASPBERry (GPU ${gpu}) -> PID $!"
        ALL_NAMES+=("GPU${gpu}-RASPBERry")

        if [ ${idx} -lt $((NUM_GPUS - 1)) ]; then sleep ${LAUNCH_DELAY_BETWEEN_GPUS}; fi
    done
else
    for ((group_idx=0; group_idx<GROUP_COUNT; group_idx++)); do
        base=$((group_idx * 3))
        gpu_per=${GPU_IDS[$base]}
        gpu_pber=${GPU_IDS[$((base + 1))]}
        gpu_rasp=${GPU_IDS[$((base + 2))]}

        python runner/run_apex_per_algo.py --config "${PER_CONFIG}" --gpu "${gpu_per}" &
        ALL_PIDS+=($!)
        echo "  [PER]       GPU ${gpu_per} -> PID $!"
        ALL_NAMES+=("GPU${gpu_per}-PER(G$((group_idx + 1)))")

        python runner/run_apex_pber_algo.py --config "${PBER_CONFIG}" --gpu "${gpu_pber}" &
        ALL_PIDS+=($!)
        echo "  [PBER]      GPU ${gpu_pber} -> PID $!"
        ALL_NAMES+=("GPU${gpu_pber}-PBER(G$((group_idx + 1)))")

        python runner/run_apex_raspberry_algo.py --config "${RASP_CONFIG}" --gpu "${gpu_rasp}" &
        ALL_PIDS+=($!)
        echo "  [RASPBERry] GPU ${gpu_rasp} -> PID $!"
        ALL_NAMES+=("GPU${gpu_rasp}-RASPBERry(G$((group_idx + 1)))")

        if [ ${group_idx} -lt $((GROUP_COUNT - 1)) ]; then sleep ${LAUNCH_DELAY_BETWEEN_GPUS}; fi
    done
fi

echo ""
echo "Submitted ${TOTAL_TASKS} APEX tasks"
for idx in "${!ALL_PIDS[@]}"; do
    printf "  %-24s -> PID:%s\n" "${ALL_NAMES[$idx]}" "${ALL_PIDS[$idx]}"
done
echo "Terminate all: kill ${ALL_PIDS[@]}"

echo "Waiting for all tasks to finish..."
for pid in "${ALL_PIDS[@]}"; do
    wait $pid 2>/dev/null || true
done
echo "Done."

#!/usr/bin/env bash

################################################################################
# JR06 block-size ablation — group 4
# Freeway seed1  m={16,32,64}
#
# Usage:
#   ./group4_freeway_s1.sh -n 0
#   ./group4_freeway_s1.sh -n 0,1,2 -m exclusive
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
        h) echo "Usage: $0 [-n GPU_IDS] [-m shared|exclusive] [-h]"; exit 0 ;;
        \?) echo "Invalid option" >&2; exit 1 ;;
    esac
done

if ! [[ "${GPU_LIST_ARG}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Error: -n only supports comma-separated GPU IDs" >&2; exit 1
fi
case "${GPU_ASSIGNMENT_MODE}" in
    shared|exclusive) ;;
    *) echo "Error: -m only supports shared or exclusive" >&2; exit 1 ;;
esac

IFS=',' read -ra GPU_IDS <<< "${GPU_LIST_ARG}"
if [ ${#GPU_IDS[@]} -eq 0 ]; then echo "Error: at least one GPU ID" >&2; exit 1; fi

NUM_GPUS=${#GPU_IDS[@]}
if [ "${GPU_ASSIGNMENT_MODE}" = "exclusive" ]; then
    if (( NUM_GPUS % 3 != 0 )); then
        echo "Error: exclusive mode requires GPU count multiple of 3 (got ${NUM_GPUS})" >&2; exit 1
    fi
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

RUNNER="runner/run_apex_raspberry_algo.py"
CFG="configs/experiments/sub"
CFG1="${CFG}/apex_raspberry_freeway_m16_seed1.yml"
CFG2="${CFG}/apex_raspberry_freeway_m32_seed1.yml"
CFG3="${CFG}/apex_raspberry_freeway_m64_seed1.yml"

for c in "${CFG1}" "${CFG2}" "${CFG3}"; do
    [ -f "${c}" ] || { echo "Error: missing config ${c}" >&2; exit 1; }
done

echo "================================================================================"
echo "JR06 group4 — freeway seed1 m={16,32,64}"
echo "GPUs: ${GPU_IDS[*]} | Mode: ${GPU_ASSIGNMENT_MODE}"
echo "================================================================================"

declare -a ALL_PIDS
declare -a ALL_NAMES

if [ "${GPU_ASSIGNMENT_MODE}" = "shared" ]; then
    for idx in "${!GPU_IDS[@]}"; do
        gpu="${GPU_IDS[$idx]}"

        python ${RUNNER} --config "${CFG1}" --gpu "${gpu}" &
        ALL_PIDS+=($!); ALL_NAMES+=("GPU${gpu}-m16")
        echo "  [1/3] m=16 (GPU ${gpu}) -> PID $!"
        sleep ${LAUNCH_DELAY_SAME_GPU}

        python ${RUNNER} --config "${CFG2}" --gpu "${gpu}" &
        ALL_PIDS+=($!); ALL_NAMES+=("GPU${gpu}-m32")
        echo "  [2/3] m=32 (GPU ${gpu}) -> PID $!"
        sleep ${LAUNCH_DELAY_SAME_GPU}

        python ${RUNNER} --config "${CFG3}" --gpu "${gpu}" &
        ALL_PIDS+=($!); ALL_NAMES+=("GPU${gpu}-m64")
        echo "  [3/3] m=64 (GPU ${gpu}) -> PID $!"

        if [ ${idx} -lt $((NUM_GPUS - 1)) ]; then sleep ${LAUNCH_DELAY_BETWEEN_GPUS}; fi
    done
else
    for ((g=0; g<NUM_GPUS/3; g++)); do
        b=$((g * 3))
        g1=${GPU_IDS[$b]}; g2=${GPU_IDS[$((b+1))]}; g3=${GPU_IDS[$((b+2))]}

        python ${RUNNER} --config "${CFG1}" --gpu "${g1}" &
        ALL_PIDS+=($!); ALL_NAMES+=("GPU${g1}-m16")
        echo "  [m=16] GPU ${g1} -> PID $!"

        python ${RUNNER} --config "${CFG2}" --gpu "${g2}" &
        ALL_PIDS+=($!); ALL_NAMES+=("GPU${g2}-m32")
        echo "  [m=32] GPU ${g2} -> PID $!"

        python ${RUNNER} --config "${CFG3}" --gpu "${g3}" &
        ALL_PIDS+=($!); ALL_NAMES+=("GPU${g3}-m64")
        echo "  [m=64] GPU ${g3} -> PID $!"

        if [ ${g} -lt $((NUM_GPUS/3 - 1)) ]; then sleep ${LAUNCH_DELAY_BETWEEN_GPUS}; fi
    done
fi

echo ""
echo "Submitted ${#ALL_PIDS[@]} tasks"
for idx in "${!ALL_PIDS[@]}"; do
    printf "  %-24s -> PID:%s\n" "${ALL_NAMES[$idx]}" "${ALL_PIDS[$idx]}"
done
echo "Terminate all: kill ${ALL_PIDS[*]}"
echo ""
echo "Waiting for all tasks to finish..."
for pid in "${ALL_PIDS[@]}"; do wait $pid 2>/dev/null || true; done
echo "Done."

#!/usr/bin/env bash

################################################################################
# SAC 消融实验启动脚本 (CarRacing 图像观测)
#
# 功能:
#   对每个指定 GPU 顺序启动 3 个 SAC 变体:
#     1) SAC-PER (经验回放 + PER)
#     2) SAC-PBER (分块回放, 无压缩)
#     3) SAC-RASPBERry (分块回放 + 压缩)
#
# 使用方法:
#   ./run_sac_ablation_CarRacing.sh                # 默认 GPU 共享 (仅使用 GPU 0)
#   ./run_sac_ablation_CarRacing.sh -n 0,1,2       # 指定逗号分隔 GPU 列表
#   ./run_sac_ablation_CarRacing.sh -m exclusive   # 开启独占模式 (需提供3的倍数GPU)
#
################################################################################

set -euo pipefail

# 默认参数
GPU_LIST_ARG="0"
GPU_ASSIGNMENT_MODE="shared"
LAUNCH_DELAY_BETWEEN_GPUS=60
LAUNCH_DELAY_SAME_GPU=120

# 解析命令行参数
while getopts "n:m:h" opt; do
    case $opt in
        n)
            GPU_LIST_ARG="$OPTARG"
            ;;
        m)
            GPU_ASSIGNMENT_MODE="$OPTARG"
            ;;
        h)
            echo "Usage: $0 [-n GPU_IDS] [-m MODE] [-h]"
            echo "Options:"
            echo "  -n GPU_IDS   Comma-separated GPU IDs (default: 0)"
            echo "  -m MODE     shared or exclusive (default: shared)"
            echo "  -h          Show this help"
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
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
        echo "Error: exclusive mode requires GPU count to be a multiple of 3 (got: ${NUM_GPUS})" >&2
        exit 1
    fi
    GROUP_COUNT=$((NUM_GPUS / 3))
    TOTAL_TASKS=$((GROUP_COUNT * 3))
else
    GROUP_COUNT=${NUM_GPUS}
fi
# 获取项目根目录（archive 移动后：上溯两层）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "================================================================================"
echo "SAC ablation (CarRacing)"
echo "GPUs: ${GPU_IDS[*]} | Mode: ${GPU_ASSIGNMENT_MODE} | Tasks: ${TOTAL_TASKS}"
echo "================================================================================"
echo ""

# 用于跟踪所有进程
declare -a ALL_PIDS
declare -a ALL_NAMES

PER_CONFIG="configs/experiments/sac/per/carracing.yml"
PBER_CONFIG="configs/experiments/sac/pber/carracing.yml"
RASP_CONFIG="configs/experiments/sac/raspberry/carracing.yml"

if [ "${GPU_ASSIGNMENT_MODE}" = "shared" ]; then
    for idx in "${!GPU_IDS[@]}"; do
        gpu="${GPU_IDS[$idx]}"
        python runner/run_sac_per_algo.py --config ${PER_CONFIG} --gpu ${gpu} &
        pid_per=$!
        ALL_PIDS+=($pid_per)
        ALL_NAMES+=("GPU${gpu}-PER")
        echo "  [1/3] SAC-PER       (GPU ${gpu}) -> PID ${pid_per}"
        sleep ${LAUNCH_DELAY_SAME_GPU}

        python runner/run_sac_pber_algo.py --config ${PBER_CONFIG} --gpu ${gpu} &
        pid_pber=$!
        ALL_PIDS+=($pid_pber)
        ALL_NAMES+=("GPU${gpu}-PBER")
        echo "  [2/3] SAC-PBER      (GPU ${gpu}) -> PID ${pid_pber}"
        sleep ${LAUNCH_DELAY_SAME_GPU}

        python runner/run_sac_raspberry_algo.py --config ${RASP_CONFIG} --gpu ${gpu} &
        pid_rasp=$!
        ALL_PIDS+=($pid_rasp)
        ALL_NAMES+=("GPU${gpu}-RASPBERry")
        echo "  [3/3] SAC-RASPBERry (GPU ${gpu}) -> PID ${pid_rasp}"
    
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
        python runner/run_sac_per_algo.py --config ${PER_CONFIG} --gpu ${gpu_per} &
        pid_per=$!
        ALL_PIDS+=($pid_per)
        ALL_NAMES+=("GPU${gpu_per}-PER(G$((group_idx + 1)))")
        echo "  [PER]       GPU ${gpu_per} -> PID ${pid_per}"

        python runner/run_sac_pber_algo.py --config ${PBER_CONFIG} --gpu ${gpu_pber} &
        pid_pber=$!
        ALL_PIDS+=($pid_pber)
        ALL_NAMES+=("GPU${gpu_pber}-PBER(G$((group_idx + 1)))")
        echo "  [PBER]      GPU ${gpu_pber} -> PID ${pid_pber}"

        python runner/run_sac_raspberry_algo.py --config ${RASP_CONFIG} --gpu ${gpu_rasp} &
        pid_rasp=$!
        ALL_PIDS+=($pid_rasp)
        ALL_NAMES+=("GPU${gpu_rasp}-RASPBERry(G$((group_idx + 1)))")
        echo "  [RASPBERry] GPU ${gpu_rasp} -> PID ${pid_rasp}"

        if [ ${group_idx} -lt $((GROUP_COUNT - 1)) ]; then
            sleep ${LAUNCH_DELAY_BETWEEN_GPUS}
        fi
    done
fi

echo ""
echo "Submitted ${TOTAL_TASKS} SAC tasks"
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


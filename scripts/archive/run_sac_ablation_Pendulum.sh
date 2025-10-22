#!/usr/bin/env bash

################################################################################
# SAC 消融实验脚本 (Pendulum)
# 
# 实验设计: 每个GPU运行一组完整的对比实验
#   - SAC-PER (基线)
#   - SAC-PBER (分块，无压缩)
#   - SAC-RASPBERry (分块+压缩)
#
# 使用方法:
#   ./run_sac_ablation_Pendulum.sh -n 4
#   ./run_sac_ablation_Pendulum.sh -n 2
#
################################################################################

set -euo pipefail

NUM_GPUS=4
LAUNCH_DELAY_BETWEEN_GPUS=60
LAUNCH_DELAY_SAME_GPU=120

while getopts "n:h" opt; do
    case $opt in
        n) NUM_GPUS=$OPTARG ;;
        h)
            echo "用法: $0 [-n NUM_GPUS] [-h]"
            exit 0
            ;;
        \?) echo "无效选项" >&2; exit 1 ;;
    esac
done

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -lt 1 ]; then
    echo "错误: GPU数量必须是正整数" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_LOG_DIR="./logs/scripts"
mkdir -p ${SCRIPT_LOG_DIR}

echo "================================================================================"
echo "🚀 SAC 消融实验 (Pendulum) - ${NUM_GPUS} GPUs"
echo "================================================================================"
echo "  环境: Pendulum-Pendulum (图像观测)"
echo "  每GPU: PER + PBER + RASPBERry"
echo "================================================================================"

declare -a ALL_PIDS
declare -a ALL_NAMES

PER_CONFIG="configs/experiments/sac/per/pendulum.yml"
PBER_CONFIG="configs/experiments/sac/pber/pendulum.yml"
RASP_CONFIG="configs/experiments/sac/raspberry/pendulum.yml"

GPU_IDS=()
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_IDS+=($i)
done

for gpu in "${GPU_IDS[@]}"; do
    echo "┌─ GPU ${gpu}: Pendulum ───────────────────────────────────────────────┐"
    log_suffix="pendulum_gpu${gpu}_${TIMESTAMP}"

    echo "  [1/3] PER..."
    python runner/run_sac_per_algo.py --config ${PER_CONFIG} --gpu ${gpu} \
        > ${SCRIPT_LOG_DIR}/sac_per_${log_suffix}.log 2>&1 &
    ALL_PIDS+=($!)
    ALL_NAMES+=("GPU${gpu}-PER")
    sleep ${LAUNCH_DELAY_SAME_GPU}

    echo "  [2/3] PBER..."
    python runner/run_sac_raspberry_algo.py --config ${PBER_CONFIG} --gpu ${gpu} \
        > ${SCRIPT_LOG_DIR}/sac_pber_${log_suffix}.log 2>&1 &
    ALL_PIDS+=($!)
    ALL_NAMES+=("GPU${gpu}-PBER")
    sleep ${LAUNCH_DELAY_SAME_GPU}

    echo "  [3/3] RASPBERry..."
    python runner/run_sac_raspberry_algo.py --config ${RASP_CONFIG} --gpu ${gpu} \
        > ${SCRIPT_LOG_DIR}/sac_raspberry_${log_suffix}.log 2>&1 &
    ALL_PIDS+=($!)
    ALL_NAMES+=("GPU${gpu}-RASPBERry")
    
    if [ ${gpu} -lt $((NUM_GPUS - 1)) ]; then
        sleep ${LAUNCH_DELAY_BETWEEN_GPUS}
    fi
    echo "└───────────────────────────────────────────────────────────────────────┘"
done

echo ""
echo "✅ 已启动 $((NUM_GPUS * 3)) 个实验"
echo "监控: nvidia-smi"
echo "停止: kill ${ALL_PIDS[@]}"

for pid in "${ALL_PIDS[@]}"; do
    wait $pid 2>/dev/null || true
done

echo "🎉 完成!"


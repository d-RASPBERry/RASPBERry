#!/usr/bin/env bash

################################################################################
# SAC 消融实验启动脚本 (LunarLanderContinuous)
#
# 功能:
#   对每个指定 GPU 按顺序启动 3 个 SAC 变体:
#     1) SAC-PER (经验回放 + PER)
#     2) SAC-PBER (分块回放, 无压缩)
#     3) SAC-RASPBERry (分块回放 + 压缩)
#
# 使用方法:
#   ./run_sac_ablation_LunarLander.sh            # 默认仅使用 GPU 0
#   ./run_sac_ablation_LunarLander.sh -n 0,1,2   # 指定逗号分隔 GPU 列表
#
################################################################################

set -euo pipefail

GPU_LIST_ARG="0"
LAUNCH_DELAY_BETWEEN_GPUS=10
LAUNCH_DELAY_SAME_GPU=120

while getopts "n:h" opt; do
    case $opt in
        n) GPU_LIST_ARG="$OPTARG" ;;
        h)
            echo "用法: $0 [-n GPU_IDS] [-h]"
            echo "选项: -n GPU_IDS (默认: 0)"
            echo "  示例: -n 0,1,2 或 -n 0,1,3"
            exit 0
            ;;
        \?) echo "无效选项: -$OPTARG" >&2; exit 1 ;;
    esac
done

if ! [[ "${GPU_LIST_ARG}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "错误: -n 仅支持逗号分隔的GPU编号列表 (示例: 0,1,3)" >&2
    exit 1
fi

IFS=',' read -ra GPU_IDS <<< "${GPU_LIST_ARG}"

if [ ${#GPU_IDS[@]} -eq 0 ]; then
    echo "错误: 至少需要一个GPU编号" >&2
    exit 1
fi

for gpu_id in "${GPU_IDS[@]}"; do
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
        echo "错误: GPU编号必须是非负整数 (收到: ${gpu_id})" >&2
        exit 1
    fi
done

NUM_GPUS=${#GPU_IDS[@]}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_LOG_DIR="./logs/scripts"
mkdir -p ${SCRIPT_LOG_DIR}

echo "================================================================================"
echo "🚀 启动 SAC 消融实验 (环境: LunarLanderContinuous)"
echo "    目标 GPU 列表: ${GPU_IDS[*]}"
echo "    每块 GPU 执行顺序: SAC-PER → SAC-PBER → SAC-RASPBERry"
echo "    输出日志目录: ${SCRIPT_LOG_DIR}"
echo "    本次计划任务数: $((NUM_GPUS * 3))"
echo "================================================================================"

declare -a ALL_PIDS
declare -a ALL_NAMES

PER_CONFIG="configs/experiments/sac/per/lunarlander.yml"
PBER_CONFIG="configs/experiments/sac/pber/lunarlander.yml"
RASP_CONFIG="configs/experiments/sac/raspberry/lunarlander.yml"

for idx in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[$idx]}"
    echo "┌─ GPU ${gpu}: 第 $((idx+1)) 组 LunarLander SAC 消融任务 ─────────────────────┐"
    log_suffix="lunarlander_gpu${gpu}_${TIMESTAMP}"

    echo "  [1/3] SAC-PER 启动 (日志: ${SCRIPT_LOG_DIR}/sac_per_${log_suffix}.log)"
    python runner/run_sac_per_algo.py --config ${PER_CONFIG} --gpu ${gpu} \
        > ${SCRIPT_LOG_DIR}/sac_per_${log_suffix}.log 2>&1 &
    ALL_PIDS+=($!)
    ALL_NAMES+=("GPU${gpu}-PER")
    echo "       后台 PID: $!"
    sleep ${LAUNCH_DELAY_SAME_GPU}

    echo "  [2/3] SAC-PBER 启动 (日志: ${SCRIPT_LOG_DIR}/sac_pber_${log_suffix}.log)"
    python runner/run_sac_pber_algo.py --config ${PBER_CONFIG} --gpu ${gpu} \
        > ${SCRIPT_LOG_DIR}/sac_pber_${log_suffix}.log 2>&1 &
    ALL_PIDS+=($!)
    ALL_NAMES+=("GPU${gpu}-PBER")
    echo "       后台 PID: $!"
    sleep ${LAUNCH_DELAY_SAME_GPU}

    echo "  [3/3] SAC-RASPBERry 启动 (日志: ${SCRIPT_LOG_DIR}/sac_raspberry_${log_suffix}.log)"
    python runner/run_sac_raspberry_algo.py --config ${RASP_CONFIG} --gpu ${gpu} \
        > ${SCRIPT_LOG_DIR}/sac_raspberry_${log_suffix}.log 2>&1 &
    ALL_PIDS+=($!)
    ALL_NAMES+=("GPU${gpu}-RASPBERry")
    echo "       后台 PID: $!"
    
    if [ ${idx} -lt $((NUM_GPUS - 1)) ]; then
        sleep ${LAUNCH_DELAY_BETWEEN_GPUS}
    fi
    echo "└─────────────────────────────────────────────────────────────────────────────┘"
done

echo ""
echo "✅ 已提交 $((NUM_GPUS * 3)) 个 SAC 后台任务"
for idx in "${!GPU_IDS[@]}"; do
    gpu=${GPU_IDS[$idx]}
    printf "  GPU %d -> PER:%s  PBER:%s  RASPBERry:%s\n" \
        "${gpu}" "${ALL_PIDS[$((idx*3))]}" "${ALL_PIDS[$((idx*3+1))]}" "${ALL_PIDS[$((idx*3+2))]}"
done
echo ""
echo "日志目录: ${SCRIPT_LOG_DIR}"
echo "监控建议: watch -n 2 'nvidia-smi'"
echo "终止全部: kill ${ALL_PIDS[@]}"
echo ""

echo "⏳ 等待完成..."
for pid in "${ALL_PIDS[@]}"; do
    wait $pid 2>/dev/null || true
done

echo "🎉 完成!"


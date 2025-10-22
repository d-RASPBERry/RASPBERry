#!/usr/bin/env bash

################################################################################
# SAC 消融实验脚本 (CarRacing)
# 
# 实验设计: 每个GPU运行一组完整的对比实验
#   - SAC-PER (基线)
#   - SAC-PBER (分块，无压缩)
#   - SAC-RASPBERry (分块+压缩)
#
# 使用方法:
#   ./run_sac_ablation_CarRacing.sh -n 4
#   ./run_sac_ablation_CarRacing.sh -n 2  # 只用2个GPU
#   ./run_sac_ablation_CarRacing.sh       # 默认4个GPU
#
################################################################################

set -euo pipefail

# 默认配置
NUM_GPUS=4
LAUNCH_DELAY_BETWEEN_GPUS=10
LAUNCH_DELAY_SAME_GPU=5

# 解析命令行参数
while getopts "n:h" opt; do
    case $opt in
        n)
            NUM_GPUS=$OPTARG
            ;;
        h)
            echo "用法: $0 [-n NUM_GPUS] [-h]"
            echo ""
            echo "选项:"
            echo "  -n NUM_GPUS    使用的GPU数量 (默认: 4)"
            echo "  -h             显示帮助信息"
            exit 0
            ;;
        \?)
            echo "无效选项: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

# 验证GPU数量
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -lt 1 ]; then
    echo "错误: GPU数量必须是正整数" >&2
    exit 1
fi

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_LOG_DIR="./logs/scripts"
mkdir -p ${SCRIPT_LOG_DIR}

echo "================================================================================"
echo "🚀 启动 SAC 消融实验 (CarRacing)"
echo "================================================================================"
echo "项目目录: ${PROJECT_ROOT}"
echo "时间戳: ${TIMESTAMP}"
echo "使用GPU: ${NUM_GPUS} 个"
echo ""
echo "实验配置:"
echo "  环境: BOX2DI-CarRacing (96x96x3 图像)"
echo "  每GPU运行: PER + PBER + RASPBERry (3个实验)"
echo "  总实验数: $((NUM_GPUS * 3)) 个"
echo ""
echo "配置文件:"
echo "  PER:       configs/experiments/sac/per/carracing.yml"
echo "  PBER:      configs/experiments/sac/pber/carracing.yml"
echo "  RASPBERry: configs/experiments/sac/raspberry/carracing.yml"
echo "================================================================================"
echo ""

# 用于跟踪所有进程
declare -a ALL_PIDS
declare -a ALL_NAMES

PER_CONFIG="configs/experiments/sac/per/carracing.yml"
PBER_CONFIG="configs/experiments/sac/pber/carracing.yml"
RASP_CONFIG="configs/experiments/sac/raspberry/carracing.yml"

# 生成GPU ID数组
GPU_IDS=()
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_IDS+=($i)
done

for gpu in "${GPU_IDS[@]}"; do
    replicate=$((gpu + 1))
    header="┌────────────────────────────────────────────────────────────────────────────┐"
    title=$(printf "│ GPU %d: CarRacing 消融实验 (重复 #%d)                                   │" "${gpu}" "${replicate}")
    footer="└────────────────────────────────────────────────────────────────────────────┘"
    echo "${header}"
    echo "${title}"
    echo "${footer}"

    log_suffix="carracing_gpu${gpu}_${TIMESTAMP}"

    # 1. PER (基线)
    echo "  [1/3] 启动 SAC-PER-CarRacing (GPU ${gpu})..."
    python runner/run_sac_per_algo.py \
        --config ${PER_CONFIG} \
        --gpu ${gpu} \
        > ${SCRIPT_LOG_DIR}/sac_per_${log_suffix}.log 2>&1 &
    pid_per=$!
    ALL_PIDS+=($pid_per)
    ALL_NAMES+=("GPU${gpu}-PER")
    echo "       PID: ${pid_per}"
    echo "       ⏳ 等待 ${LAUNCH_DELAY_SAME_GPU} 秒..."
    sleep ${LAUNCH_DELAY_SAME_GPU}

    # 2. PBER (分块，无压缩)
    echo "  [2/3] 启动 SAC-PBER-CarRacing (GPU ${gpu})..."
    python runner/run_sac_raspberry_algo.py \
        --config ${PBER_CONFIG} \
        --gpu ${gpu} \
        > ${SCRIPT_LOG_DIR}/sac_pber_${log_suffix}.log 2>&1 &
    pid_pber=$!
    ALL_PIDS+=($pid_pber)
    ALL_NAMES+=("GPU${gpu}-PBER")
    echo "       PID: ${pid_pber}"
    echo "       ⏳ 等待 ${LAUNCH_DELAY_SAME_GPU} 秒..."
    sleep ${LAUNCH_DELAY_SAME_GPU}

    # 3. RASPBERry (分块+压缩)
    echo "  [3/3] 启动 SAC-RASPBERry-CarRacing (GPU ${gpu})..."
    python runner/run_sac_raspberry_algo.py \
        --config ${RASP_CONFIG} \
        --gpu ${gpu} \
        > ${SCRIPT_LOG_DIR}/sac_raspberry_${log_suffix}.log 2>&1 &
    pid_rasp=$!
    ALL_PIDS+=($pid_rasp)
    ALL_NAMES+=("GPU${gpu}-RASPBERry")
    echo "       PID: ${pid_rasp}"
    
    if [ ${gpu} -lt $((NUM_GPUS - 1)) ]; then
        echo "       ⏳ 等待 ${LAUNCH_DELAY_BETWEEN_GPUS} 秒后启动下一组..."
        sleep ${LAUNCH_DELAY_BETWEEN_GPUS}
    fi
    echo ""
done

echo "================================================================================"
echo "✅ 所有实验已启动 (共 $((NUM_GPUS * 3)) 个进程)"
echo "================================================================================"
echo ""
echo "GPU 分配 (PER → PBER → RASPBERry):"
for idx in "${!GPU_IDS[@]}"; do
    gpu=${GPU_IDS[$idx]}
    per_pid=${ALL_PIDS[$((idx * 3))]}
    pber_pid=${ALL_PIDS[$((idx * 3 + 1))]}
    rasp_pid=${ALL_PIDS[$((idx * 3 + 2))]}
    printf "  GPU %d: PID %s (PER) + %s (PBER) + %s (RASPBERry)  [重复 #%d]\n" \
        "${gpu}" "${per_pid}" "${pber_pid}" "${rasp_pid}" "$((gpu + 1))"
done
echo ""
echo "监控命令: watch -n 2 'nvidia-smi; echo; ps aux | grep \"run_sac\"'"
echo "杀死所有: kill ${ALL_PIDS[@]}"
echo ""

echo "⏳ 等待所有实验完成..."
for idx in "${!ALL_PIDS[@]}"; do
    wait ${ALL_PIDS[$idx]} 2>/dev/null || echo "  进程 ${ALL_PIDS[$idx]} (${ALL_NAMES[$idx]}) 已退出"
done

echo ""
echo "🎉 所有实验已完成! 请查看 MLflow 追踪结果"


#!/usr/bin/env bash

################################################################################
# SAC 消融实验脚本 (BipedalWalker)
# 
# 实验设计: 每个GPU运行一组完整的对比实验
#   - SAC-PER (基线)
#   - SAC-PBER (分块，无压缩)
#   - SAC-RASPBERry (分块+压缩)
#
# 使用方法:
#   ./run_sac_ablation_BipedalWalker.sh -n 4
#   ./run_sac_ablation_BipedalWalker.sh -n 2  # 只用2个GPU
#   ./run_sac_ablation_BipedalWalker.sh       # 默认4个GPU
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
            echo ""
            echo "示例:"
            echo "  $0 -n 4        # 使用4个GPU"
            echo "  $0 -n 2        # 使用2个GPU"
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

# 获取项目根目录（archive 移动后：上溯两层）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_LOG_DIR="./logs/scripts"
mkdir -p ${SCRIPT_LOG_DIR}

echo "================================================================================"
echo "🚀 启动 SAC 消融实验 (BipedalWalker)"
echo "================================================================================"
echo "项目目录: ${PROJECT_ROOT}"
echo "时间戳: ${TIMESTAMP}"
echo "使用GPU: ${NUM_GPUS} 个"
echo ""
echo "实验配置:"
echo "  环境: BOX2DV-BipedalWalker-v3 (24维状态 → 4维动作)"
echo "  每GPU运行: PER + PBER + RASPBERry (3个实验)"
echo "  总实验数: $((NUM_GPUS * 3)) 个"
echo ""
echo "配置文件:"
echo "  PER:       configs/experiments/sac/per/bipedalwalker.yml"
echo "  PBER:      configs/experiments/sac/pber/bipedalwalker.yml"
echo "  RASPBERry: configs/experiments/sac/raspberry/bipedalwalker.yml"
echo "================================================================================"
echo ""

# 用于跟踪所有进程
declare -a ALL_PIDS
declare -a ALL_NAMES

PER_CONFIG="configs/experiments/sac/per/bipedalwalker.yml"
PBER_CONFIG="configs/experiments/sac/pber/bipedalwalker.yml"
RASP_CONFIG="configs/experiments/sac/raspberry/bipedalwalker.yml"

# 生成GPU ID数组
GPU_IDS=()
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_IDS+=($i)
done

for gpu in "${GPU_IDS[@]}"; do
    replicate=$((gpu + 1))
    header="┌────────────────────────────────────────────────────────────────────────────┐"
    title=$(printf "│ GPU %d: BipedalWalker 消融实验 (重复 #%d)                               │" "${gpu}" "${replicate}")
    footer="└────────────────────────────────────────────────────────────────────────────┘"
    echo "${header}"
    echo "${title}"
    echo "${footer}"

    log_suffix="bipedalwalker_gpu${gpu}_${TIMESTAMP}"

    # 1. PER (基线)
    echo "  [1/3] 启动 SAC-PER-BipedalWalker (GPU ${gpu})..."
    python runner/run_sac_per_algo.py \
        --config ${PER_CONFIG} \
        --gpu ${gpu} \
        > ${SCRIPT_LOG_DIR}/sac_per_${log_suffix}.log 2>&1 &
    pid_per=$!
    ALL_PIDS+=($pid_per)
    ALL_NAMES+=("GPU${gpu}-PER")
    echo "       PID: ${pid_per}"
    echo "       配置: ${PER_CONFIG}"
    echo "       ⏳ 等待 ${LAUNCH_DELAY_SAME_GPU} 秒..."
    sleep ${LAUNCH_DELAY_SAME_GPU}

    # 2. PBER (分块，无压缩)
    echo "  [2/3] 启动 SAC-PBER-BipedalWalker (GPU ${gpu})..."
    python runner/run_sac_raspberry_algo.py \
        --config ${PBER_CONFIG} \
        --gpu ${gpu} \
        > ${SCRIPT_LOG_DIR}/sac_pber_${log_suffix}.log 2>&1 &
    pid_pber=$!
    ALL_PIDS+=($pid_pber)
    ALL_NAMES+=("GPU${gpu}-PBER")
    echo "       PID: ${pid_pber}"
    echo "       配置: ${PBER_CONFIG}"
    echo "       ⏳ 等待 ${LAUNCH_DELAY_SAME_GPU} 秒..."
    sleep ${LAUNCH_DELAY_SAME_GPU}

    # 3. RASPBERry (分块+压缩)
    echo "  [3/3] 启动 SAC-RASPBERry-BipedalWalker (GPU ${gpu})..."
    python runner/run_sac_raspberry_algo.py \
        --config ${RASP_CONFIG} \
        --gpu ${gpu} \
        > ${SCRIPT_LOG_DIR}/sac_raspberry_${log_suffix}.log 2>&1 &
    pid_rasp=$!
    ALL_PIDS+=($pid_rasp)
    ALL_NAMES+=("GPU${gpu}-RASPBERry")
    echo "       PID: ${pid_rasp}"
    echo "       配置: ${RASP_CONFIG}"
    
    # 如果不是最后一个GPU，等待后再启动下一组
    if [ ${gpu} -lt $((NUM_GPUS - 1)) ]; then
        echo "       ⏳ 等待 ${LAUNCH_DELAY_BETWEEN_GPUS} 秒后启动下一组..."
        sleep ${LAUNCH_DELAY_BETWEEN_GPUS}
    fi
    echo ""
done

################################################################################
# 实验状态总结
################################################################################
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
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "监控命令"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "查看所有实验状态:"
echo "  watch -n 2 'nvidia-smi; echo; ps aux | grep \"run_sac\"'"
echo ""
echo "查看单个实验日志:"
echo "  tail -f ${SCRIPT_LOG_DIR}/sac_per_bipedalwalker_gpu0_${TIMESTAMP}.log"
echo "  tail -f ${SCRIPT_LOG_DIR}/sac_pber_bipedalwalker_gpu0_${TIMESTAMP}.log"
echo "  tail -f ${SCRIPT_LOG_DIR}/sac_raspberry_bipedalwalker_gpu0_${TIMESTAMP}.log"
echo ""
echo "检查进程状态:"
for idx in "${!ALL_PIDS[@]}"; do
    echo "  ps -p ${ALL_PIDS[$idx]} -o pid,comm,%cpu,%mem,etime,cmd  # ${ALL_NAMES[$idx]}"
done
echo ""
echo "杀死所有实验 (慎用!):"
echo "  kill ${ALL_PIDS[@]}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

################################################################################
# 等待所有实验完成
################################################################################
echo "⏳ 等待所有实验完成..."
echo "   (可以按 Ctrl+C 退出监控，后台进程会继续运行)"
echo ""

for idx in "${!ALL_PIDS[@]}"; do
    pid=${ALL_PIDS[$idx]}
    name=${ALL_NAMES[$idx]}
    echo "等待 ${name} (PID: ${pid})..."
    wait $pid 2>/dev/null || echo "  ⚠️  进程 ${pid} (${name}) 已退出"
done

echo ""
echo "================================================================================"
echo "🎉 所有实验已完成!"
echo "================================================================================"
echo "结果位置: 请查看 MLflow 实验追踪"
echo "  PER:       experiment=SAC-BipedalWalker, tags.buffer=PER"
echo "  PBER:      experiment=SAC-BipedalWalker-PBER, tags.buffer=PBER"
echo "  RASPBERry: experiment=SAC-BipedalWalker, tags.buffer=RASPBERry"
echo "================================================================================"


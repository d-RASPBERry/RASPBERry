#!/usr/bin/env bash

################################################################################
# SAC 4-GPU 并行实验脚本 (Walker2d - MuJoCo)
# 
# 实验计划:
#   GPU 0-3: 全部运行 GYM-Walker2d-v4 (向量环境, 17维状态 → 6维动作)
#
# 每个GPU运行1组实验(2个算法):
#   - SAC-PER (基线)
#   - SAC-RASPBERry (压缩版本)
#
# 配置系统:
#   使用 configs/experiments/sac/ 下的实验配置
#   环境配置已包含在实验配置文件中
#   日志路径由 runtime.yml 统一管理
#
# 使用方法:
#   ./run_sac_4gpu_parallel_Walker2d.sh
#
################################################################################

set -euo pipefail

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_LOG_DIR="./logs/scripts"

mkdir -p ${SCRIPT_LOG_DIR}

echo "================================================================================"
echo "🚀 启动 SAC 4-GPU 并行实验 (Walker2d - MuJoCo)"
echo "================================================================================"
echo "项目目录: ${PROJECT_ROOT}"
echo "时间戳: ${TIMESTAMP}"
echo "脚本日志: ${SCRIPT_LOG_DIR}"
echo ""
echo "实验配置:"
echo "  环境: GYM-Walker2d-v4 (17维状态 → 6维动作, MuJoCo)"
echo "  GPU: 0-3 (4个重复实验)"
echo "  算法: SAC-PER + SAC-RASPBERry"
echo ""
echo "配置文件:"
echo "  PER:       configs/experiments/sac/per/walker2d.yml"
echo "  RASPBERry: configs/experiments/sac/raspberry/walker2d.yml"
echo ""
echo "优化参数:"
echo "  - train_batch_size: 512"
echo "  - chunk_size: 24 (RASPBERry)"
echo "  - replay buffer: 1M"
echo "  - training_intensity: 2.0"
echo ""
echo "注意: 训练日志路径将由 runner 在启动时打印"
echo "================================================================================"
echo ""

# 用于跟踪所有进程的PID
declare -a ALL_PIDS
declare -a ALL_NAMES

GPU_IDS=(0 1 2 3)
PER_CONFIG="configs/experiments/sac/per/walker2d.yml"
RASP_CONFIG="configs/experiments/sac/raspberry/walker2d.yml"

LAUNCH_DELAY_BETWEEN_GPUS=10  # GPU组之间间隔10秒
LAUNCH_DELAY_SAME_GPU=5       # 同一GPU上两个实验间隔5秒 (向量环境)

for gpu in "${GPU_IDS[@]}"; do
    replicate=$((gpu + 1))
    header="┌────────────────────────────────────────────────────────────────────────────┐"
    title=$(printf "│ GPU %d: Walker2d-v4 (重复实验 #%d)                                       │" "${gpu}" "${replicate}")
    footer="└────────────────────────────────────────────────────────────────────────────┘"
    echo "${header}"
    echo "${title}"
    echo "${footer}"

    log_suffix="walker2d_gpu${gpu}_${TIMESTAMP}"

    echo "  [1/2] 启动 SAC-PER-Walker2d (GPU ${gpu})..."
    python runner/run_sac_per_algo.py \
        --config ${PER_CONFIG} \
        --gpu ${gpu} \
        > ${SCRIPT_LOG_DIR}/sac_per_${log_suffix}.log 2>&1 &
    pid_per=$!
    ALL_PIDS+=($pid_per)
    ALL_NAMES+=("GPU${gpu}-SAC-PER-Walker2d")
    echo "       PID: ${pid_per}"
    echo "       配置: ${PER_CONFIG}"
    echo "       ⏳ 等待 ${LAUNCH_DELAY_SAME_GPU} 秒后启动 RASPBERry..."
    sleep ${LAUNCH_DELAY_SAME_GPU}

    echo "  [2/2] 启动 SAC-RASPBERry-Walker2d (GPU ${gpu})..."
    python runner/run_sac_raspberry_algo.py \
        --config ${RASP_CONFIG} \
        --gpu ${gpu} \
        > ${SCRIPT_LOG_DIR}/sac_raspberry_${log_suffix}.log 2>&1 &
    pid_rasp=$!
    ALL_PIDS+=($pid_rasp)
    ALL_NAMES+=("GPU${gpu}-SAC-RASPBERry-Walker2d")
    echo "       PID: ${pid_rasp}"
    echo "       配置: ${RASP_CONFIG}"
    
    # 如果不是最后一个GPU，等待间隔后再启动下一组
    if [ ${gpu} -lt ${GPU_IDS[-1]} ]; then
        echo "       ⏳ 等待 ${LAUNCH_DELAY_BETWEEN_GPUS} 秒后启动下一组 GPU 实验..."
        sleep ${LAUNCH_DELAY_BETWEEN_GPUS}
    fi
    echo ""
done

################################################################################
# 实验状态总结
################################################################################
echo "================================================================================"
echo "✅ 所有实验已启动 (共 8 个进程)"
echo "================================================================================"
echo ""
echo "GPU 分配:"
for idx in "${!GPU_IDS[@]}"; do
    gpu=${GPU_IDS[$idx]}
    per_name="GPU${gpu}-SAC-PER-Walker2d"
    rasp_name="GPU${gpu}-SAC-RASPBERry-Walker2d"
    per_pid=${ALL_PIDS[$((idx * 2))]}
    rasp_pid=${ALL_PIDS[$((idx * 2 + 1))]}
    printf "  GPU %d: PID %s (PER)  + %s (RASPBERry)  [重复 #%d]\n" \
        "${gpu}" "${per_pid}" "${rasp_pid}" "$((gpu + 1))"
done
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "监控命令"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. 查看脚本日志 (查看启动状态和训练日志路径):"
for gpu in "${GPU_IDS[@]}"; do
    log_suffix="walker2d_gpu${gpu}_${TIMESTAMP}"
    echo "   # GPU ${gpu} (重复 #$((gpu + 1)))"
    echo "   tail -f ${SCRIPT_LOG_DIR}/sac_per_${log_suffix}.log"
    echo "   tail -f ${SCRIPT_LOG_DIR}/sac_raspberry_${log_suffix}.log"
    echo ""
done
echo ""
echo "2. 快速监控 (所有实验):"
echo "   watch -n 5 'tail -n 2 ${SCRIPT_LOG_DIR}/*_${TIMESTAMP}.log | grep -E \"Iter|reward\"'"
echo ""
echo "3. 检查进程状态:"
ps_args=()
for idx in "${!ALL_PIDS[@]}"; do
    ps_args+=(${ALL_PIDS[$idx]})
done
echo "   ps -p ${ps_args[*]}"
echo ""
echo "4. GPU 使用情况:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "提示: 训练日志的实际路径已由各个 runner 打印在脚本日志中"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏳ 等待所有实验完成 (预计24小时)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

################################################################################
# 等待所有进程完成
################################################################################

# 等待所有进程
failed=0
for i in "${!ALL_PIDS[@]}"; do
    pid=${ALL_PIDS[$i]}
    name=${ALL_NAMES[$i]}
    
    echo "等待 ${name} (PID: ${pid}) 完成..."
    wait $pid
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "  ✅ ${name} 成功完成"
    else
        echo "  ❌ ${name} 失败 (exit code: ${exit_code})"
        failed=$((failed + 1))
    fi
done

echo ""
echo "================================================================================"
echo "📊 实验完成总结"
echo "================================================================================"
echo "总进程数: ${#ALL_PIDS[@]}"
echo "成功: $((${#ALL_PIDS[@]} - failed))"
echo "失败: ${failed}"
echo ""


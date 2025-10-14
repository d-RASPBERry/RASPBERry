#!/usr/bin/env bash

################################################################################
# SAC 4-GPU 并行实验脚本
# 
# 实验计划:
#   GPU 0-3: 全部运行 LunarLanderContinuous (向量环境, 8维状态 → 2维动作)
#
# 每个GPU运行1组实验(2个算法):
#   - SAC-PER (基线)
#   - SAC-RASPBERry (压缩版本)
#
# 使用方法:
#   ./run_sac_4gpu_parallel_LunarLander.sh
#
################################################################################

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs"

mkdir -p ${LOG_DIR}

echo "================================================================================"
echo "🚀 启动 SAC 4-GPU 并行实验"
echo "================================================================================"
echo "时间戳: ${TIMESTAMP}"
echo "日志目录: ${LOG_DIR}"
echo ""
echo "实验配置:"
echo "  GPU 0-3: LunarLanderContinuous (向量)"
echo ""
echo "每个GPU运行: SAC-PER + SAC-RASPBERry"
echo "================================================================================"
echo ""

# 用于跟踪所有进程的PID
declare -a ALL_PIDS
declare -a ALL_NAMES

GPU_IDS=(0 1 2 3)
ENV_NAME="BOX2DV-LunarLanderContinuous"
PER_CONFIG="configs/sac_per_vector.yml"
RASP_CONFIG="configs/sac_raspberry_vector.yml"

for gpu in "${GPU_IDS[@]}"; do
    replicate=$((gpu + 1))
    header="┌────────────────────────────────────────────────────────────────────────────┐"
    title=$(printf "│ GPU %d: LunarLanderContinuous (重复实验 #%d)                             │" "${gpu}" "${replicate}")
    footer="└────────────────────────────────────────────────────────────────────────────┘"
    echo "${header}"
    echo "${title}"
    echo "${footer}"

    log_suffix="lunarlander_gpu${gpu}_${TIMESTAMP}"

    echo "  [1/2] 启动 SAC-PER-LunarLander (GPU ${gpu})..."
    python runner/run_sac_per_algo.py \
        --env ${ENV_NAME} \
        --config ${PER_CONFIG} \
        --gpu ${gpu} \
        > ${LOG_DIR}/sac_per_${log_suffix}.log 2>&1 &
    pid_per=$!
    ALL_PIDS+=($pid_per)
    ALL_NAMES+=("GPU${gpu}-SAC-PER-LunarLander")
    echo "       PID: ${pid_per}"

    echo "  [2/2] 启动 SAC-RASPBERry-LunarLander (GPU ${gpu})..."
    python runner/run_sac_raspberry_algo.py \
        --env ${ENV_NAME} \
        --config ${RASP_CONFIG} \
        --gpu ${gpu} \
        > ${LOG_DIR}/sac_raspberry_${log_suffix}.log 2>&1 &
    pid_rasp=$!
    ALL_PIDS+=($pid_rasp)
    ALL_NAMES+=("GPU${gpu}-SAC-RASPBERry-LunarLander")
    echo "       PID: ${pid_rasp}"
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
    per_name="GPU${gpu}-SAC-PER-LunarLander"
    rasp_name="GPU${gpu}-SAC-RASPBERry-LunarLander"
    per_pid=${ALL_PIDS[$((idx * 2))]}
    rasp_pid=${ALL_PIDS[$((idx * 2 + 1))]}
    printf "  GPU %d: PID %s (PER)  + %s (RASPBERry)  [LunarLander重复实验 #%d]\n" \
        "${gpu}" "${per_pid}" "${rasp_pid}" "$((gpu + 1))"
done
echo ""
echo "监控日志:"
for gpu in "${GPU_IDS[@]}"; do
    log_suffix="lunarlander_gpu${gpu}_${TIMESTAMP}"
    echo "  # GPU ${gpu} - LunarLander (重复实验 #$((gpu + 1)))"
    echo "  tail -f ${LOG_DIR}/sac_per_${log_suffix}.log"
    echo "  tail -f ${LOG_DIR}/sac_raspberry_${log_suffix}.log"
    echo ""
done
echo ""
echo "快速监控 (所有实验最新进展):"
echo "  watch -n 5 'tail -n 2 ${LOG_DIR}/*_${TIMESTAMP}.log | grep -E \"Iter|reward\"'"
echo ""
echo "检查进程状态:"
ps_args=()
for idx in "${!ALL_PIDS[@]}"; do
    ps_args+=(${ALL_PIDS[$idx]})
done
echo "  ps -p ${ps_args[*]}"
echo ""
echo "================================================================================"
echo "⏳ 等待所有实验完成..."
echo "================================================================================"
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
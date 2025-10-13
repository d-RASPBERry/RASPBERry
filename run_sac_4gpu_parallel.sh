#!/usr/bin/env bash

################################################################################
# SAC 4-GPU 并行实验脚本
# 
# 实验计划:
#   GPU 0: LunarLanderContinuous  (向量环境, 8维状态 → 2维动作)
#   GPU 1: BipedalWalker          (向量环境, 24维状态 → 4维动作)
#   GPU 2: Pendulum               (图像环境, 84×84×3 → 连续动作)
#   GPU 3: CarRacing              (图像环境, 96×96×3 → 连续动作)
#
# 每个GPU运行1组实验(2个算法):
#   - SAC-PER (基线)
#   - SAC-RASPBERry (压缩版本)
#
# 使用方法:
#   ./run_sac_4gpu_parallel.sh
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
echo "  GPU 0: LunarLanderContinuous (向量)"
echo "  GPU 1: BipedalWalker (向量)"
echo "  GPU 2: Pendulum (图像)"
echo "  GPU 3: CarRacing (图像)"
echo ""
echo "每个GPU运行: SAC-PER + SAC-RASPBERry"
echo "================================================================================"
echo ""

# 用于跟踪所有进程的PID
declare -a ALL_PIDS
declare -a ALL_NAMES

################################################################################
# GPU 0: LunarLanderContinuous (向量环境)
################################################################################
echo "┌────────────────────────────────────────────────────────────────────────────┐"
echo "│ GPU 0: LunarLanderContinuous                                               │"
echo "└────────────────────────────────────────────────────────────────────────────┘"

# SAC-PER on LunarLanderContinuous
echo "  [1/2] 启动 SAC-PER-LunarLander..."
python runner/run_sac_per_algo.py \
    --env BOX2DV-LunarLanderContinuous \
    --config configs/sac_per_vector.yml \
    --gpu 0 \
    > ${LOG_DIR}/sac_per_lunarlander_${TIMESTAMP}.log 2>&1 &
PID_GPU0_PER=$!
ALL_PIDS+=($PID_GPU0_PER)
ALL_NAMES+=("GPU0-SAC-PER-LunarLander")
echo "       PID: ${PID_GPU0_PER}"

# SAC-RASPBERry on LunarLanderContinuous
echo "  [2/2] 启动 SAC-RASPBERry-LunarLander..."
python runner/run_sac_raspberry_algo.py \
    --env BOX2DV-LunarLanderContinuous \
    --config configs/sac_raspberry_vector.yml \
    --gpu 0 \
    > ${LOG_DIR}/sac_raspberry_lunarlander_${TIMESTAMP}.log 2>&1 &
PID_GPU0_RASP=$!
ALL_PIDS+=($PID_GPU0_RASP)
ALL_NAMES+=("GPU0-SAC-RASPBERry-LunarLander")
echo "       PID: ${PID_GPU0_RASP}"
echo ""

################################################################################
# GPU 1: BipedalWalker (向量环境)
################################################################################
echo "┌────────────────────────────────────────────────────────────────────────────┐"
echo "│ GPU 1: BipedalWalker                                                       │"
echo "└────────────────────────────────────────────────────────────────────────────┘"

# SAC-PER on BipedalWalker
echo "  [1/2] 启动 SAC-PER-BipedalWalker..."
python runner/run_sac_per_algo.py \
    --env BOX2DV-BipedalWalker \
    --config configs/sac_per_vector.yml \
    --gpu 1 \
    > ${LOG_DIR}/sac_per_bipedalwalker_${TIMESTAMP}.log 2>&1 &
PID_GPU1_PER=$!
ALL_PIDS+=($PID_GPU1_PER)
ALL_NAMES+=("GPU1-SAC-PER-BipedalWalker")
echo "       PID: ${PID_GPU1_PER}"

# SAC-RASPBERry on BipedalWalker
echo "  [2/2] 启动 SAC-RASPBERry-BipedalWalker..."
python runner/run_sac_raspberry_algo.py \
    --env BOX2DV-BipedalWalker \
    --config configs/sac_raspberry_vector.yml \
    --gpu 1 \
    > ${LOG_DIR}/sac_raspberry_bipedalwalker_${TIMESTAMP}.log 2>&1 &
PID_GPU1_RASP=$!
ALL_PIDS+=($PID_GPU1_RASP)
ALL_NAMES+=("GPU1-SAC-RASPBERry-BipedalWalker")
echo "       PID: ${PID_GPU1_RASP}"
echo ""

################################################################################
# GPU 2: Pendulum (图像环境)
################################################################################
echo "┌────────────────────────────────────────────────────────────────────────────┐"
echo "│ GPU 2: Pendulum                                                            │"
echo "└────────────────────────────────────────────────────────────────────────────┘"

# SAC-PER on Pendulum
echo "  [1/2] 启动 SAC-PER-Pendulum..."
python runner/run_sac_per_algo.py \
    --env Pendulum-Pendulum \
    --config configs/sac_per_image.yml \
    --gpu 2 \
    > ${LOG_DIR}/sac_per_pendulum_${TIMESTAMP}.log 2>&1 &
PID_GPU2_PER=$!
ALL_PIDS+=($PID_GPU2_PER)
ALL_NAMES+=("GPU2-SAC-PER-Pendulum")
echo "       PID: ${PID_GPU2_PER}"

# SAC-RASPBERry on Pendulum
echo "  [2/2] 启动 SAC-RASPBERry-Pendulum..."
python runner/run_sac_raspberry_algo.py \
    --env Pendulum-Pendulum \
    --config configs/sac_raspberry_image.yml \
    --gpu 2 \
    > ${LOG_DIR}/sac_raspberry_pendulum_${TIMESTAMP}.log 2>&1 &
PID_GPU2_RASP=$!
ALL_PIDS+=($PID_GPU2_RASP)
ALL_NAMES+=("GPU2-SAC-RASPBERry-Pendulum")
echo "       PID: ${PID_GPU2_RASP}"
echo ""

################################################################################
# GPU 3: CarRacing (图像环境)
################################################################################
echo "┌────────────────────────────────────────────────────────────────────────────┐"
echo "│ GPU 3: CarRacing                                                           │"
echo "└────────────────────────────────────────────────────────────────────────────┘"

# SAC-PER on CarRacing
echo "  [1/2] 启动 SAC-PER-CarRacing..."
python runner/run_sac_per_algo.py \
    --env BOX2DI-CarRacing \
    --config configs/sac_per_image.yml \
    --gpu 3 \
    > ${LOG_DIR}/sac_per_carracing_${TIMESTAMP}.log 2>&1 &
PID_GPU3_PER=$!
ALL_PIDS+=($PID_GPU3_PER)
ALL_NAMES+=("GPU3-SAC-PER-CarRacing")
echo "       PID: ${PID_GPU3_PER}"

# SAC-RASPBERry on CarRacing
echo "  [2/2] 启动 SAC-RASPBERry-CarRacing..."
python runner/run_sac_raspberry_algo.py \
    --env BOX2DI-CarRacing \
    --config configs/sac_raspberry_image.yml \
    --gpu 3 \
    > ${LOG_DIR}/sac_raspberry_carracing_${TIMESTAMP}.log 2>&1 &
PID_GPU3_RASP=$!
ALL_PIDS+=($PID_GPU3_RASP)
ALL_NAMES+=("GPU3-SAC-RASPBERry-CarRacing")
echo "       PID: ${PID_GPU3_RASP}"
echo ""

################################################################################
# 实验状态总结
################################################################################
echo "================================================================================"
echo "✅ 所有实验已启动 (共 8 个进程)"
echo "================================================================================"
echo ""
echo "GPU 分配:"
echo "  GPU 0: PID ${PID_GPU0_PER} (PER)  + ${PID_GPU0_RASP} (RASPBERry)  [LunarLander]"
echo "  GPU 1: PID ${PID_GPU1_PER} (PER)  + ${PID_GPU1_RASP} (RASPBERry)  [BipedalWalker]"
echo "  GPU 2: PID ${PID_GPU2_PER} (PER)  + ${PID_GPU2_RASP} (RASPBERry)  [Pendulum]"
echo "  GPU 3: PID ${PID_GPU3_PER} (PER)  + ${PID_GPU3_RASP} (RASPBERry)  [CarRacing]"
echo ""
echo "监控日志:"
echo "  # GPU 0 - LunarLander"
echo "  tail -f ${LOG_DIR}/sac_per_lunarlander_${TIMESTAMP}.log"
echo "  tail -f ${LOG_DIR}/sac_raspberry_lunarlander_${TIMESTAMP}.log"
echo ""
echo "  # GPU 1 - BipedalWalker"
echo "  tail -f ${LOG_DIR}/sac_per_bipedalwalker_${TIMESTAMP}.log"
echo "  tail -f ${LOG_DIR}/sac_raspberry_bipedalwalker_${TIMESTAMP}.log"
echo ""
echo "  # GPU 2 - Pendulum"
echo "  tail -f ${LOG_DIR}/sac_per_pendulum_${TIMESTAMP}.log"
echo "  tail -f ${LOG_DIR}/sac_raspberry_pendulum_${TIMESTAMP}.log"
echo ""
echo "  # GPU 3 - CarRacing"
echo "  tail -f ${LOG_DIR}/sac_per_carracing_${TIMESTAMP}.log"
echo "  tail -f ${LOG_DIR}/sac_raspberry_carracing_${TIMESTAMP}.log"
echo ""
echo "快速监控 (所有实验最新进展):"
echo "  watch -n 5 'tail -n 2 ${LOG_DIR}/*_${TIMESTAMP}.log | grep -E \"Iter|reward\"'"
echo ""
echo "检查进程状态:"
echo "  ps -p ${PID_GPU0_PER},${PID_GPU0_RASP},${PID_GPU1_PER},${PID_GPU1_RASP},${PID_GPU2_PER},${PID_GPU2_RASP},${PID_GPU3_PER},${PID_GPU3_RASP}"
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
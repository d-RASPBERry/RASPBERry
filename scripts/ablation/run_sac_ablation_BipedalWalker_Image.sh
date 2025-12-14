#!/usr/bin/env bash

################################################################################
# SAC 消融实验启动脚本 (BipedalWalker - Box2D 图像观测)
#
# 功能:
#   对每个指定 GPU 顺序启动 3 个 SAC 变体:
#     1) SAC-PER (经验回放 + PER)
#     2) SAC-PBER (分块回放, 无压缩)
#     3) SAC-RASPBERry (分块回放 + 压缩)
#
# 使用方法:
#   ./run_sac_ablation_BipedalWalker_Image.sh                # 默认 GPU 共享 (仅使用 GPU 0)
#   ./run_sac_ablation_BipedalWalker_Image.sh -n 0,1,2       # 指定逗号分隔 GPU 列表
#   ./run_sac_ablation_BipedalWalker_Image.sh -m exclusive   # 开启独占模式 (需提供3的倍数GPU)
#
################################################################################

set -euo pipefail

# 默认参数
GPU_LIST_ARG="0"
GPU_ASSIGNMENT_MODE="shared"
LAUNCH_DELAY_BETWEEN_GPUS=60
LAUNCH_DELAY_SAME_GPU=120

# Python 解释器（默认使用本机 Conda 环境；也可通过环境变量覆盖）
PYTHON_BIN="${PYTHON_BIN:-/home/seventheli/anaconda3/envs/RASPBERRY/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "错误: 找不到可执行的 Python: ${PYTHON_BIN}" >&2
    echo "请先安装/激活环境，或 export PYTHON_BIN=/path/to/python" >&2
    exit 1
fi

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
            echo "用法: $0 [-n GPU_IDS] [-h]"
            echo "选项:"
            echo "  -n GPU_IDS   逗号分隔的 GPU 编号列表 (默认: 0)"
            echo "  -m MODE      shared 或 exclusive (默认: shared)"
            echo "  -h           显示帮助信息"
            exit 0
            ;;
        \?)
            echo "无效选项: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

if ! [[ "${GPU_LIST_ARG}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "错误: -n 仅支持逗号分隔的GPU编号 (示例: 0,1,2)" >&2
    exit 1
fi

case "${GPU_ASSIGNMENT_MODE}" in
    shared|exclusive) ;;
    *)
        echo "错误: -m 仅支持 shared 或 exclusive (收到: ${GPU_ASSIGNMENT_MODE})" >&2
        exit 1
        ;;
esac

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
TOTAL_TASKS=$((NUM_GPUS * 3))
GROUP_COUNT=0

if [ "${GPU_ASSIGNMENT_MODE}" = "exclusive" ]; then
    if (( NUM_GPUS % 3 != 0 )); then
        echo "错误: exclusive 模式需要 GPU 数量为 3 的倍数 (收到 ${NUM_GPUS})" >&2
        exit 1
    fi
    GROUP_COUNT=$((NUM_GPUS / 3))
    TOTAL_TASKS=$((GROUP_COUNT * 3))
else
    GROUP_COUNT=${NUM_GPUS}
fi

# 获取项目根目录（上溯两层）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_LOG_DIR="./logs/scripts"
mkdir -p ${SCRIPT_LOG_DIR}

echo "================================================================================"
echo "🚀 启动 SAC 消融实验 (环境: BipedalWalker - Box2D 图像观测)"
echo "    目标 GPU 列表: ${GPU_IDS[*]}"
if [ "${GPU_ASSIGNMENT_MODE}" = "exclusive" ]; then
    echo "    模式: 每个实验独占单独 GPU (共 ${GROUP_COUNT} 组)"
else
    echo "    模式: 每块 GPU 依次运行 SAC-PER → SAC-PBER → SAC-RASPBERry"
fi
echo "    输出日志目录: ${SCRIPT_LOG_DIR}"
echo "    本次计划任务数: ${TOTAL_TASKS}"
echo "================================================================================"
echo ""

# 用于跟踪所有进程
declare -a ALL_PIDS
declare -a ALL_NAMES

PER_CONFIG="configs/experiments/sac/per/bipedalwalker_image.yml"
PBER_CONFIG="configs/experiments/sac/pber/bipedalwalker_image.yml"
RASP_CONFIG="configs/experiments/sac/raspberry/bipedalwalker_image.yml"

if [ "${GPU_ASSIGNMENT_MODE}" = "shared" ]; then
    for idx in "${!GPU_IDS[@]}"; do
        gpu="${GPU_IDS[$idx]}"
        echo "┌─ GPU ${gpu}: 第 $((idx+1)) 组 BipedalWalker-Image SAC 消融任务 ───────────┐"
        log_suffix="bipedalwalker_image_gpu${gpu}_${TIMESTAMP}"

        echo "  [1/3] SAC-PER 启动 (日志: ${SCRIPT_LOG_DIR}/sac_per_${log_suffix}.log)"
        "${PYTHON_BIN}" runner/run_sac_per_algo.py \
            --config ${PER_CONFIG} \
            --gpu ${gpu} \
            > ${SCRIPT_LOG_DIR}/sac_per_${log_suffix}.log 2>&1 &
        pid_per=$!
        ALL_PIDS+=(${pid_per})
        ALL_NAMES+=("GPU${gpu}-PER")
        echo "       后台 PID: ${pid_per}"
        sleep ${LAUNCH_DELAY_SAME_GPU}

        echo "  [2/3] SAC-PBER 启动 (日志: ${SCRIPT_LOG_DIR}/sac_pber_${log_suffix}.log)"
        "${PYTHON_BIN}" runner/run_sac_pber_algo.py \
            --config ${PBER_CONFIG} \
            --gpu ${gpu} \
            > ${SCRIPT_LOG_DIR}/sac_pber_${log_suffix}.log 2>&1 &
        pid_pber=$!
        ALL_PIDS+=(${pid_pber})
        ALL_NAMES+=("GPU${gpu}-PBER")
        echo "       后台 PID: ${pid_pber}"
        sleep ${LAUNCH_DELAY_SAME_GPU}

        echo "  [3/3] SAC-RASPBERry 启动 (日志: ${SCRIPT_LOG_DIR}/sac_raspberry_${log_suffix}.log)"
        "${PYTHON_BIN}" runner/run_sac_raspberry_algo.py \
            --config ${RASP_CONFIG} \
            --gpu ${gpu} \
            > ${SCRIPT_LOG_DIR}/sac_raspberry_${log_suffix}.log 2>&1 &
        pid_rasp=$!
        ALL_PIDS+=(${pid_rasp})
        ALL_NAMES+=("GPU${gpu}-RASPBERry")
        echo "       后台 PID: ${pid_rasp}"

        if [ ${idx} -lt $((NUM_GPUS - 1)) ]; then
            echo "       ⏳ 等待 ${LAUNCH_DELAY_BETWEEN_GPUS} 秒后启动下一组..."
            sleep ${LAUNCH_DELAY_BETWEEN_GPUS}
        fi
        echo "└────────────────────────────────────────────────────────────────────────┘"
        echo ""
    done
else
    for ((group_idx=0; group_idx<GROUP_COUNT; group_idx++)); do
        base=$((group_idx * 3))
        gpu_per=${GPU_IDS[$base]}
        gpu_pber=${GPU_IDS[$((base + 1))]}
        gpu_rasp=${GPU_IDS[$((base + 2))]}
        echo "┌─ 第 $((group_idx + 1)) 组 BipedalWalker-Image SAC 消融任务 (独占模式) ───────┐"
        log_suffix="bipedalwalker_image_group$((group_idx + 1))_${TIMESTAMP}"

        echo "  [PER] 使用 GPU ${gpu_per} (日志: ${SCRIPT_LOG_DIR}/sac_per_${log_suffix}.log)"
        "${PYTHON_BIN}" runner/run_sac_per_algo.py \
            --config ${PER_CONFIG} \
            --gpu ${gpu_per} \
            > ${SCRIPT_LOG_DIR}/sac_per_${log_suffix}.log 2>&1 &
        pid_per=$!
        ALL_PIDS+=(${pid_per})
        ALL_NAMES+=("GPU${gpu_per}-PER(G$((group_idx + 1)))")
        echo "       后台 PID: ${pid_per}"

        echo "  [PBER] 使用 GPU ${gpu_pber} (日志: ${SCRIPT_LOG_DIR}/sac_pber_${log_suffix}.log)"
        "${PYTHON_BIN}" runner/run_sac_pber_algo.py \
            --config ${PBER_CONFIG} \
            --gpu ${gpu_pber} \
            > ${SCRIPT_LOG_DIR}/sac_pber_${log_suffix}.log 2>&1 &
        pid_pber=$!
        ALL_PIDS+=(${pid_pber})
        ALL_NAMES+=("GPU${gpu_pber}-PBER(G$((group_idx + 1)))")
        echo "       后台 PID: ${pid_pber}"

        echo "  [RASPBERry] 使用 GPU ${gpu_rasp} (日志: ${SCRIPT_LOG_DIR}/sac_raspberry_${log_suffix}.log)"
        "${PYTHON_BIN}" runner/run_sac_raspberry_algo.py \
            --config ${RASP_CONFIG} \
            --gpu ${gpu_rasp} \
            > ${SCRIPT_LOG_DIR}/sac_raspberry_${log_suffix}.log 2>&1 &
        pid_rasp=$!
        ALL_PIDS+=(${pid_rasp})
        ALL_NAMES+=("GPU${gpu_rasp}-RASPBERry(G$((group_idx + 1)))")
        echo "       后台 PID: ${pid_rasp}"

        if [ ${group_idx} -lt $((GROUP_COUNT - 1)) ]; then
            echo "       ⏳ 等待 ${LAUNCH_DELAY_BETWEEN_GPUS} 秒后启动下一组..."
            sleep ${LAUNCH_DELAY_BETWEEN_GPUS}
        fi
        echo "└────────────────────────────────────────────────────────────────────────┘"
        echo ""
    done
fi

echo "✅ 已提交 ${TOTAL_TASKS} 个 SAC 后台任务"
for idx in "${!ALL_PIDS[@]}"; do
    printf "  %-20s -> PID:%s\n" "${ALL_NAMES[$idx]}" "${ALL_PIDS[$idx]}"
done
echo ""
echo "日志目录: ${SCRIPT_LOG_DIR}"
echo "监控建议: watch -n 2 'nvidia-smi'"
echo "终止全部: kill ${ALL_PIDS[@]}"
echo ""

echo "⏳ 等待所有任务结束..."
for idx in "${!ALL_PIDS[@]}"; do
    wait ${ALL_PIDS[$idx]} 2>/dev/null || echo "  ⚠️  进程 ${ALL_PIDS[$idx]} (${ALL_NAMES[$idx]}) 已退出"
done

echo "🎉 完成!"


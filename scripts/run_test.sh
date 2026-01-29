#!/usr/bin/env bash

################################################################################
# Unified test launcher - APEX/DDQN/SAC (PBER + RASPBERry)
#
# Runs the following experiments:
#  - APEX: PBER + RASPBERry on Pong
#  - DDQN: PBER + RASPBERry on Pong
#  - SAC : PBER + RASPBERry on LunarLanderContinuous
#
# All jobs use the default GPU (id 0).
# Executes up to 4 jobs in parallel.
################################################################################

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    conda activate RASPBERRY
else
    echo "Error: conda.sh not found at ${HOME}/anaconda3/etc/profile.d/conda.sh" >&2
    exit 1
fi

GPU_ID="0"
MAX_CONCURRENT=4

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_LOG_DIR="${PROJECT_ROOT}/logs/test"
mkdir -p "${SCRIPT_LOG_DIR}"

TASKS=(
    "APEX-PBER-Pong|runner/run_apex_pber_algo.py|configs/apex_pber_atari.yml|Atari-PongNoFrameskip-v4"
    "APEX-RASPBERry-Pong|runner/run_apex_raspberry_algo.py|configs/apex_raspberry_atari.yml|Atari-PongNoFrameskip-v4"
    "DDQN-PBER-Pong|runner/run_ddqn_pber_algo.py|configs/experiments/ddqn/pber/pong.yml|Atari-PongNoFrameskip-v4"
    "DDQN-RASPBERry-Pong|runner/run_ddqn_raspberry_algo.py|configs/experiments/ddqn/raspberry/pong.yml|Atari-PongNoFrameskip-v4"
    "SAC-PBER-LunarLander|runner/run_sac_pber_algo.py|configs/experiments/sac/pber/lunarlander.yml|BOX2DV-LunarLanderContinuous"
    "SAC-RASPBERry-LunarLander|runner/run_sac_raspberry_algo.py|configs/experiments/sac/raspberry/lunarlander.yml|BOX2DV-LunarLanderContinuous"
)

declare -a PIDS
declare -a NAMES

trim_dead_pids() {
    local -a new_pids=()
    local -a new_names=()
    for idx in "${!PIDS[@]}"; do
        local pid="${PIDS[$idx]}"
        if kill -0 "${pid}" 2>/dev/null; then
            new_pids+=("${pid}")
            new_names+=("${NAMES[$idx]}")
        fi
    done
    PIDS=("${new_pids[@]}")
    NAMES=("${new_names[@]}")
}

wait_for_slot() {
    while true; do
        trim_dead_pids
        if [ "${#PIDS[@]}" -lt "${MAX_CONCURRENT}" ]; then
            break
        fi
        sleep 5
    done
}

echo "================================================================================"
echo "Unified test launcher"
echo "GPU: ${GPU_ID} | Max concurrent: ${MAX_CONCURRENT}"
echo "Log dir: ${SCRIPT_LOG_DIR}"
echo "================================================================================"

for task in "${TASKS[@]}"; do
    IFS='|' read -r name script config env_name <<< "${task}"
    log_file="${SCRIPT_LOG_DIR}/${name}_gpu${GPU_ID}_${TIMESTAMP}.log"
    cmd="python ${script} --config ${config} --env ${env_name} --gpu ${GPU_ID}"

    wait_for_slot
    echo "[Launch] ${name} | GPU ${GPU_ID} | log: ${log_file}"
    ${cmd} > "${log_file}" 2>&1 &
    PIDS+=("$!")
    NAMES+=("${name}")
done

echo ""
echo "Submitted ${#TASKS[@]} tasks. Waiting for completion..."
for pid in "${PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
done
echo "Done."

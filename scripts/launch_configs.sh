#!/usr/bin/env bash

################################################################################
# Generic config launcher
################################################################################

set -euo pipefail

GPU_LIST_ARG="0"
TASKS_PER_GPU=1
CONFIG_LIST_FILE=""
LAUNCH_DELAY=5
RUNNER_OVERRIDE=""
ALGORITHM="auto"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN=0
FAILED_TASKS=0

usage() {
    echo "Usage: $0 [-a auto|ddqn|apex|sac] [-n GPU_IDS] [-j TASKS_PER_GPU] [-f CONFIG_LIST] [-r RUNNER] [-p PYTHON] [-d SECONDS] [-y] CONFIG..."
    echo ""
    echo "Options:"
    echo "  -a ALGORITHM       Validate configs against one algorithm, default: auto"
    echo "  -n GPU_IDS         Comma-separated GPU IDs, default: 0"
    echo "  -j TASKS_PER_GPU   Max concurrent tasks per GPU, default: 1"
    echo "  -f CONFIG_LIST     File containing one config path per line"
    echo "  -r RUNNER          Optional runner override for all configs"
    echo "  -p PYTHON          Python executable, default: \$PYTHON_BIN or python"
    echo "  -d SECONDS         Delay between launches, default: 5"
    echo "  -y                 Dry run: print scheduled commands without launching"
    echo ""
    echo "Examples:"
    echo "  $0 -a ddqn -n 0 -j 3 configs/experiments/seeded/ddqn/per/freeway_seed0.yml \\"
    echo "     configs/experiments/seeded/ddqn/pber/freeway_seed0.yml \\"
    echo "     configs/experiments/seeded/ddqn/raspberry/freeway_seed0.yml"
    echo ""
    echo "  $0 -a apex -n 0,1 -j 1 -f /tmp/configs.txt"
}

while getopts "a:n:j:f:r:p:d:yh" opt; do
    case "${opt}" in
        a) ALGORITHM="${OPTARG}" ;;
        n) GPU_LIST_ARG="${OPTARG}" ;;
        j) TASKS_PER_GPU="${OPTARG}" ;;
        f) CONFIG_LIST_FILE="${OPTARG}" ;;
        r) RUNNER_OVERRIDE="${OPTARG}" ;;
        p) PYTHON_BIN="${OPTARG}" ;;
        d) LAUNCH_DELAY="${OPTARG}" ;;
        y) DRY_RUN=1 ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option" >&2; usage >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

if ! [[ "${GPU_LIST_ARG}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Error: -n only supports comma-separated GPU IDs (e.g., 0,1,2)" >&2
    exit 1
fi
if ! [[ "${TASKS_PER_GPU}" =~ ^[0-9]+$ ]] || [ "${TASKS_PER_GPU}" -lt 1 ]; then
    echo "Error: -j must be a positive integer" >&2
    exit 1
fi
if ! [[ "${LAUNCH_DELAY}" =~ ^[0-9]+$ ]]; then
    echo "Error: -d must be a non-negative integer" >&2
    exit 1
fi
case "${ALGORITHM}" in
    auto|ddqn|apex|sac) ;;
    *) echo "Error: -a only supports auto, ddqn, apex, or sac" >&2; exit 1 ;;
esac
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Error: python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

IFS=',' read -ra GPU_IDS <<< "${GPU_LIST_ARG}"
if [ ${#GPU_IDS[@]} -eq 0 ]; then
    echo "Error: at least one GPU ID is required" >&2
    exit 1
fi
for gpu_id in "${GPU_IDS[@]}"; do
    if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "Error: invalid GPU ID ${gpu_id}" >&2
        exit 1
    fi
done

declare -a CONFIGS
if [ -n "${CONFIG_LIST_FILE}" ]; then
    [ -f "${CONFIG_LIST_FILE}" ] || { echo "Error: missing config list ${CONFIG_LIST_FILE}" >&2; exit 1; }
    while IFS= read -r line || [ -n "${line}" ]; do
        case "${line}" in
            ""|\#*) continue ;;
            *) CONFIGS+=("${line}") ;;
        esac
    done < "${CONFIG_LIST_FILE}"
fi
for cfg in "$@"; do
    CONFIGS+=("${cfg}")
done
if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "Error: at least one config path is required" >&2
    usage >&2
    exit 1
fi

algorithm_for_config() {
    local cfg="$1"

    case "${cfg}" in
        */ddqn/*|configs/ddqn_*.yml) echo "ddqn" ;;
        */apex/*|configs/apex_*.yml|configs/experiments/sub/apex_raspberry_*) echo "apex" ;;
        */sac/*|configs/sac_*.yml) echo "sac" ;;
        *)
            echo "Error: cannot infer algorithm for config ${cfg}" >&2
            return 1
            ;;
    esac
}

runner_for_config() {
    local cfg="$1"

    if [ -n "${RUNNER_OVERRIDE}" ]; then
        echo "${RUNNER_OVERRIDE}"
        return 0
    fi

    case "${cfg}" in
        */ddqn/per/*|configs/ddqn_per*.yml) echo "runner/run_ddqn_per_algo.py" ;;
        */ddqn/pber/*|configs/ddqn_pber*.yml) echo "runner/run_ddqn_pber_algo.py" ;;
        */ddqn/raspberry/*|configs/ddqn_raspberry*.yml) echo "runner/run_ddqn_raspberry_algo.py" ;;
        */apex/per/*|configs/apex_per*.yml) echo "runner/run_apex_per_algo.py" ;;
        */apex/pber/*|configs/apex_pber*.yml) echo "runner/run_apex_pber_algo.py" ;;
        */apex/raspberry/*|configs/apex_raspberry*.yml) echo "runner/run_apex_raspberry_algo.py" ;;
        */sac/per/*|configs/sac_per*.yml) echo "runner/run_sac_per_algo.py" ;;
        */sac/pber/*|configs/sac_pber*.yml) echo "runner/run_sac_pber_algo.py" ;;
        */sac/raspberry/*|configs/sac_raspberry*.yml) echo "runner/run_sac_raspberry_algo.py" ;;
        configs/experiments/sub/apex_raspberry_*) echo "runner/run_apex_raspberry_algo.py" ;;
        *)
            echo "Error: cannot infer runner for config ${cfg}; pass -r RUNNER" >&2
            return 1
            ;;
    esac
}

for cfg in "${CONFIGS[@]}"; do
    [ -f "${cfg}" ] || { echo "Error: missing config ${cfg}" >&2; exit 1; }
    if [ "${ALGORITHM}" != "auto" ]; then
        config_algorithm="$(algorithm_for_config "${cfg}")"
        if [ "${config_algorithm}" != "${ALGORITHM}" ]; then
            echo "Error: ${cfg} is ${config_algorithm}, but launcher algorithm is ${ALGORITHM}" >&2
            exit 1
        fi
    fi
    runner="$(runner_for_config "${cfg}")"
    [ -f "${runner}" ] || { echo "Error: missing runner ${runner}" >&2; exit 1; }
done

declare -a ALL_PIDS
declare -a ALL_NAMES

terminate_children() {
    if [ ${#ALL_PIDS[@]} -gt 0 ]; then
        echo ""
        echo "Terminating submitted tasks..."
        kill "${ALL_PIDS[@]}" 2>/dev/null || true
    fi
}
trap terminate_children INT TERM

slot_gpu() {
    local slot="$1"
    local gpu_idx=$((slot % ${#GPU_IDS[@]}))
    echo "${GPU_IDS[$gpu_idx]}"
}

launch_config() {
    local cfg="$1"
    local gpu="$2"
    local runner
    runner="$(runner_for_config "${cfg}")"

    if [ "${DRY_RUN}" -eq 1 ]; then
        printf "  [dry-run] GPU=%s runner=%s config=%s\n" "${gpu}" "${runner}" "${cfg}"
        return 0
    fi

    "${PYTHON_BIN}" "${runner}" --config "${cfg}" --gpu "${gpu}" &
    local pid=$!
    ALL_PIDS+=("${pid}")
    ALL_NAMES+=("GPU${gpu}:${runner}:${cfg}")
    printf "  GPU=%s PID=%s runner=%s config=%s\n" "${gpu}" "${pid}" "${runner}" "${cfg}"
}

wait_for_wave() {
    if [ "${DRY_RUN}" -eq 1 ]; then
        return 0
    fi
    if [ ${#ALL_PIDS[@]} -eq 0 ]; then
        return 0
    fi
    local wave_failed=0
    echo ""
    echo "Waiting for current wave..."
    for idx in "${!ALL_PIDS[@]}"; do
        local pid="${ALL_PIDS[$idx]}"
        local name="${ALL_NAMES[$idx]}"
        if wait "${pid}" 2>/dev/null; then
            printf "  OK   PID=%s %s\n" "${pid}" "${name}"
        else
            local rc=$?
            printf "  FAIL PID=%s rc=%s %s\n" "${pid}" "${rc}" "${name}" >&2
            FAILED_TASKS=$((FAILED_TASKS + 1))
            wave_failed=1
        fi
    done
    ALL_PIDS=()
    ALL_NAMES=()
    return "${wave_failed}"
}

TOTAL_SLOTS=$((${#GPU_IDS[@]} * TASKS_PER_GPU))
TOTAL_TASKS=${#CONFIGS[@]}

echo "================================================================================"
echo "Config launcher"
echo "GPUs: ${GPU_IDS[*]} | tasks_per_gpu: ${TASKS_PER_GPU} | total_slots: ${TOTAL_SLOTS}"
echo "Algorithm: ${ALGORITHM} | Tasks: ${TOTAL_TASKS} | Python: ${PYTHON_BIN}"
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "Mode: dry-run"
fi
echo "================================================================================"

for idx in "${!CONFIGS[@]}"; do
    if (( idx > 0 && idx % TOTAL_SLOTS == 0 )); then
        if ! wait_for_wave; then
            echo "Error: current wave failed; not launching remaining configs" >&2
            exit 1
        fi
        echo ""
        echo "Launching next wave..."
    fi

    slot=$((idx % TOTAL_SLOTS))
    gpu="$(slot_gpu "${slot}")"
    launch_config "${CONFIGS[$idx]}" "${gpu}"

    if [ "${DRY_RUN}" -eq 0 ] && [ "${LAUNCH_DELAY}" -gt 0 ] && [ "${idx}" -lt $((TOTAL_TASKS - 1)) ]; then
        sleep "${LAUNCH_DELAY}"
    fi
done

if ! wait_for_wave; then
    echo "Error: ${FAILED_TASKS} task(s) failed" >&2
    exit 1
fi
echo "Done."

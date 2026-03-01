#!/usr/bin/env bash
# GPU group 1 — LavaCrossingS9N1 seed1  m={8,16,32}
set -euo pipefail

GPU=""
while getopts "n:h" opt; do
    case "${opt}" in
        n) GPU="${OPTARG}" ;;
        h) echo "Usage: $0 -n GPU_ID"; exit 0 ;;
        \?) echo "Invalid option" >&2; exit 1 ;;
    esac
done
if [[ -z "${GPU}" ]]; then echo "Error: -n GPU_ID is required." >&2; exit 1; fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUNNER="runner/run_apex_raspberry_algo.py"
CFG="configs/experiments/sub"
DELAY=60

echo "[group1] GPU=${GPU}  lava seed1 m={8,16,32}"

python ${RUNNER} --config ${CFG}/apex_raspberry_lavacrossing_s9n1_m8_seed1.yml  --gpu "${GPU}"
sleep ${DELAY}
python ${RUNNER} --config ${CFG}/apex_raspberry_lavacrossing_s9n1_m16_seed1.yml --gpu "${GPU}"
sleep ${DELAY}
python ${RUNNER} --config ${CFG}/apex_raspberry_lavacrossing_s9n1_m32_seed1.yml --gpu "${GPU}"

echo "[group1] Done."

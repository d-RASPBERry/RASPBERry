#!/usr/bin/env bash
# GPU group 4 — Freeway seed1  m={16,32,64}
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

echo "[group4] GPU=${GPU}  freeway seed1 m={16,32,64}"

python ${RUNNER} --config ${CFG}/apex_raspberry_freeway_m16_seed1.yml --gpu "${GPU}"
sleep ${DELAY}
python ${RUNNER} --config ${CFG}/apex_raspberry_freeway_m32_seed1.yml --gpu "${GPU}"
sleep ${DELAY}
python ${RUNNER} --config ${CFG}/apex_raspberry_freeway_m64_seed1.yml --gpu "${GPU}"

echo "[group4] Done."

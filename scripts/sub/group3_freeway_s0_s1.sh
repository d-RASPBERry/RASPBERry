#!/usr/bin/env bash
# GPU group 3 — Freeway seed0 m={32,64}  +  Freeway seed1 m=8
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

echo "[group3] GPU=${GPU}  freeway-s0-m32 / freeway-s0-m64 / freeway-s1-m8"

python ${RUNNER} --config ${CFG}/apex_raspberry_freeway_m32_seed0.yml --gpu "${GPU}"
sleep ${DELAY}
python ${RUNNER} --config ${CFG}/apex_raspberry_freeway_m64_seed0.yml --gpu "${GPU}"
sleep ${DELAY}
python ${RUNNER} --config ${CFG}/apex_raspberry_freeway_m8_seed1.yml  --gpu "${GPU}"

echo "[group3] Done."

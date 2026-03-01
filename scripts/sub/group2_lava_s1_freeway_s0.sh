#!/usr/bin/env bash
# GPU group 2 — LavaCrossing seed1 m=64  +  Freeway seed0 m={8,16}
set -euo pipefail

GPU="${1:?Usage: $0 GPU_ID}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUNNER="runner/run_apex_raspberry_algo.py"
CFG="configs/experiments/sub"
DELAY=60

echo "[group2] GPU=${GPU}  lava-s1-m64 / freeway-s0-m8 / freeway-s0-m16"

python ${RUNNER} --config ${CFG}/apex_raspberry_lavacrossing_s9n1_m64_seed1.yml --gpu "${GPU}"
sleep ${DELAY}
python ${RUNNER} --config ${CFG}/apex_raspberry_freeway_m8_seed0.yml            --gpu "${GPU}"
sleep ${DELAY}
python ${RUNNER} --config ${CFG}/apex_raspberry_freeway_m16_seed0.yml           --gpu "${GPU}"

echo "[group2] Done."

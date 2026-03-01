#!/usr/bin/env bash
# GPU group 3 — Freeway seed0 m={32,64}  +  Freeway seed1 m=8
set -euo pipefail

GPU="${1:?Usage: $0 GPU_ID}"
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

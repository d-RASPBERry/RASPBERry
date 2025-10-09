#!/usr/bin/env bash
set -euo pipefail

# 2h budget per run comparing PER vs RASPBERry-Ray for SAC
# - SAC-PER: RLlib native PER (baseline)
# - SAC-RASPBERry-Ray: Block replay + Ray-based compression (d_raspberry_ray.py)

ROOT_DIR="/home/seventheli/research/RASPBERry"
CONDA_ENV_NAME="RASPBERRY"

# === 运行参数：所有参数从 YAML 配置读取 === #
# max_time_s 和 max_iterations 在各自的 config.yaml 中配置

LOG_DIR="$ROOT_DIR/tests/logs/compare_sac"
MONITOR_SLEEP_SECONDS=1800

mkdir -p "$LOG_DIR"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

ensure_log_dir() {
  local target_path=$1
  local parent_dir
  parent_dir=$(dirname "$target_path")
  mkdir -p "$parent_dir"
}

init_log_file() {
  local target_path=$1
  ensure_log_dir "$target_path"
  printf "LOG_PATH=%s\n\n" "$target_path" > "$target_path"
}

run_with_env() {
  local log_file=$1
  shift
  (
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
    ensure_log_dir "$log_file"
    nohup "$@" >> "$log_file" 2>&1 &
    echo $!
  )
}

PER_LOG="$LOG_DIR/sac_per_stdout.log"
RASPBERRY_RAY_LOG="$LOG_DIR/sac_raspberry_ray_stdout.log"
MONITOR_LOG="$LOG_DIR/sac_monitor.log"
PIDS_FILE="$LOG_DIR/sac_pids.txt"

init_log_file "$PER_LOG"
init_log_file "$RASPBERRY_RAY_LOG"
init_log_file "$MONITOR_LOG"

PER_PID=$(run_with_env "$PER_LOG" python "$ROOT_DIR/tests/run_sac_per_algo.py")
echo "Started SAC-PER PID=$PER_PID (reading config from YAML)"

RASPBERRY_RAY_PID=$(run_with_env "$RASPBERRY_RAY_LOG" python "$ROOT_DIR/tests/run_sac_raspberry_algo.py")
echo "Started SAC-RASPBERry-Ray PID=$RASPBERRY_RAY_PID (reading config from YAML)"

printf "LOG_PATH=%s\nPER=%s RASP_RAY=%s\n" "$PIDS_FILE" "$PER_PID" "$RASPBERRY_RAY_PID" > "$PIDS_FILE"

(
  while kill -0 $PER_PID 2>/dev/null || kill -0 $RASPBERRY_RAY_PID 2>/dev/null; do
    date
    ps -o pid,pcpu,pmem,etime,cmd -p $PER_PID -p $RASPBERRY_RAY_PID || true
    sleep "$MONITOR_SLEEP_SECONDS"
  done
) >> "$MONITOR_LOG" 2>&1 &
MON_PID=$!

echo "Monitor PID=$MON_PID (tracking PER and RASPBERry-Ray)"


#!/bin/bash
# SAC 对比实验：PER vs RASPBERry-Ray
# 运行时间：2小时

set -e

LOG_DIR="tests/logs/compare_sac"
mkdir -p "$LOG_DIR"

echo "开始 SAC 对比实验（2小时）"
echo "================================"

# 启动 SAC-PER
echo "启动 SAC-PER..."
nohup python tests/run_sac_per_algo.py \
    --max-time 7200 \
    > "$LOG_DIR/sac_per_stdout.log" 2>&1 &
PER_PID=$!
echo "Started SAC-PER PID=$PER_PID (reading config from YAML)"

# 启动 SAC-RASPBERry-Ray
echo "启动 SAC-RASPBERry-Ray..."
nohup python tests/run_sac_raspberry_algo.py \
    --max-time 7200 \
    > "$LOG_DIR/sac_raspberry_ray_stdout.log" 2>&1 &
RASP_PID=$!
echo "Started SAC-RASPBERry-Ray PID=$RASP_PID (reading config from YAML)"

# 启动监控进程
echo "启动监控进程..."
nohup bash -c "
LOG_PATH='$LOG_DIR/sac_monitor.log'
echo 'LOG_PATH=$LOG_PATH' > \$LOG_PATH
echo '' >> \$LOG_PATH
while kill -0 $PER_PID 2>/dev/null || kill -0 $RASP_PID 2>/dev/null; do
    date >> \$LOG_PATH
    ps -p $PER_PID,$RASP_PID -o pid,%cpu,%mem,etime,cmd --no-headers 2>/dev/null >> \$LOG_PATH || true
    sleep 1800  # 每30分钟记录一次
done
" &
MONITOR_PID=$!
echo "Monitor PID=$MONITOR_PID (tracking PER and RASPBERry-Ray)"


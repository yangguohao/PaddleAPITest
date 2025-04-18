#!/bin/bash

# 配置参数
INPUT_FILE="tester/api_config/api_config_merged_temp1.txt"
LOG_DIR="tester/api_config/test_log"
NUM_GPUS=8

mkdir -p "$LOG_DIR"

# 执行程序
python engine.py --accuracy=True \
        --api_config_file="$INPUT_FILE" \
        --num_gpus=$NUM_GPUS \
        >> "$LOG_DIR/log.log" 2>&1 &

PYTHON_PID=$!

echo -e "\n\033[32m执行中... 另开终端运行监控:\033[0m"
echo -e "1. GPU使用:   watch -n 1 nvidia-smi"
echo -e "2. 详细日志:  ls -lh $LOG_DIR"

wait $PYTHON_PID

# 终止所有任务
# ctrl+c 或
# pkill -f 'python engine.py'

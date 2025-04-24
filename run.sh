#!/bin/bash

# 配置参数
INPUT_FILE="tester/api_config/api_config_support2torch_1.txt"
LOG_DIR="tester/api_config/test_log"
NUM_GPUS=8
NUM_WORKERS_PER_GPU=1

mkdir -p "$LOG_DIR" || { echo "无法创建日志目录 $LOG_DIR"; exit 1; }

# 执行程序
nohup python engineV2.py --accuracy=True \
        --api_config_file="$INPUT_FILE" \
        --num_gpus=$NUM_GPUS \
        --num_workers_per_gpu=$NUM_WORKERS_PER_GPU \
        >> "$LOG_DIR/log.log" 2>&1 &

PYTHON_PID=$!

echo -e "\n\033[32m执行中... 另开终端运行监控:\033[0m"
echo -e "1. GPU使用:   watch -n 1 nvidia-smi"
echo -e "2. 详细日志:  ls -lh $LOG_DIR"
echo -e "3. 终止任务:  kill $PYTHON_PID"
echo -e "\n进程已在后台运行，关闭终端不会影响进程执行。"

exit 0

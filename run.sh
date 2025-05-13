#!/bin/bash

# 配置参数
# NUM_GPUS!=0 时，engineV2 不受外部 "CUDA_VISIBLE_DEVICES" 影响
INPUT_FILE="tester/api_config/mytmp.txt"
LOG_DIR="tester/api_config/test_log"
NUM_GPUS=-1
NUM_WORKERS_PER_GPU=-1
GPU_IDS="0,1,2,3"
REQUIRED_MEMORY=10

mkdir -p "$LOG_DIR" || { echo "无法创建日志目录 $LOG_DIR"; exit 1; }

# 执行程序
nohup python3.10 engineV2.py --accuracy=True \
        --api_config_file="$INPUT_FILE" \
        --num_gpus=$NUM_GPUS \
        --num_workers_per_gpu=$NUM_WORKERS_PER_GPU \
        >> "$LOG_DIR/log.log" 2>&1 &

# --gpu_ids="$GPU_IDS" \
# --required_memory=$REQUIRED_MEMORY \

PYTHON_PID=$!

echo -e "\n\033[32m执行中... 另开终端运行监控:\033[0m"
echo -e "1. GPU使用:   watch -n 1 nvidia-smi"
echo -e "2. 详细日志:  ls -lh $LOG_DIR"
echo -e "3. 终止任务:  kill $PYTHON_PID"
echo -e "\n进程已在后台运行，关闭终端不会影响进程执行。"

exit 0

# watch -n 1 nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv

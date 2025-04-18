#!/bin/bash

# 配置参数
SPLIT_DIR="tester/api_config/split_parts"
LOG_DIR="tester/api_config/split_test_log"
JOBLOG="$LOG_DIR/parallel_joblog.txt"

# 准备目录
mkdir -p "$LOG_DIR" || { echo "无法创建目录"; exit 1; }

# 并行执行（使用CUDA_VISIBLE_DEVICES指定GPU）
parallel --verbose -j8 --joblog "$JOBLOG" --halt now,fail=1 --link\
    "CUDA_VISIBLE_DEVICES={1} python engine.py --accuracy=True \
    --api_config_file=\"$SPLIT_DIR/part_{2}\" \
    >> \"$LOG_DIR/gpu_{1}.log\" 2>&1" ::: $(seq 0 7) ::: $(seq -w 00 07) &

PARALLEL_PID=$!

echo -e "\n\033[32m执行中... 另开终端运行监控:\033[0m"
echo -e "1. GPU使用:   watch -n 1 nvidia-smi"
echo -e "2. 任务进度:  tail -f $JOBLOG"
echo -e "3. 详细日志:  ls -lh $LOG_DIR/gpu_*.log"

wait $PARALLEL_PID

# 清理临时文件（可选）
# rm -rf "$SPLIT_DIR"

# 终止所有任务
# pkill -f 'parallel.*engine.py'
# 或
# pkill -f 'python engine.py'

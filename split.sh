#!/bin/bash

# 配置参数
INPUT_FILE="tester/api_config/api_config_merged_temp.txt"
SPLIT_DIR="tester/api_config/split_parts"

# 准备目录
mkdir -p "$SPLIT_DIR" || { echo "无法创建目录"; exit 1; }

# 分割文件
echo "正在分割输入文件..."
TOTAL_LINES=$(wc -l < "$INPUT_FILE")
LINES_PER_PART=$(( ($TOTAL_LINES + 7) / 8 ))
split -d -l $LINES_PER_PART "$INPUT_FILE" "$SPLIT_DIR/part_"

echo "已生成分割文件："
ls "${SPLIT_DIR}"/part_*

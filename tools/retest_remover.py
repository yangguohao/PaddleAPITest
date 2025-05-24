# 重测配置移除小工具（timeout、crash、oom、skip）
# @author: cangtianhuang

import os
from pathlib import Path

TEST_LOG_PATH = Path("tester/api_config/test_log")

LOG_FILES = [
    "api_config_timeout",
    "api_config_crash",
    "api_config_oom",
    "api_config_skip",
]

api_configs = set()
checkpoint_file = TEST_LOG_PATH / "checkpoint.txt"
if not checkpoint_file.exists():
    exit(0)
try:
    with checkpoint_file.open("r") as f:
        api_configs = set(line.strip() for line in f if line.strip())
        print(f"checkpoint remaining: {len(api_configs)}", flush=True)
except Exception as err:
    print(f"Error reading {checkpoint_file}: {err}", flush=True)

retest_configs = set()
for log_name in LOG_FILES:
    log_file = TEST_LOG_PATH / f"{log_name}.txt"
    if not log_file.exists():
        continue
    try:
        with log_file.open("r") as f:
            lines = set(line.strip() for line in f if line.strip())
            retest_configs.update(lines)
            print(f"{log_name} remaining: {len(lines)}", flush=True)
    except Exception as err:
        print(f"Error reading {log_file}: {err}", flush=True)

if retest_configs:
    try:
        with checkpoint_file.open("r") as f:
            lines = set(line.strip() for line in f if line.strip())
            lines -= retest_configs
            print(f"checkpoint remaining: {len(lines)}", flush=True)
        with checkpoint_file.open("w") as f:
            f.writelines(f"{line}\n" for line in sorted(lines))
    except Exception as err:
        print(f"Error reading {checkpoint_file}: {err}", flush=True)
else:
    print("No retest config(s) found.", flush=True)

for log_name in LOG_FILES:
    log_file = TEST_LOG_PATH / f"{log_name}.txt"
    if not log_file.exists():
        continue
    try:
        os.remove(log_file)
    except Exception as err:
        print(f"Error removing {log_file}: {err}", flush=True)

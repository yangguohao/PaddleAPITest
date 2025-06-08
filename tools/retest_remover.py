# 重测配置移除小工具（timeout、crash、oom、skip）
# @author: cangtianhuang
# @date: 2025-06-08

import os
from pathlib import Path

TEST_LOG_PATH = Path("tester/api_config/test_log")

LOG_PREFIXES = {
    "timeout": "api_config_timeout",
    "crash": "api_config_crash",
    "oom": "api_config_oom",
    "skip": "api_config_skip",
}

checkpoint_configs = set()
checkpoint_file = TEST_LOG_PATH / "checkpoint.txt"
if not checkpoint_file.exists():
    print("No checkpoint file found", flush=True)
    exit(0)

try:
    with checkpoint_file.open("r") as f:
        checkpoint_configs = set(line.strip() for line in f if line.strip())
except Exception as err:
    print(f"Error reading {checkpoint_file}: {err}", flush=True)
    exit(0)
print(f"Read {len(checkpoint_configs)} api config(s) from checkpoint", flush=True)

retest_configs = set()
for log_type, prefix in LOG_PREFIXES.items():
    log_file = TEST_LOG_PATH / f"{prefix}.txt"
    if not log_file.exists():
        continue
    try:
        with log_file.open("r") as f:
            lines = set(line.strip() for line in f if line.strip())
            retest_configs.update(lines)
            print(f"Read {len(lines)} api config(s) from {log_file}", flush=True)
    except Exception as err:
        print(f"Error reading {log_file}: {err}", flush=True)
        exit(0)

if retest_configs:
    checkpoint_count = len(checkpoint_configs)
    checkpoint_configs -= retest_configs
    print(
        f"checkpoint removed: {checkpoint_count - len(checkpoint_configs)}", flush=True
    )
    print(f"checkpoint remaining: {len(checkpoint_configs)}", flush=True)
    try:
        with checkpoint_file.open("w") as f:
            f.writelines(f"{line}\n" for line in sorted(checkpoint_configs))
    except Exception as err:
        print(f"Error writing {checkpoint_file}: {err}", flush=True)
        exit(0)
else:
    print("No retest config(s) found", flush=True)

for prefix in LOG_PREFIXES.values():
    log_file = TEST_LOG_PATH / f"{prefix}.txt"
    if not log_file.exists():
        continue
    try:
        os.remove(log_file)
    except Exception as err:
        print(f"Error removing {log_file}: {err}", flush=True)
        exit(0)

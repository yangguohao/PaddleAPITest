# 检查重复配置小工具
# @author: cangtianhaung

from collections import defaultdict
from pathlib import Path

TEST_LOG_PATH = Path("tester/api_config/test_log")

LOG_PREFIXES = {
    "checkpoint": "checkpoint",
    "pass": "api_config_pass",
    "paddle_error": "api_config_paddle_error",
    "torch_error": "api_config_torch_error",
    "paddle_to_torch_failed": "api_config_paddle_to_torch_failed",
    "accuracy_error": "api_config_accuracy_error",
    "timeout": "api_config_timeout",
    "crash": "api_config_crash",
    "oom": "api_config_oom",
}

api_config_locations = defaultdict(set)
log_counts = {}
api_configs = set()
checkpoint_file = TEST_LOG_PATH / "checkpoint.txt"

if not checkpoint_file.exists():
    exit(0)

checkpoint_configs = set()
try:
    with checkpoint_file.open("r") as f:
        checkpoint_configs = set(line.strip() for line in f if line.strip())
        log_counts["checkpoint"] = len(checkpoint_configs)
except Exception as err:
    print(f"Error reading {checkpoint_file}: {err}", flush=True)

# 读取其他日志文件
for log_type, prefix in LOG_PREFIXES.items():
    if log_type == "checkpoint":
        continue
    log_file = TEST_LOG_PATH / f"{prefix}.txt"
    if not log_file.exists():
        continue
    try:
        with log_file.open("r") as f:
            lines = set(line.strip() for line in f if line.strip())
            log_counts[log_type] = len(lines)
            for config in lines:
                api_config_locations[config].add(log_type)
    except Exception as err:
        print(f"Error reading {log_file}: {err}", flush=True)

# 找出重复的配置
duplicate_configs = {config: locations for config, locations in api_config_locations.items() 
                    if len(locations) > 1}

if duplicate_configs:
    print("\n重复的API配置及其出现位置:", flush=True)
    for config, locations in sorted(duplicate_configs.items()):
        print(f"{config}: {', '.join(sorted(locations))}", flush=True)

# 计算skip数量
# api_configs = checkpoint_configs - set(api_config_locations.keys())
# if api_configs:
#     log_counts["skip"] = len(api_configs)
#     skip_file = TEST_LOG_PATH / "api_config_skip.txt"
#     try:
#         with skip_file.open("w") as f:
#             f.writelines(f"{line}\n" for line in sorted(api_configs))
#     except Exception as err:
#         print(f"Error writing to {skip_file}: {err}", flush=True)

# 打印统计结果
print("\n统计结果:", flush=True)
for log_type, count in log_counts.items():
    print(f"{log_type}: {count}", flush=True)

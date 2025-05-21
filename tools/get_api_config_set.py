# 获取 api 集合小工具
# @author: cangtianhaung

from pathlib import Path

FILE_PATH = Path("tester/api_config/api_config_tmp.txt")
OUTPUT_PATH = Path("tester/api_config/api_config_set.txt")

api_configs = set()
try:
    with open(FILE_PATH, "r") as f:
        api_configs = set(line.strip() for line in f if line.strip())
except Exception as err:
    print(f"Error reading {FILE_PATH}: {err}", flush=True)
    exit(0)
print(f"{len(api_configs)} api config(s) read from {FILE_PATH}", flush=True)

try:
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r") as f:
            api_configs.update(line.strip() for line in f if line.strip())
    with open(OUTPUT_PATH, "w") as f:
        f.writelines(f"{line}\n" for line in sorted(api_configs))
except Exception as err:
    print(f"Error writing {OUTPUT_PATH}: {err}", flush=True)
    exit(0)

print(f"{len(api_configs)} api config(s) written to {OUTPUT_PATH}", flush=True)

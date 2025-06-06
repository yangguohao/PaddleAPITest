# 获取 api 配置集合小工具
# @author: cangtianhaung

from pathlib import Path

INPUT_PATH = Path("tester/api_config/api_config_tmp.txt")
OUTPUT_PATH = INPUT_PATH

api_configs = set()
try:
    with open(INPUT_PATH, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        print(f"{len(lines)} api config(s) read from {INPUT_PATH}", flush=True)
        api_configs = set(lines)
except Exception as err:
    print(f"Error reading {INPUT_PATH}: {err}", flush=True)
    exit(0)

try:
    if OUTPUT_PATH != INPUT_PATH and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r") as f:
            api_configs.update(line.strip() for line in f if line.strip())
    with open(OUTPUT_PATH, "w") as f:
        f.writelines(f"{line}\n" for line in sorted(api_configs))
except Exception as err:
    print(f"Error writing {OUTPUT_PATH}: {err}", flush=True)
    exit(0)
print(f"{len(api_configs)} api config(s) written to {OUTPUT_PATH}", flush=True)

# 获取 api 集合小工具
# @author: cangtianhuang
# @date: 2025-06-08

from pathlib import Path

INPUT_PATH = Path("tester/api_config/api_config_tmp.txt")
OUTPUT_PATH = INPUT_PATH

api_apis = set()
count = 0
try:
    with open(INPUT_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                api_api = line.split("(", 1)[0]
                api_apis.add(api_api)
                count += 1
except Exception as err:
    print(f"Error reading {INPUT_PATH}: {err}", flush=True)
    exit(0)
print(f"Read {count} api(s) from {INPUT_PATH}", flush=True)

try:
    if OUTPUT_PATH != INPUT_PATH and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r") as f:
            api_apis.update(line.strip() for line in f if line.strip())
    with open(OUTPUT_PATH, "w") as f:
        f.writelines(f"{line}\n" for line in sorted(api_apis))
except Exception as err:
    print(f"Error writing {OUTPUT_PATH}: {err}", flush=True)
    exit(0)
print(f"Write {len(api_apis)} api(s) to {OUTPUT_PATH}", flush=True)

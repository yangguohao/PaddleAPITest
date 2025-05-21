# 获取 api 集合小工具
# @author: cangtianhaung

from pathlib import Path

FILE_PATH = Path("tester/api_config/api_config_tmp.txt")
OUTPUT_PATH = Path("tester/api_config/api_set_tmp.txt")

api_names = set()
try:
    with open(FILE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                api_name = line.split('(', 1)[0]
                api_names.add(api_name)
except Exception as err:
    print(f"Error reading {FILE_PATH}: {err}", flush=True)
    exit(0)
print(f"{len(api_names)} api(s) read from {FILE_PATH}", flush=True)

try:
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r") as f:
            api_names.update(line.strip() for line in f if line.strip())
    with open(OUTPUT_PATH, "w") as f:
        f.writelines(f"{line}\n" for line in sorted(api_names))
except Exception as err:
    print(f"Error writing {OUTPUT_PATH}: {err}", flush=True)
    exit(0)

print(f"{len(api_names)} api(s) written to {OUTPUT_PATH}", flush=True)

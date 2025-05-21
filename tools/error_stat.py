# test_log 一键整理小工具（engineV2版）：log_digester_lite + get_api_set + get_api_config_set
# @author: cangtianhuang

from pathlib import Path
import re
from collections import defaultdict

TEST_LOG_PATH = Path("tester/api_config/test_log")
OUTPUT_PATH = Path("report/0size_tensor_gpu/20250521/paddleonly")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# log_digester_lite
pattern = re.compile(
    r"^(\[[^\]]+\]|\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+|W\d{4} \d{2}:\d{2}:\d{2}\.\d+)"
)

categorized_logs = defaultdict(list)
current_category = None
current_content = []

LOG_PATH = TEST_LOG_PATH / "log_inorder.log"
try:
    with LOG_PATH.open("r") as f:
        input_text = f.read()
except Exception as err:
    print(f"Error reading {LOG_PATH}: {err}", flush=True)
    exit(1)

for line in input_text.split("\n"):
    match = pattern.match(line)
    if match:
        if current_category:
            categorized_logs[current_category].append("\n".join(current_content))

        if match.group(1).startswith("W"):
            current_category = None
            current_content = []
        elif match.group(1).startswith("["):
            current_category = match.group(1)
            current_content = [line]
        else:
            current_category = None
            current_content = []
    elif current_category:
        current_content.append(line)

if current_category:
    categorized_logs[current_category].append("\n".join(current_content))

output_log = OUTPUT_PATH / "error_log.log"
with open(output_log, "w") as f:
    for category in sorted(categorized_logs.keys()):
        if category == "[Pass]":
            continue
        f.write(f"=== {category} ===\n\n")
        categorized_logs[category].sort()
        for content in categorized_logs[category]:
            f.write(content + "\n\n")
        f.write("\n")

# get_api_set + get_api_config_set
ERROR_LOG = [
    "api_config_accuracy_error.txt",
    "api_config_crash.txt",
    "api_config_paddle_error.txt",
    "api_config_torch_error.txt",
    "api_config_paddle_to_torch_failed.txt",
    "api_config_timeout",
]
api_names = set()
api_configs = set()
for file_name in ERROR_LOG:
    FILE_PATH = TEST_LOG_PATH / file_name
    if not FILE_PATH.exists():
        continue
    try:
        with open(FILE_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    api_name = line.split("(", 1)[0]
                    api_names.add(api_name)
                    api_configs.add(line)
    except Exception as err:
        print(f"Error reading {file_name}: {err}", flush=True)
        exit(0)
print(f"Read {len(api_names)} api(s)", flush=True)
print(f"Read {len(api_configs)} api config(s)", flush=True)

# get_api_set
API_OUTPUT_PATH = OUTPUT_PATH / "error_api.txt"
try:
    with open(API_OUTPUT_PATH, "w") as f:
        f.writelines(f"{line}\n" for line in sorted(api_names))
except Exception as err:
    print(f"Error writing {API_OUTPUT_PATH}: {err}", flush=True)
    exit(0)
print(f"Write {len(api_names)} api(s)", flush=True)

# get_api_config_set
CONFIG_OUTPUT_PATH = OUTPUT_PATH / "error_config.txt"
try:
    with open(CONFIG_OUTPUT_PATH, "w") as f:
        f.writelines(f"{line}\n" for line in sorted(api_configs))
except Exception as err:
    print(f"Error writing {CONFIG_OUTPUT_PATH}: {err}", flush=True)
    exit(0)

print(f"Write {len(api_configs)} api config(s)", flush=True)

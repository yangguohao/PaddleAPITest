# test_log 一键整理小工具（engineV2有序版）：log_digester_lite + get_api_set + get_api_config_set
# @author: cangtianhuang

from pathlib import Path
import re

TEST_LOG_PATH = Path("tester/api_config/test_log")
OUTPUT_PATH = Path("report/0size_tensor_gpu/20250521/accuracy")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# log_digester_lite
pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+")
warning_pattern = re.compile(r"^W0521 \d{2}:\d{2}:\d{2}\.\d{6} \d+ gpu_resources.cc")
test_config_pattern = re.compile(r"test begin: (.*)$")

logs = []
in_test_block = False
current_content = []

LOG_PATH = TEST_LOG_PATH / "log_inorder.log"
try:
    with LOG_PATH.open("r") as f:
        input_text = f.read()
except Exception as err:
    print(f"Error reading {LOG_PATH}: {err}", flush=True)
    exit(1)

for line in input_text.split("\n"):
    if warning_pattern.match(line):
        continue

    if pattern.match(line):
        if in_test_block:
            if current_content:
                logs.append("\n".join(current_content))
            current_content = []
            in_test_block = False
        if "test begin:" in line:
            in_test_block = True
            current_content = [line]
        continue
    
    if in_test_block:
        current_content.append(line)

if current_content:
    logs.append("\n".join(current_content))
    current_content = []


def get_sort_key(content):
    for line in content.split("\n"):
        match = test_config_pattern.search(line)
        if match:
            return match.group(1)
    return ""


sorted_logs = []
for content in logs:
    sorted_logs.append((get_sort_key(content), content))
sorted_logs.sort(key=lambda x: x[0])

output_log = OUTPUT_PATH / "error_log.log"
try:
    with open(output_log, "w") as f:
        for _, content in sorted_logs:
            f.write(content + "\n\n")
except Exception as err:
    print(f"Error writing {output_log}: {err}", flush=True)
    exit(0)
print(f"Read and write {len(logs)} log(s)", flush=True)

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

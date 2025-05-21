# 整理 log.log 小工具（engineV2版）
# @author: cangtianhaung

from pathlib import Path
import re
from collections import defaultdict

pattern = re.compile(r'^(\[[^\]]+\]|\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+|W\d{4} \d{2}:\d{2}:\d{2}\.\d+)')

categorized_logs = defaultdict(list)
current_category = None
current_content = []

test_log_path = Path("tester/api_config/test_log")
input_log = test_log_path / "log.log"
try:
    with input_log.open("r") as f:
        input_text = f.read()
except Exception as err:
    print(f"Error reading {input_log}: {err}", flush=True)
    exit(1)

for line in input_text.split('\n'):
    match = pattern.match(line)
    if match:
        if current_category:
            categorized_logs[current_category].append('\n'.join(current_content))
        
        if match.group(1).startswith('W'):
            current_category = None
            current_content = []
        elif match.group(1).startswith('['):
            current_category = match.group(1)
            current_content = [line]
        else:
            current_category = None
            current_content = []
    elif current_category:
        current_content.append(line)

if current_category:
    categorized_logs[current_category].append('\n'.join(current_content))

output_log = test_log_path / "log_categorized.log"
with open(output_log, 'w') as f:
    for category in sorted(categorized_logs.keys()):
        f.write(f"=== {category} ===\n")
        categorized_logs[category].sort()
        for content in categorized_logs[category]:
            f.write(content + '\n')
        f.write('\n')

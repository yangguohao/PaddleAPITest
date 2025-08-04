import os
import re
from pathlib import Path

import yaml

INPUT_DIR = Path("tools/api_tracer/trace_output_tmp")


def parse_api(api):
    if ".Tensor." in api:
        return api
    parts = api.rsplit(".", 1)
    if len(parts) == 2 and re.match(r"_?[a-zA-Z][a-zA-Z0-9_]*", parts[1]):
        return parts[0]
    return api


def process_file(input_path, target_apis):
    input_name = input_path.name
    output_name = input_name[4:]
    output_path = input_path.parent / output_name

    output_excluded_name = output_name.replace(".txt", "_excluded.txt")
    output_excluded_path = input_path.parent / output_excluded_name

    apis = set()
    with input_path.open("r") as f:
        apis = set([line.strip() for line in f if line.strip()])
    print(f"Read {len(apis)} apis from {input_path}", flush=True)

    alias_apis = set()
    excluded_apis = set()
    for api in apis:
        alias_api = api
        if alias_api not in target_apis:
            excluded_apis.add(api)
            continue
        alias_apis.add(parse_api(api))

    with output_path.open("w") as f:
        f.writelines(f"{line}\n" for line in sorted(alias_apis))
    print(f"Write {len(alias_apis)} alias apis to {output_path}", flush=True)

    with output_excluded_path.open("w") as f:
        f.writelines(f"{line}\n" for line in sorted(excluded_apis))
    print(
        f"Write {len(excluded_apis)} excluded apis to {output_excluded_path}",
        flush=True,
    )


def main():
    yaml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "api_list",
        "torch_api_list.yaml",
    )
    target_apis = []
    with open(yaml_path, "r", encoding="utf-8") as f:
        target_apis = yaml.safe_load(f)
    print(f"Loaded {len(target_apis)} target APIs.")

    input_files = list(INPUT_DIR.glob("api_apis.txt"))
    if not input_files:
        print(f"No input files found in {INPUT_DIR}", flush=True)
        return

    for input_file in sorted(input_files):
        process_file(input_file, target_apis)


if __name__ == "__main__":
    main()

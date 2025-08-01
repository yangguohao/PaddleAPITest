import re
from pathlib import Path

INPUT_DIR = Path("tools/api_tracer/trace_output_tmp")


def parse_api(api):
    if ".Tensor." in api:
        return api
    if re.search(r"\.[A-Z][a-zA-Z0-9]*\.", api):
        return api.rsplit(".", 1)[0]
    return api


def process_file(input_path):
    input_name = input_path.name
    output_name = input_name[4:]
    output_path = input_path.parent / output_name

    apis = set()
    with input_path.open("r") as f:
        apis = set([line.strip() for line in f if line.strip()])
    print(f"Read {len(apis)} apis from {input_path}", flush=True)

    alias_apis = set()
    for api in apis:
        alias_apis.add(parse_api(api))

    with output_path.open("w") as f:
        f.writelines(f"{line}\n" for line in sorted(alias_apis))
    print(f"Write {len(alias_apis)} alias apis to {output_path}", flush=True)


def main():
    input_files = list(INPUT_DIR.glob("api_apis*.txt"))
    if not input_files:
        print(f"No input files found in {INPUT_DIR}", flush=True)
        return

    for input_file in sorted(input_files):
        process_file(input_file)


if __name__ == "__main__":
    main()

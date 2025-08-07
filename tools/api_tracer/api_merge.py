from pathlib import Path

apis = {}
folder_path = Path("tools/api_tracer/trace_output")
for dir_path in folder_path.glob("*/"):
    dir_name = dir_path.name
    files = dir_path.glob("*")
    for file in files:
        model_name = f"{dir_name}/{file.name}"
        print("Reading ", model_name)
        apis[model_name] = set(file.read_text().splitlines())

all = set()
for api_set in apis.values():
    all.update(api_set)

result = "API\t"
for name in apis.keys():
    result += name + "\t"
result += "\n"
for api in all:
    result += api + "\t"
    for name in apis.keys():
        if api in apis[name]:
            result += "是\t"
        else:
            result += "否\t"
    result += "\n"

with open("tools/api_tracer/trace_output/apis.txt", "w") as f:
    f.writelines(result)

import json
import collections

with open("api_mapping.json", "r") as f:
    api_mapping = json.load(f)


regular_dict = collections.OrderedDict()
manual_matcher_dict = collections.OrderedDict()
paddle_multi_arg_to_torch_1_arg_dict = collections.OrderedDict()
paddle_args_grate_than_torch_dict = collections.OrderedDict()
default_value_diff_dict = collections.OrderedDict()

for api in api_mapping:
    if len(api_mapping[api]) == 0:
        continue
    if "Matcher" not in api_mapping[api]:
        continue
    if "paddle_api" not in api_mapping[api]:
        continue
    if "args_list" not in api_mapping[api]:
        continue
    
    paddle_api = api_mapping[api]["paddle_api"]
    torch_api = api
    matcher = api_mapping[api]["Matcher"]
    is_generic_matcher = matcher == "GenericMatcher"
    min_input_args = api_mapping[api]["min_input_args"] if "min_input_args" in api_mapping[api] else 0
    args_list = api_mapping[api]["args_list"]
    kwargs_change = api_mapping[api]["kwargs_change"] if "kwargs_change" in api_mapping[api] else {}
    unsupport_args = api_mapping[api]["unsupport_args"] if "unsupport_args" in api_mapping[api] else []
    paddle_default_kwargs = api_mapping[api]["paddle_default_kwargs"] if "paddle_default_kwargs" in api_mapping[api] else []
    is_can_convert_directly = True if len(paddle_default_kwargs)==0 else False
    is_paddle_multi_arg_to_torch_1_arg = False
    
    paddle_torch_args_map = collections.OrderedDict()

    for arg in args_list:
        if arg in unsupport_args:
            continue
        if arg in kwargs_change:
            if isinstance(kwargs_change[arg], list):
                for sub_arg in kwargs_change[arg]:
                    paddle_torch_args_map[sub_arg] = arg
                is_paddle_multi_arg_to_torch_1_arg = True
            else:
                if kwargs_change[arg] == "":
                    continue
                paddle_torch_args_map[kwargs_change[arg]] = arg
        else:
            paddle_torch_args_map[arg] = arg
    count = 0
    for default_arg in paddle_default_kwargs:
        if default_arg in paddle_torch_args_map:
            count += 1
        else:
            paddle_torch_args_map[default_arg] = ""
    if count == 0:
        is_paddle_args_grate_than_torch = True
    else:
        is_paddle_args_grate_than_torch = False
    
    item = collections.OrderedDict()
    item["torch_api"] = torch_api
    item["paddle_torch_args_map"] = paddle_torch_args_map
    item["min_input_args"] = min_input_args
    
    if is_generic_matcher:
        if is_can_convert_directly:
            if is_paddle_multi_arg_to_torch_1_arg:
                paddle_multi_arg_to_torch_1_arg_dict[paddle_api] = item
            else:
                regular_dict[paddle_api] = item
        else:
            if is_paddle_args_grate_than_torch:
                paddle_args_grate_than_torch_dict[paddle_api] = item
            else:
                item["paddle_default_kwargs"] = paddle_default_kwargs
                default_value_diff_dict[paddle_api] = item
    else:
        item["matcher"] = matcher
        manual_matcher_dict[paddle_api] = item
    
    
with open("paddle2torch_regular_dict.json", "w") as f:
    json.dump(regular_dict, f, indent=6)
with open("paddle2torch_manual_matcher_dict.json", "w") as f:
    json.dump(manual_matcher_dict, f, indent=6)
with open("paddle2torch_paddle_multi_arg_to_torch_1_arg_dict.json", "w") as f:
    json.dump(paddle_multi_arg_to_torch_1_arg_dict, f, indent=6)
with open("paddle2torch_paddle_args_grate_than_torch_dict.json", "w") as f:
    json.dump(paddle_args_grate_than_torch_dict, f, indent=6)
with open("paddle2torch_default_value_diff_dict.json", "w") as f:
    json.dump(default_value_diff_dict, f, indent=6)

with open("regular_api.yaml", "w") as f:
    f.write('apis:\n')
    for api in regular_dict:
        f.write("  -  "+api+'\n')

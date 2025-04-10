import os
import json
import collections

paddle_to_torch_wrong_config = [
    "paddle.matmul",
    "paddle.nn.functional.adaptive_avg_pool2d",
    "paddle.nn.functional.adaptive_avg_pool3d",
    "paddle.nn.functional.channel_shuffle",
    "paddle.nn.functional.conv1d",
    "paddle.nn.functional.conv1d_transpose",
    "paddle.nn.functional.conv2d",
    "paddle.nn.functional.conv2d_transpose",
    "paddle.nn.functional.conv3d",
    "paddle.nn.functional.conv3d_transpose",
    "paddle.nn.functional.gaussian_nll_loss",
    "paddle.nn.functional.group_norm",
    "paddle.nn.functional.interpolate",
    "paddle.nn.functional.local_response_norm",
    "paddle.nn.functional.lp_pool1d",
    "paddle.nn.functional.lp_pool2d",
    "paddle.nn.functional.max_pool1d",
    "paddle.nn.functional.max_pool2d",
    "paddle.nn.functional.max_pool3d",
    "paddle.nn.functional.max_unpool1d",
    "paddle.nn.functional.max_unpool2d",
    "paddle.nn.functional.max_unpool3d",
    "paddle.nn.functional.pixel_shuffle",
    "paddle.nn.functional.pixel_unshuffle",
    "paddle.nn.functional.prelu",
    "paddle.nn.functional.selu",
    "paddle.as_strided",
]

class Paddle2torchConverter:
    def __init__(self):
        self.rules = {}
        self.load_paddle2torch_dict()
        self.load_rules()

    def load_paddle2torch_dict(self):
        with open(os.path.abspath(os.path.dirname(__file__)) + '/paddle2torch_regular_dict.json', "r") as f:
            self.paddle2torch_map = json.load(f, object_pairs_hook=collections.OrderedDict)

    def load_rules(self):
        with open(os.path.abspath(os.path.dirname(__file__)) + '/rules.json', "r") as f:
            self.rules = json.load(f)


def paddle_to_torch(paddle_api):
	if paddle_api in paddle_to_torch_wrong_config:
		return None, None
	if paddle_api not in paddle_to_torch_map:
		return None, None
	return paddle_to_torch_map[paddle_api]["torch_api"], paddle_to_torch_map[paddle_api]["paddle_torch_args_map"]

def transform(paddle_api, paddle_args, paddle_kwargs):
    torch_api_str, paddle_to_torch_args_map = paddle_to_torch(api_name)
    if paddle_to_torch_args_map is None:
        print("[paddle_to_torch2]", self.api_config.config, "\napi need manual fix")
        api_config_paddle_to_torch_faild.write(self.api_config.config+"\n")
        api_config_paddle_to_torch_faild.flush()
        return None, None, None
    torch_api = eval(torch_api_str)

    api_sig = inspect.signature(paddle_api)
    api_args_list = list(api_sig.parameters.keys())
    paddle_kwargs = collections.OrderedDict()

    for i, arg_config in enumerate(paddle_args):
        paddle_kwargs[api_args_list[i]] = arg_config

    for key, arg_config in self.api_config.kwargs.items():
        self.paddle_merged_kwargs_config[key] = arg_config

    self.torch_args_config = []
    self.torch_kwargs_config = collections.OrderedDict()

    if self.api_config.api_name in ["paddle.Tensor.__getitem__", "paddle.Tensor.__setitem__"]:
        self.torch_args_config = self.paddle_args_config
    else:
        first = True
        for key, value in self.paddle_merged_kwargs_config.items():
            if first and "paddle.Tensor." in self.api_config.api_name:
                self.torch_args_config.append(value)
                first = False
                continue
            first = False
            if key == "name":
                continue
            if key not in paddle_to_torch_args_map:
                print("[paddle_to_torch]", self.api_config.config, "\n ", key, "not in paddle_to_torch_args_map, can not call torch")
                api_config_paddle_to_torch_faild.write(self.api_config.config+"\n")
                api_config_paddle_to_torch_faild.flush()
                return False

            self.torch_kwargs_config[paddle_to_torch_args_map[key]] = value

    return True
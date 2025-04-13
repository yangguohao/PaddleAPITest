import os
import json
from collections import OrderedDict
from typing import Dict, Type
from .base_rule import ConvertResult, BaseRule
from .generic_rule import GenericRule, ErrorRule
from . import custom_rules

paddle2torch_wrong_config = [
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

class Paddle2TorchConverter:
    def __init__(self):
        self.rules = {}
        self.load_dict_and_rules()

    def load_dict_and_rules(self):
        rule_cls_map: Dict[str, Type[BaseRule]] = {
            "GenericRule": GenericRule,
            "ErrorRule": ErrorRule,
        }
        for rule_name in custom_rules.__all__:
            rule_cls_map[rule_name] = getattr(custom_rules, rule_name)

        with open(os.path.abspath(os.path.dirname(__file__)) + '/paddle2torch_custom_dict.json', "r") as f:
            self.paddle2torch_custom_map = json.load(f, object_pairs_hook=OrderedDict)
        for key, value in self.paddle2torch_custom_map.items():
            if key.startswith("paddle.") and key not in self.rules:
                rule_name = value.get("Rule")
                if rule_name and rule_name in rule_cls_map:
                    self.rules[key] = rule_cls_map[rule_name]()
                else:
                    error_msg = (
                        f"{key} doesn't have 'Rule' field" if not rule_name 
                        else f"{rule_name} for {key} is not implemented"
                    )
                    self.rules[key] = ErrorRule(error_msg)

        with open(os.path.abspath(os.path.dirname(__file__)) + '/paddle2torch_generic_dict.json', "r") as f:
            self.paddle2torch_generic_map = json.load(f, object_pairs_hook=OrderedDict)
        for key, value in self.paddle2torch_generic_map.items():
            if key.startswith("paddle.") and key not in self.rules:
                self.rules[key] = GenericRule(value)

        with open(os.path.abspath(os.path.dirname(__file__)) + '/paddle2torch_regular_dict.json', "r") as f:
            self.paddle2torch_regular_map = json.load(f, object_pairs_hook=OrderedDict)
        for key, value in self.paddle2torch_generic_map.items():
            if key.startswith("paddle.") and key not in self.rules:
                self.rules[key] = GenericRule(value)

    def convert(self, paddle_api, paddle_args, paddle_kwargs) -> ConvertResult:
        if paddle_api in paddle2torch_wrong_config or paddle_api not in self.rules:
            return ConvertResult().error(f"{paddle_api} is not supported")
        rule = self.rules[paddle_api]
        return rule.apply(paddle_api, paddle_args, paddle_kwargs)

import inspect
import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Type
from unittest import result

import paddle
import torch

from . import custom_rules
from .base_rule import BaseRule, ConvertResult
from .generic_rule import ErrorRule, GenericRule

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
        self.mapping = {}
        self.load_dict_and_rules()

    def load_dict_and_rules(self):
        rule_cls_map: Dict[str, Type[BaseRule]] = {
            "GenericRule": GenericRule,
            "ErrorRule": ErrorRule,
        }
        for rule_name in custom_rules.__all__:
            rule_cls_map[rule_name] = getattr(custom_rules, rule_name)

        with open(os.path.abspath(os.path.dirname(__file__)) + '/paddle2torch_custom_dict.json', "r") as f:
            paddle2torch_custom_map = json.load(f, object_pairs_hook=OrderedDict)
            for key, value in paddle2torch_custom_map.items():
                if key.startswith("paddle.") and key not in self.rules:
                    self.mapping[key] = value
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
            paddle2torch_generic_map = json.load(f, object_pairs_hook=OrderedDict)
            for key, value in paddle2torch_generic_map.items():
                if key.startswith("paddle.") and key not in self.rules:
                    self.mapping[key] = value
                    self.rules[key] = GenericRule()

        with open(os.path.abspath(os.path.dirname(__file__)) + '/paddle2torch_regular_dict.json', "r") as f:
            paddle2torch_regular_map = json.load(f, object_pairs_hook=OrderedDict)
            for key, value in paddle2torch_regular_map.items():
                if key.startswith("paddle.") and key not in self.rules:
                    self.mapping[key] = value
                    self.rules[key] = GenericRule()

    def convert(self, paddle_api: str) -> ConvertResult:
        if paddle_api in paddle2torch_wrong_config or paddle_api not in self.rules:
            return ConvertResult.error(paddle_api, f"{paddle_api} is not supported")
        rule = self.rules[paddle_api]
        rule.preprocess(self.mapping[paddle_api])
        if paddle_api not in rule.cached_results:
            rule.cached_results[paddle_api] = rule.apply(paddle_api)
        return rule.cached_results[paddle_api]
    
    def execute(self, convert_result: ConvertResult, paddle_api: str, torch_args: List, torch_kwargs: Dict) -> Any:
        if not convert_result.is_supported or not convert_result.code:
            return None
        exec_globals = {
            "__builtins__": __builtins__,
            "torch": torch
        }
        exec_locals: Dict[str, Any] = {"result": None}

        # 将 torch tensors 映射至 paddle 同名参数
        paddle_api_sig = inspect.signature(eval(paddle_api))
        paddle_params = list(paddle_api_sig.parameters.keys())

        if len(torch_args) > len(paddle_params):
            raise ValueError(f"Too many arguments for {paddle_api}")
        
        exec_locals.update(zip(paddle_params, torch_args))
        exec_locals.update(torch_kwargs)

        exec('\n'.join(convert_result.code), exec_globals, exec_locals)
        if convert_result.output_var:
            result_var = convert_result.output_var
            if result_var not in exec_locals:
                raise ValueError(f"Variable {result_var} not found in the execution context")
            return exec_locals.get(result_var)
        return exec_locals.get("result")

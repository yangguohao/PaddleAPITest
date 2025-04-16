import inspect
import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Type

import paddle
import torch

from . import rules
from .rules import BaseRule, ConvertResult, ErrorRule, GenericRule

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
        for rule_name in rules.__all__:
            rule_cls_map[rule_name] = getattr(rules, rule_name)

        with open(
            os.path.abspath(os.path.dirname(__file__))
            + "/paddle2torch_custom_dict.json",
            "r",
        ) as f:
            paddle2torch_custom_map = json.load(f, object_pairs_hook=OrderedDict)
            for key, value in paddle2torch_custom_map.items():
                if key.startswith("paddle.") and key not in self.rules:
                    self.mapping[key] = value
                    rule_name = value.get("Rule")
                    if rule_name and rule_name in rule_cls_map:
                        self.rules[key] = rule_cls_map[rule_name]()
                    else:
                        error_msg = (
                            f"{key} doesn't have 'Rule' field"
                            if not rule_name
                            else f"{rule_name} for {key} is not implemented"
                        )
                        self.rules[key] = ErrorRule(error_msg)

        with open(
            os.path.abspath(os.path.dirname(__file__))
            + "/paddle2torch_generic_dict.json",
            "r",
        ) as f:
            paddle2torch_generic_map = json.load(f, object_pairs_hook=OrderedDict)
            for key, value in paddle2torch_generic_map.items():
                if key.startswith("paddle.") and key not in self.rules:
                    self.mapping[key] = value
                    self.rules[key] = GenericRule()

        with open(
            os.path.abspath(os.path.dirname(__file__))
            + "/paddle2torch_regular_dict.json",
            "r",
        ) as f:
            paddle2torch_regular_map = json.load(f, object_pairs_hook=OrderedDict)
            for key, value in paddle2torch_regular_map.items():
                if key.startswith("paddle.") and key not in self.rules:
                    self.mapping[key] = value
                    self.rules[key] = GenericRule()

    def convert(self, paddle_api: str) -> ConvertResult:
        """
        将 Paddle API 转换为 Torch API

        Args:
            paddle_api (str): 需要转换的 Paddle API 名称

        Returns:
            ConvertResult: 转换结果，包括转换后的 Torch API 代码、输出变量或错误信息

        """
        if paddle_api in paddle2torch_wrong_config:
            return ConvertResult.error(
                paddle_api, f"{paddle_api} is in the wrong config"
            )
        if paddle_api not in self.rules:
            return ConvertResult.error(
                paddle_api, f"Rule for {paddle_api} is not implemented"
            )
        rule = self.rules[paddle_api]
        rule.read_mapping(self.mapping[paddle_api])
        if paddle_api not in rule.cached_results:
            rule.cached_results[paddle_api] = rule.apply(paddle_api)
        return rule.cached_results[paddle_api]

    @staticmethod
    def execute(
        convert_result: ConvertResult,
        paddle_api: str,
        torch_args: List,
        torch_kwargs: OrderedDict,
    ) -> Any:
        """
        执行转换后的代码。

        Args:
            convert_result (ConvertResult): 转换结果对象
            paddle_api (str): Paddle API 名称
            torch_args (List): 传递给 Paddle API 的含有 Torch Tensors 的位置参数列表
            torch_kwargs (OrderedDict): 传递给 Paddle API 的含有 Torch Tensors 的关键字参数字典

        Returns:
            Any: 执行结果

        Raises:
            RuntimeError: 执行转换后的代码时发生异常
            ValueError: 转换结果中指定的输出变量在执行上下文中不存在
        """
        if not convert_result.is_supported or not convert_result.code:
            return None
        exec_globals = {"__builtins__": __builtins__, "torch": torch}
        exec_locals: Dict[str, Any] = {
            "result": None,
            "args": torch_args,
            "kwargs": torch_kwargs,
        }

        is_tensor_method = paddle_api.startswith("paddle.Tensor.")

        # 将 args 中的 torch tensors 映射至 paddle 同名参数
        paddle_api_sig = inspect.signature(eval(paddle_api))
        paddle_params = list(paddle_api_sig.parameters.keys())
        if is_tensor_method:
            exec_locals["_tmp_tensor"] = torch_args[0]
            exec_locals.update(zip(paddle_params, torch_args[1:]))
        else:
            exec_locals.update(zip(paddle_params, torch_args))

        exec_locals.update(torch_kwargs)

        try:
            exec("\n".join(convert_result.code), exec_globals, exec_locals)
        except Exception as e:
            raise RuntimeError(
                f"Error during execution of converted code: {str(e)}"
            ) from e
        if convert_result.output_var:
            result_var = convert_result.output_var
            if result_var not in exec_locals:
                raise ValueError(
                    f"Variable {result_var} not found in the execution context"
                )
            return exec_locals.get(result_var)
        return exec_locals.get("result")

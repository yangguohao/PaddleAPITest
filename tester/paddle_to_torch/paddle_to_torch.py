import os
import json
import collections
from typing import Type
from .base_rule import ConvertResult, BaseRule

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

class Paddle2torchConverter:
    def __init__(self):
        self.rules = {}
        self.load_paddle2torch_dict()
        self.load_rules()

    def load_paddle2torch_dict(self):
        with open(os.path.abspath(os.path.dirname(__file__)) + '/paddle2torch_regular_dict.json', "r") as f:
            self.paddle2torch_map = json.load(f, object_pairs_hook=collections.OrderedDict)

    def load_rules(self):
        from . import custom_rules
        
        for name in custom_rules.__all__:
            rule_class: Type[BaseRule] = getattr(custom_rules, name)
            if rule_class.paddle_api:
                self.rules[rule_class.paddle_api] = rule_class()
        with open(os.path.abspath(os.path.dirname(__file__)) + '/rules.json', "r") as f:
            self.rules = json.load(f)

    def convert(self, paddle_api, paddle_args, paddle_kwargs) -> ConvertResult:
        if paddle_api in paddle2torch_wrong_config or paddle_api not in self.paddle2torch_map:
            return ConvertResult().error('Not supported')
        rule = self.rules[paddle_api]
        return rule.apply(paddle_args, paddle_kwargs)

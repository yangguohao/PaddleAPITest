from .api_config import TensorConfig, APIConfig, analyse_configs

import re
import collections
import paddle
import numpy
import math
import json
import paddle
import inspect
from .base import APITestBase
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))[0:os.path.dirname(os.path.realpath(__file__)).index("PaddleAPITest")+13]

api_config_accuracy_error = open(DIR_PATH+"/tester/api_config/test_log/api_config_accuracy_error.txt", "a")
api_config_paddle_error = open(DIR_PATH+"/tester/api_config/test_log/api_config_paddle_error.txt", "a")
api_config_pass = open(DIR_PATH+"/tester/api_config/test_log/api_config_pass.txt", "a")

class APITestPaddleOnly(APITestBase):
    def __init__(self, api_config):
        self.api_config = api_config
    def test(self):
        if self.need_skip():
            return

        if not self.ana_paddle_api_info():
            return

        try:
            if not self.gen_paddle_input():
                return
            if (self.api_config.api_name[-1] == "_" and self.api_config.api_name[-2:] != "__") or self.api_config.api_name == "paddle.Tensor.__setitem__":
                args, kwargs = self.copy_paddle_input()
            else:
                args = self.paddle_args
                kwargs = self.paddle_kwargs
            paddle_output = self.paddle_api(*tuple(args), **kwargs)
            del args
            del kwargs
            if not self.is_forward_only() and not (self.api_config.api_name == "paddle.assign" and isinstance(self.paddle_args[0], list)) and not (self.api_config.api_name == "paddle.assign" and len(self.paddle_args) > 1 and self.paddle_args[1] is not None):
                inputs_list = self.get_paddle_input_list()
                result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(paddle_output)
                if len(inputs_list) != 0 and len(result_outputs) != 0 and len(result_outputs_grads) != 0:
                    out_grads = paddle.grad(result_outputs, inputs_list, grad_outputs=result_outputs_grads,allow_unused=True)
            self.clear_paddle_tensor()
        except Exception as err:
            paddle_output = None
            result_outputs = None
            result_outputs_grads = None
            out_grads = None
            if "gradient_accumulator.cc" in str(err) or "Out of memory" in str(err):
                return
            print("[paddle error]", self.api_config.config, "\n", str(err))
            api_config_paddle_error.write(self.api_config.config+"\n")
            api_config_paddle_error.flush()
            return

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            print("[cuda error]", self.api_config.config, "\n", str(err))
            paddle_output = None
            result_outputs = None
            result_outputs_grads = None
            out_grads = None
            api_config_paddle_error.write(self.api_config.config+"\n")
            api_config_paddle_error.flush()
            return

        paddle_output = None
        result_outputs = None
        result_outputs_grads = None
        out_grads = None
        print("[Pass]", self.api_config.config)
        api_config_pass.write(self.api_config.config+"\n")
        api_config_pass.flush()
  

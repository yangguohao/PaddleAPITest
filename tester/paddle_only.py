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
api_config_accuracy_error = open("/data/OtherRepo/PaddleAPITest/tester/api_config/test_log/api_config_accuracy_error.txt", "a")
api_config_paddle_error = open("/data/OtherRepo/PaddleAPITest/tester/api_config/test_log/api_config_paddle_error.txt", "a")
api_config_pass = open("/data/OtherRepo/PaddleAPITest/tester/api_config/test_log/api_config_pass.txt", "a")

class APITestPaddleOnly(APITestBase):
    def __init__(self, api_config):
        self.api_config = api_config
    def test(self):
        if not self.ana_paddle_api_info():
            return

        try:
            if not self.gen_paddle_input():
                return
            if self.api_config.api_name[-1] == "_" and self.api_config.api_name[-2:] != "__":
                args, kwargs = self.copy_paddle_input()
            else:
                args = self.paddle_args
                kwargs = self.paddle_kwargs
            paddle_output = self.paddle_api(*tuple(args), **kwargs)
            del args
            del kwargs
            if not self.is_forward_only():
                inputs_list = self.get_paddle_input_list()
                result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(paddle_output)
                if len(inputs_list) != 0 and len(result_outputs) != 0 and len(result_outputs_grads) != 0:
                    out_grads = paddle.grad(result_outputs, inputs_list, grad_outputs=result_outputs_grads)
            self.clear_paddle_tensor()
        except Exception as err:
            print("[paddle error]", self.api_config.config, "\n", str(err))
            paddle_output = None
            result_outputs = None
            result_outputs_grads = None
            out_grads = None
            api_config_paddle_error.write(self.api_config.config+"\n")
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
            return

        paddle_output = None
        result_outputs = None
        result_outputs_grads = None
        out_grads = None
        print("[Pass]", self.api_config.config)
        api_config_pass.write(self.api_config.config+"\n")
  

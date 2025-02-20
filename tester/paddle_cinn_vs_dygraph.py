from .api_config import TensorConfig, APIConfig, analyse_configs

import re
import collections
import paddle
import numpy
import math
import json
import paddle
from paddle.jit import to_static
import inspect
from .base import APITestBase
import time
from func_timeout import func_set_timeout

api_config_accuracy_error = open("/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/api_config_accuracy_error.txt", "a")
api_config_paddle_error = open("/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/api_config_paddle_error.txt", "a")
api_config_pass = open("/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/api_config_pass.txt", "a")

class APITestCINNVSDygraph(APITestBase):
    def __init__(self, api_config):
        self.api_config = api_config
    @func_set_timeout(600)
    def test(self):
        if self.need_skip():
            print("[Skip]")
            return

        if not self.ana_paddle_api_info():
            print("ana_paddle_api_info failed")
            return

        try:
            def func_backward(result_outputs, inputs_list, result_outputs_grads):
                return paddle.grad(result_outputs, inputs_list, grad_outputs=result_outputs_grads,allow_unused=True)

            build_strategy = paddle.static.BuildStrategy()
            build_strategy.build_cinn_pass = True
            @to_static(full_graph=True, build_strategy=build_strategy)
            def func_backward_static(result_outputs, inputs_list, result_outputs_grads):
                return func_backward(result_outputs, inputs_list, result_outputs_grads)

            def func(args, kwargs):
                return self.paddle_api(*tuple(args), **kwargs)

            @to_static(full_graph=True, build_strategy=build_strategy)
            def func_static(args, kwargs):
                return func(args, kwargs)

            if not self.gen_paddle_input():
                print("gen_paddle_input failed")
                return

            paddle_output = func(self.paddle_args, self.paddle_kwargs)
            paddle_output_static = func_static(self.paddle_args, self.paddle_kwargs)

            # if not self.is_forward_only() and not (self.api_config.api_name == "paddle.assign" and isinstance(self.paddle_args[0], list)) and not (self.api_config.api_name == "paddle.assign" and len(self.paddle_args) > 1 and self.paddle_args[1] is not None):
            #     inputs_list = self.get_paddle_input_list()
            #     result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(paddle_output)
            #     if len(inputs_list) != 0 and len(result_outputs) != 0 and len(result_outputs_grads) != 0:
            #         out_grads = func_backward(result_outputs, inputs_list, result_outputs_grads)
            #         out_grads_static = func_backward_static(result_outputs, inputs_list, result_outputs_grads)
            #         print("out_grads_static = ", out_grads_static)
        except Exception as err:
            if "gradient_accumulator.cc" in str(err) or "Out of memory" in str(err):
                return
            print("[paddle error]", self.api_config.config, "\n", str(err))
            api_config_paddle_error.write(self.api_config.config+"\n")
            api_config_paddle_error.flush()
            if "CUDA error" in str(err):
                raise Exception(err)
            return

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            print("[cuda error]", self.api_config.config, "\n", str(err))
            api_config_paddle_error.write(self.api_config.config+"\n")
            api_config_paddle_error.flush()
            return

        if self.api_config.api_name == "paddle.broadcast_shape":
            return

        if isinstance(paddle_output, paddle.Tensor):
            try:
                if paddle_output.dtype == paddle.bfloat16:
                    paddle_output = paddle.cast(paddle_output, dtype="float32")
                    paddle_output_static = paddle.cast(paddle_output_static, dtype="float32")
                self.np_assert_accuracy(paddle_output.numpy(), paddle_output_static.cpu().numpy(), 1e-2, 1e-2, self.api_config)
            except Exception as err:
                print("[accuracy error]", self.api_config.config, "\n", str(err))
                paddle_output_static = None
                paddle_output = None
                api_config_accuracy_error.write(self.api_config.config+"\n")
                api_config_accuracy_error.flush()
                return
        elif isinstance(paddle_output, (list, tuple)):
            if isinstance(paddle_output, tuple):
                paddle_output = list(paddle_output)
            if not isinstance(paddle_output_static, (list, tuple)):
                print("[output type diff error]", self.api_config.config)
                paddle_output_static = None
                paddle_output = None
                return
            if isinstance(paddle_output_static, tuple):
                paddle_output_static = list(paddle_output_static)
            if len(paddle_output) != len(paddle_output_static):
                print("[output type diff error]", self.api_config.config)
                paddle_output_static = None
                paddle_output = None
                return
            for i in range(len(paddle_output)):
                if not isinstance(paddle_output[i], paddle.Tensor):
                    print("not compare ", paddle_output[i], paddle_output_static[i])
                else:
                    try:
                        if paddle_output[i].dtype == paddle.bfloat16:
                            paddle_output[i] = paddle.cast(paddle_output[i], dtype="float32")
                            paddle_output_static[i] = paddle.cast(paddle_output_static[i], dtype="float32")
                        self.np_assert_accuracy(paddle_output[i].numpy(), paddle_output_static[i].cpu().numpy(), 1e-2, 1e-2, self.api_config)
                    except Exception as err:
                        print("[accuracy error]", self.api_config.config, "\n", str(err))
                        paddle_output_static = None
                        paddle_output = None
                        api_config_accuracy_error.write(self.api_config.config+"\n")
                        api_config_accuracy_error.flush()
                        return

        print("[Pass]", self.api_config.config)
        api_config_pass.write(self.api_config.config+"\n")
        api_config_pass.flush()
  

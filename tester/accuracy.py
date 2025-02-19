from .api_config import TensorConfig, APIConfig, analyse_configs

import re
import collections
import paddle
import numpy
import math
import json
import torch
import paddle
import inspect
from .base import APITestBase
import os
import time
from func_timeout import func_set_timeout

DIR_PATH = os.path.dirname(os.path.realpath(__file__))[0:os.path.dirname(os.path.realpath(__file__)).index("PaddleAPITest")+13]

api_config_accuracy_error = open(DIR_PATH+"/tester/api_config/test_log/api_config_accuracy_error.txt", "a")
api_config_paddle_error = open(DIR_PATH+"/tester/api_config/test_log/api_config_paddle_error.txt", "a")
api_config_pass = open(DIR_PATH+"/tester/api_config/test_log/api_config_pass.txt", "a")
api_config_torch_error = open(DIR_PATH+"/tester/api_config/test_log/api_config_torch_error.txt", "a")

class APITestAccuracy(APITestBase):
    def __init__(self, api_config):
        self.api_config = api_config
    
    @func_set_timeout(600)
    def test(self):
        if self.need_skip():
            print("[Skip]")
            return

        if not self.ana_api_info():
            print("ana_api_info failed")
            return

        try:
            device = torch.device("cuda:0")
            torch.set_default_device(device)
            if not self.gen_torch_input():
                print("gen_torch_input failed")
                return
            if "paddle.Tensor." in self.api_config.api_name:
                api = getattr(self.torch_args[0], self.torch_api_str[self.torch_api_str.rindex(".")+1:])
                args = []
                if len(self.torch_args) > 1:
                    args = self.torch_args[1:]
                torch_output = api(*tuple(args), **self.torch_kwargs)
                del args
            else:
                torch_output = self.torch_api(*tuple(self.torch_args), **self.torch_kwargs)
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            print("[torch error]", self.api_config.config, "\n", str(err))
            api_config_torch_error.write(self.api_config.config+"\n")
            api_config_torch_error.flush()
            if "CUDA error" in str(err):
                raise Exception(err)
            return

        if self.need_check_grad():
            try:
                inputs_list = self.get_torch_input_list()
                result_outputs, result_outputs_grads = self.gen_torch_output_and_output_grad(torch_output)
                if len(inputs_list) != 0 and len(result_outputs) != 0 and len(result_outputs_grads) != 0:
                    del self.torch_args
                    del self.torch_kwargs
                    torch_out_grads = torch.autograd.grad(result_outputs, inputs_list, grad_outputs=result_outputs_grads)
                    torch_grad_success = True
                else:
                    torch_grad_success = False
                    torch_out_grads = None
                del inputs_list
                del result_outputs
                del result_outputs_grads
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                torch_grad_success = False
                if "CUDA error" in str(err):
                    raise Exception(err)
        else:
            del self.torch_args
            del self.torch_kwargs

        try:
            if not self.gen_paddle_input():
                print("gen_paddle_input failed")
                return
            if "paddle.Tensor." in self.api_config.api_name:
                api = getattr(self.paddle_args[0], self.api_config.api_name[self.api_config.api_name.rindex(".")+1:])
                args = []
                if len(self.paddle_args) > 1:
                    args = self.paddle_args[1:]
                paddle_output = api(*tuple(args), **self.paddle_kwargs)
            else:
                paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
        except Exception as err:
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

        if isinstance(paddle_output, paddle.Tensor):
            if isinstance(torch_output, torch.Tensor):
                try:
                    if paddle_output.dtype == paddle.bfloat16:
                        paddle_output = paddle.cast(paddle_output, dtype="float32")
                        torch_output = torch_output.to(dtype=torch.float32)
                    self.np_assert_accuracy(paddle_output.numpy(), torch_output.cpu().numpy(), 1e-2, 1e-2, self.api_config)
                except Exception as err:
                    print("[accuracy error]", self.api_config.config, "\n", str(err))
                    api_config_accuracy_error.write(self.api_config.config+"\n")
                    api_config_accuracy_error.flush()
                    return
            else:
                print("[accuracy error]", self.api_config.config, "\n[output type diff error1], ", type(torch_output))
                api_config_accuracy_error.write(self.api_config.config+"\n")
                api_config_accuracy_error.flush()
                return
        elif isinstance(paddle_output, (list, tuple)):
            if isinstance(paddle_output, tuple):
                paddle_output = list(paddle_output)
            if not isinstance(torch_output, (list, tuple)):
                print("[output type diff error]", self.api_config.config)
                return
            if isinstance(torch_output, tuple):
                torch_output = list(torch_output)
            if len(paddle_output) != len(torch_output):
                print("[accuracy error]", self.api_config.config, "\n[output type diff error2], ", len(paddle_output), len(torch_output))
                api_config_accuracy_error.write(self.api_config.config+"\n")
                api_config_accuracy_error.flush()
                return
            for i in range(len(paddle_output)):
                if not isinstance(paddle_output[i], paddle.Tensor):
                    print("not compare ", paddle_output[i], torch_output[i])
                elif not isinstance(torch_output[i], torch.Tensor):
                    print("[accuracy error]", self.api_config.config, "\n[output type diff error3], ", type(torch_output[i]))
                    api_config_accuracy_error.write(self.api_config.config+"\n")
                    api_config_accuracy_error.flush()
                    return
                else:
                    try:
                        if paddle_output[i].dtype == paddle.bfloat16:
                            paddle_output[i] = paddle.cast(paddle_output[i], dtype="float32")
                            torch_output[i] = torch_output[i].to(dtype=torch.float32)
                        self.np_assert_accuracy(paddle_output[i].numpy(), torch_output[i].cpu().numpy(), 1e-2, 1e-2, self.api_config)
                    except Exception as err:
                        print("[accuracy error]", self.api_config.config, "\n", str(err))
                        api_config_accuracy_error.write(self.api_config.config+"\n")
                        api_config_accuracy_error.flush()
                        return

        if self.need_check_grad() and torch_grad_success:
            try:
                paddle_out_grads = None
                inputs_list = self.get_paddle_input_list()
                result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(paddle_output)
                if len(inputs_list) != 0 and len(result_outputs) != 0 and len(result_outputs_grads) != 0:
                    del self.paddle_args
                    del self.paddle_kwargs
                    paddle_out_grads = paddle.grad(result_outputs, inputs_list, grad_outputs=result_outputs_grads,allow_unused=True)
                del inputs_list
                del result_outputs
                del result_outputs_grads
            except Exception as err:
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

            if isinstance(paddle_out_grads, paddle.Tensor):
                if isinstance(torch_out_grads, torch.Tensor):
                    try:
                        if paddle_out_grads.dtype == paddle.bfloat16:
                            paddle_out_grads = paddle.cast(paddle_out_grads, dtype="float32")
                            torch_out_grads = torch_out_grads.to(dtype=torch.float32)
                        self.np_assert_accuracy(paddle_out_grads.numpy(), torch_out_grads.cpu().numpy(), 1e-2, 1e-2, self.api_config)
                    except Exception as err:
                        print("[accuracy error]", self.api_config.config, "\n", str(err))
                        api_config_accuracy_error.write(self.api_config.config+"\n")
                        api_config_accuracy_error.flush()
                        return
                else:
                    print("[accuracy error]", self.api_config.config, "\n[output type diff error1], ", type(torch_out_grads))
                    api_config_accuracy_error.write(self.api_config.config+"\n")
                    api_config_accuracy_error.flush()
                    return
            elif isinstance(paddle_out_grads, (list, tuple)):
                if isinstance(paddle_out_grads, tuple):
                    paddle_out_grads = list(paddle_out_grads)
                if not isinstance(torch_out_grads, (list, tuple)):
                    print("[output type diff error]", self.api_config.config)
                    return
                if isinstance(torch_out_grads, tuple):
                    torch_out_grads = list(torch_out_grads)
                if len(paddle_out_grads) != len(torch_out_grads):
                    print("[accuracy error]", self.api_config.config, "\n[output type diff error2], ", len(paddle_out_grads), len(torch_out_grads))
                    api_config_accuracy_error.write(self.api_config.config+"\n")
                    api_config_accuracy_error.flush()
                    return
                for i in range(len(paddle_out_grads)):
                    if not isinstance(paddle_out_grads[i], paddle.Tensor):
                        print("not compare ", paddle_out_grads[i], torch_out_grads[i])
                    elif not isinstance(torch_out_grads[i], torch.Tensor):
                        print("[accuracy error]", self.api_config.config, "\n[output type diff error3], ", type(torch_out_grads[i]))
                        api_config_accuracy_error.write(self.api_config.config+"\n")
                        api_config_accuracy_error.flush()
                        return
                    else:
                        try:
                            if paddle_out_grads[i].dtype == paddle.bfloat16:
                                paddle_out_grads[i] = paddle.cast(paddle_out_grads[i], dtype="float32")
                                torch_out_grads[i] = torch_out_grads[i].to(dtype=torch.float32)
                            self.np_assert_accuracy(paddle_out_grads[i].numpy(), torch_out_grads[i].cpu().numpy(), 1e-2, 1e-2, self.api_config)
                        except Exception as err:
                            print("[accuracy error]", self.api_config.config, "\n", str(err))
                            api_config_accuracy_error.write(self.api_config.config+"\n")
                            api_config_accuracy_error.flush()
                            return

        print("[Pass]", self.api_config.config)
        api_config_pass.write(self.api_config.config+"\n")
        api_config_pass.flush()
  

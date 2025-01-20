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

api_config_accuracy_error = open("/data/OtherRepo/PaddleAPITest/tester/api_config/api_config_accuracy_error.txt", "w")
api_config_paddle_error = open("/data/OtherRepo/PaddleAPITest/tester/api_config/api_config_paddle_error.txt", "w")
api_config_pass = open("/data/OtherRepo/PaddleAPITest/tester/api_config/api_config_pass.txt", "w")
api_config_torch_error = open("/data/OtherRepo/PaddleAPITest/tester/api_config/api_config_torch_error.txt", "w")

class APITestAccuracy(APITestBase):
    def __init__(self, api_config):
        self.api_config = api_config
    def test(self):
        if not self.ana_api_info():
            return

        device = torch.device("cuda:0")
        torch.set_default_device(device)

        try:
            torch_output = self.torch_api(*tuple(self.torch_args), **self.torch_kwargs)
        except Exception as err:
            print("[torch error]", self.api_config.config, "\n", str(err))
            api_config_torch_error.write(self.api_config.config+"\n")
            return

        try:
            paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
        except Exception as err:
            print("[paddle error]", self.api_config.config, "\n", str(err))
            api_config_paddle_error.write(self.api_config.config+"\n")
            return

        if isinstance(paddle_output, paddle.Tensor):
            if isinstance(torch_output, torch.Tensor):
                try:
                    self.np_assert_accuracy(paddle_output.numpy(), torch_output.cpu().numpy(), 1e-2, 1e-2, self.api_config)
                except Exception as err:
                    print("[accuracy error]", self.api_config.config, "\n", str(err))
                    api_config_accuracy_error.write(self.api_config.config+"\n")
                    return
            else:
                print("[output type diff error]", self.api_config.config)
        elif isinstance(paddle_output, (list, tuple)):
            if isinstance(paddle_output, tuple):
                paddle_output = list(paddle_output)
            if not isinstance(torch_output, (list, tuple)):
                print("[output type diff error]", self.api_config.config)
                return
            if isinstance(torch_output, tuple):
                torch_output = list(torch_output)
            if len(paddle_output) != len(torch_output):
                print("[output type diff error]", self.api_config.config)
                return
            for i in range(len(paddle_output)):
                try:
                    self.np_assert_accuracy(paddle_output[i].numpy(), torch_output[i].cpu().numpy(), 1e-2, 1e-2, self.api_config)
                except Exception as err:
                    print("[accuracy error]", self.api_config.config, "\n", str(err))
                    api_config_accuracy_error.write(self.api_config.config+"\n")
                    return

        print("[Pass]", self.api_config.config)
        api_config_pass.write(self.api_config.config+"\n")
  

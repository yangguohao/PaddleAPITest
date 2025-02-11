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
            with paddle.no_grad():
                if not self.gen_paddle_input():
                    return
                paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
                self.clear_paddle_tensor()
        except Exception as err:
            print("[paddle error]", self.api_config.config, "\n", str(err))
            paddle_output = None
            api_config_paddle_error.write(self.api_config.config+"\n")
            return

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            print("[cuda error]", self.api_config.config, "\n", str(err))
            paddle_output = None
            api_config_paddle_error.write(self.api_config.config+"\n")
            return

        paddle_output = None
        print("[Pass]", self.api_config.config)
        api_config_pass.write(self.api_config.config+"\n")
  

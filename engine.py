from tester import TensorConfig, APIConfig, analyse_configs
from tester import APITestAccuracy
import re
import collections
import paddle
import numpy
import math
import json
import torch
import paddle
import inspect

if __name__ == '__main__':
    api_configs = analyse_configs("/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config_big_tensor.txt")
    for i in range(len(api_configs)):
        api_config = api_configs[i]
        print("test begin:", api_config.config, flush=True)
        case = APITestAccuracy(api_config)
        case.test()
        case.clear_tensor()
        api_configs[i] = None
        del case
        del api_config
        torch.cuda.empty_cache()
        paddle.device.cuda.empty_cache()

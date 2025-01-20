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
    api_configs = analyse_configs("/data/OtherRepo/PaddleAPITest/tester/api_config/api_config.txt")
    for api_config in api_configs:
        case = APITestAccuracy(api_config)
        case.test()
        case.clear_tensor()

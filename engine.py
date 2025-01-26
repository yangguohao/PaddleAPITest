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
import argparse
import subprocess

# if __name__ == '__main__':
#     api_configs = analyse_configs("/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config_big_tensor.txt")
#     for i in range(len(api_configs)):
#         api_config = api_configs[i]
#         print("test begin:", api_config.config, flush=True)
#         case = APITestAccuracy(api_config)
#         case.test()
#         case.clear_tensor()
#         api_configs[i] = None
#         del case
#         del api_config
#         torch.cuda.empty_cache()
#         paddle.device.cuda.empty_cache()

# python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config_big_tensor.txt > log.log 2>&1
def main():
    parser = argparse.ArgumentParser(
        description='API Test'
    )
    parser.add_argument(
        '--api_config_file',
        default="",
    )
    parser.add_argument(
        '--api_config',
        default="",
    )

    options = parser.parse_args()

    if options.api_config != "":
        api_config = APIConfig(options.api_config)
        print("test begin:", api_config.config, flush=True)
        case = APITestAccuracy(api_config)
        case.test()
        case.clear_tensor()
        del case
        del api_config
        torch.cuda.empty_cache()
        paddle.device.cuda.empty_cache()
    elif options.api_config_file != "":
        with open(options.api_config_file, "r") as f:
            configs = f.readlines()
            f.close()

        for config in configs:
            config = config.replace("\n", "")
            cmd = ["python", "engine.py", "--api_config=" + config]
            res = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            # res.wait()
            print(str(res.stdout.read(), encoding="utf-8"), flush=True)
            print(str(res.stderr.read(), encoding="utf-8"), flush=True)
            # res.terminate()

if __name__ == '__main__':
    main()

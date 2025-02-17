from tester import TensorConfig, APIConfig, analyse_configs
from tester import APITestAccuracy, APITestPaddleOnly, APITestCINNVSDygraph
import re
import collections
import paddle
import numpy
import math
import json
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

# python engine.py --api_config_file=/host_home/wanghuan29/PaddleAPITest/tester/api_config/api_config_big_tensor.txt > tester/api_config/test_log/log.log 2>&1
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
    parser.add_argument(
        '--paddle_only',
        default=False,
    )
    parser.add_argument(
        '--paddle_cinn',
        default=False,
    )
    parser.add_argument(
        '--accuracy',
        default=False,
    )

    options = parser.parse_args()

    if options.paddle_only:
        if options.api_config != "":
            api_config = APIConfig(options.api_config)
            print("test begin:", api_config.config, flush=True)
            print(api_config)
            case = APITestPaddleOnly(api_config)
            case.test()
            case.clear_tensor()
            del case
            del api_config
        elif options.api_config_file != "":
            api_configs = analyse_configs(options.api_config_file)
            for i in range(len(api_configs)):
                api_config = api_configs[i]
                print("test begin:", api_config.config, flush=True)
                case = APITestPaddleOnly(api_config)
                case.test()
                case.clear_tensor()
                api_configs[i] = None
                del case
                del api_config
    if options.paddle_cinn:
        if options.api_config != "":
            api_config = APIConfig(options.api_config)
            print("test begin:", api_config.config, flush=True)
            print(api_config)
            case = APITestCINNVSDygraph(api_config)
            case.test()
            case.clear_tensor()
            del case
            del api_config
        elif options.api_config_file != "":
            checkpoint_r = open("/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/checkpoint.txt", "r")
            finish_configs = checkpoint_r.readlines()
            checkpoint_r.close()
            checkpoint = open("/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/checkpoint.txt", "a")
            api_configs = open(options.api_config_file, "r")
            for api_config_str in api_configs:
                if api_config_str in finish_configs:
                    continue
                api_config = APIConfig(api_config_str)
                print("test begin:", api_config.config, flush=True)
                checkpoint.write(api_config.config+"\n")
                checkpoint.flush()
                case = APITestCINNVSDygraph(api_config)
                case.test()
                case.clear_tensor()
                del case
                del api_config
    if options.accuracy:
        if options.api_config != "":
            api_config = APIConfig(options.api_config)
            print("test begin:", api_config.config, flush=True)
            case = APITestAccuracy(api_config)
            case.test()
            case.clear_tensor()
            del case
            del api_config
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

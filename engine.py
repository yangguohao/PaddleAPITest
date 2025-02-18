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
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))[0:os.path.dirname(os.path.realpath(__file__)).index("PaddleAPITest")+13]

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

    test_class = APITestAccuracy
    if options.paddle_only:
        test_class = APITestPaddleOnly
    elif options.paddle_cinn:
        test_class = APITestCINNVSDygraph
    elif options.accuracy:
        test_class = APITestAccuracy            

    if options.api_config != "":
        api_config = APIConfig(options.api_config)
        print("test begin:", api_config.config, flush=True)
        case = test_class(api_config)
        case.test()
        case.clear_tensor()
        del case
        del api_config
    elif options.api_config_file != "":
        try:
            checkpoint_r = open(DIR_PATH+"/tester/api_config/test_log/checkpoint.txt", "r")
            finish_configs = set(checkpoint_r.readlines())
            checkpoint_r.close()
        except Exception as err:
            finish_configs = set()
        checkpoint = open(DIR_PATH+"/tester/api_config/test_log/checkpoint.txt", "a")
        api_config_file = open(options.api_config_file, "r")
        api_configs = set(api_config_file.readlines())
        api_configs = api_configs - finish_configs
        for api_config_str in sorted(api_configs):
            api_config = APIConfig(api_config_str)
            print("test begin:", api_config.config, flush=True)
            checkpoint.write(api_config.config+"\n")
            checkpoint.flush()
            case = test_class(api_config)
            case.test()
            case.clear_tensor()
            del case
            del api_config



        # elif options.api_config_file != "":
        #     with open(options.api_config_file, "r") as f:
        #         configs = f.readlines()
        #         f.close()

        #     for config in configs:
        #         config = config.replace("\n", "")
        #         cmd = ["python", "engine.py", "--api_config=" + config]
        #         res = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        #         # res.wait()
        #         print(str(res.stdout.read(), encoding="utf-8"), flush=True)
        #         print(str(res.stderr.read(), encoding="utf-8"), flush=True)
        #         # res.terminate()

if __name__ == '__main__':
    main()

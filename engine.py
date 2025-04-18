import argparse
import os
from datetime import datetime

from filelock import FileLock
import filelock

from tester import (APIConfig, APITestAccuracy, APITestCINNVSDygraph,
                    APITestPaddleOnly)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))[0:os.path.dirname(os.path.realpath(__file__)).index("PaddleAPITest")+13]
TEST_LOG_PATH = os.path.join(DIR_PATH, "tester/api_config/test_log")
os.makedirs(TEST_LOG_PATH, exist_ok=True)
CHECKPOINT_FILE = os.path.join(TEST_LOG_PATH, "checkpoint.txt")

def get_notsupport_config():
    not_support_files = [
        "tester/api_config/api_config_merged_not_support_amp.txt",
        "tester/api_config/api_config_merged_not_support_arange.txt",
        "tester/api_config/api_config_merged_not_support_empty.txt",
        "tester/api_config/api_config_merged_not_support_eye.txt",
        "tester/api_config/api_config_merged_not_support_flatten.txt",
        "tester/api_config/api_config_merged_not_support_full.txt",
        "tester/api_config/api_config_merged_not_support_getset_item.txt",
        "tester/api_config/api_config_merged_not_support_reshape.txt",
        "tester/api_config/api_config_merged_not_support_slice.txt",
        "tester/api_config/api_config_merged_not_support_sparse.txt",
        "tester/api_config/api_config_merged_not_support_tensor_init.txt",
        "tester/api_config/api_config_merged_not_support_topk.txt",
        "tester/api_config/api_config_merged_not_support_zeros.txt",
        "tester/api_config/api_config_merged_not_support.txt",
    ]
    configs = set()

    for flie in not_support_files:
        with open(DIR_PATH+"/"+flie, "r") as f:
            origin_configs = f.readlines()
            f.close()

        for config in origin_configs:
            configs.add(config)
    return configs


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
    parser.add_argument(
        '--test_amp',
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
        print("test begin:", options.api_config, flush=True)
        try:
            api_config = APIConfig(options.api_config)
        except Exception as err:
            print("[config parse error]", options.api_config, str(err))
            return

        case = test_class(api_config, options.test_amp)
        case.test()
        case.clear_tensor()
        del case
        del api_config
    elif options.api_config_file != "":
        lock_file = CHECKPOINT_FILE + ".lock"  # 锁文件
        not_support_file = os.path.join(DIR_PATH, "tester/api_config/api_config_merged_not_support.txt")

        finish_configs = set()
        try:
            with FileLock(lock_file, timeout=30):
                with open(CHECKPOINT_FILE, "r") as checkpoint_r:
                    finish_configs = set(checkpoint_r.readlines())
        except FileNotFoundError:
            pass
        except filelock.Timeout:
            print("Timeout waiting for lock on", CHECKPOINT_FILE)
            return

        not_support_api_config = set()
        try:
            with open(not_support_file, "r") as f:
                not_support_api_config = set(f.readlines())
        except FileNotFoundError:
            pass

        with open(options.api_config_file, "r") as api_config_file:
            api_configs = set(api_config_file.readlines())
        api_configs = api_configs - finish_configs - not_support_api_config
        api_configs = sorted(api_configs)

        for api_config_str in api_configs:
            with FileLock(lock_file, timeout=30):  # 使用文件锁保护写入
                with open(CHECKPOINT_FILE, "a") as checkpoint:
                    checkpoint.write(api_config_str)
                    checkpoint.flush()

            print(datetime.now(), "test begin:", api_config_str, flush=True)
            try:
                api_config = APIConfig(api_config_str)
            except Exception as err:
                print("[config parse error]", api_config_str, str(err))
                continue

            case = test_class(api_config, options.test_amp)
            try:
                case.test()
            except Exception as err:
                if "CUDA error" in str(err) or "memory corruption" in str(err) or "CUDA out of memory" in str(err):
                    exit(1)
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

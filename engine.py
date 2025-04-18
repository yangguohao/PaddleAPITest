import argparse
import atexit
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import set_start_method

from tester import (APIConfig, APITestAccuracy, APITestCINNVSDygraph,
                    APITestPaddleOnly)
from tester.api_config.log_writer import (DIR_PATH, flush_buffer, read_log,
                                          write_to_log)

_executor = None

def cleanup():
    global _executor
    try:
        if _executor is not None:
            _executor.shutdown(wait=False)
            _executor = None
    except Exception as e:
        print(f"Error during cleanup: {e}", flush=True)

def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)

def register_cleanup():
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)

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


def run_test_case(api_config_str, options, gpu_id):
    """Run a single test case for the given API configuration."""
    atexit.register(flush_buffer)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    test_class = APITestAccuracy
    if options.paddle_only:
        test_class = APITestPaddleOnly
    elif options.paddle_cinn:
        test_class = APITestCINNVSDygraph
    elif options.accuracy:
        test_class = APITestAccuracy

    write_to_log("checkpoint", api_config_str)
    print(f"{datetime.now()} test begin: {api_config_str}", flush=True)
    try:
        api_config = APIConfig(api_config_str)
    except Exception as err:
        print(f"[config parse error] {api_config_str} {str(err)}", flush=True)
        return False

    case = test_class(api_config, options.test_amp)
    try:
        case.test()
    except Exception as err:
        print(f"[test error] {api_config_str} {str(err)}", flush=True)
        return False
    finally:
        case.clear_tensor()
        del case
        del api_config
    return True


def main():
    global _executor
    set_start_method('spawn', force=True)
    register_cleanup()

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
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=1,
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
        finish_configs = read_log("checkpoint")
        not_support_api_config = read_log("not_support")
        with open(options.api_config_file, "r") as api_config_file:
            api_configs = set(api_config_file.readlines())
        api_configs = api_configs - finish_configs - not_support_api_config
        api_configs = sorted(api_configs)
        print(len(api_configs), "cases will be tested.", flush=True)

        if options.num_gpus > 1:
            # Multi GPUs execution
            num_workers = options.num_gpus
            print(f"Using {num_workers} GPU(s) via {num_workers} worker processes.")
            _executor = ProcessPoolExecutor(max_workers=num_workers)
            try:
                futures = []
                for i, config_str in enumerate(api_configs):
                    gpu_id = (i + 4) % num_workers
                    future = _executor.submit(run_test_case, config_str.strip(), options, gpu_id)
                    futures.append(future)
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"[FATAL] A worker process might have crashed: {exc}")
            finally:
                _executor.shutdown()
                flush_buffer()
                _executor = None
        else:
            # Single GPU execution
            for config in api_configs:
                run_test_case(config, options, gpu_id=0)

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

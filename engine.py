import argparse
import atexit
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import active_children, set_start_method

from tester import (APIConfig, APITestAccuracy, APITestCINNVSDygraph,
                    APITestPaddleOnly)
from tester.api_config.log_writer import DIR_PATH, write_to_log, read_log

_executor = None


def cleanup():
    global _executor
    print(f"{datetime.now()} Starting cleanup", flush=True)

    if _executor is not None:
        try:
            print(f"{datetime.now()} Shutting down executor", flush=True)
            _executor.shutdown(wait=False)
            _executor = None
            print(f"{datetime.now()} Executor shutdown completed", flush=True)
        except Exception as e:
            print(f"{datetime.now()} Error shutting down executor: {e}", flush=True)

    for process in active_children():
        pid = process.pid
        try:
            if process.is_alive():
                print(f"{datetime.now()} Terminating process {pid}", flush=True)
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    print(f"{datetime.now()} Killing process {pid}", flush=True)
                    process.kill()
                    process.join()
        except Exception as e:
            print(f"{datetime.now()} Error terminating process {pid}: {e}", flush=True)
        finally:
            print(f"{datetime.now()} Process {pid} terminated", flush=True)

    print(f"{datetime.now()} Cleanup completed", flush=True)


def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)

def register_cleanup():
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


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
        with open(os.path.join(DIR_PATH, flie), "r") as f:
            origin_configs = f.readlines()
            f.close()

        for config in origin_configs:
            configs.add(config)
    return configs


def run_test_case(api_config_str, options, gpu_id):
    """Run a single test case for the given API configuration."""
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
    print(f"main process: {os.getpid()}")
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
        default=0,
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
        print(len(finish_configs), "cases have been tested.", flush=True)
        with open(os.path.join(DIR_PATH, "tester/api_config/api_config_merged_not_support.txt"), "r") as f:
            not_support_api_config = set(line.strip() for line in f if line.strip())
        with open(options.api_config_file, "r") as f:
            api_configs = set(line.strip() for line in f if line.strip())
        api_configs = api_configs - finish_configs - not_support_api_config
        api_configs = sorted(api_configs)
        print(len(api_configs), "cases will be tested.", flush=True)

        if options.num_gpus > 0:
            # Multi GPUs execution
            num_gpus = options.num_gpus
            num_workers = num_gpus * 2
            print(f"Using {num_gpus} GPU(s) via {num_workers} worker processes.")
            _executor = ProcessPoolExecutor(max_workers=num_workers)
            try:
                futures = []
                for i, config_str in enumerate(api_configs):
                    gpu_id = i % num_gpus
                    future = _executor.submit(run_test_case, config_str.strip(), options, gpu_id)
                    futures.append(future)
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"[FATAL] A worker process might have crashed: {exc}")
            finally:
                _executor.shutdown(wait=True)
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

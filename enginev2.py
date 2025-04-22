import argparse
import atexit
from contextlib import contextmanager
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import active_children, set_start_method
from typing import Set

import numpy as np

from tester import APIConfig, APITestAccuracy, APITestCINNVSDygraph, APITestPaddleOnly
from tester.api_config.log_writer import DIR_PATH, read_log, write_to_log


def cleanup(executor=None):
    print(f"{datetime.now()} Cleanup started", flush=True)
    if executor is not None:
        try:
            executor.shutdown(wait=False)
        except Exception as e:
            print(f"{datetime.now()} Error shutting down executor: {e}", flush=True)

    for process in active_children():
        pid = process.pid
        try:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join()
        except Exception as e:
            print(f"{datetime.now()} Error terminating process {pid}: {e}", flush=True)
        finally:
            print(f"{datetime.now()} Process {pid} terminated", flush=True)
    print(f"{datetime.now()} Cleanup completed", flush=True)


def get_notsupport_config() -> Set[str]:
    """Load not-supported API configurations from files."""
    not_support_files = [
        f"tester/api_config/api_config_merged_not_support_{suffix}.txt"
        for suffix in [
            "amp",
            "arange",
            "empty",
            "eye",
            "flatten",
            "full",
            "getset_item",
            "reshape",
            "slice",
            "sparse",
            "tensor_init",
            "topk",
            "zeros",
            "",
        ]
    ]
    configs = set()
    for file_path in not_support_files:
        try:
            with open(os.path.join(DIR_PATH, file_path), "r") as f:
                configs.update(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found", flush=True)
    return configs


def estimate_timeout(api_config) -> float:
    """Estimate timeout based on tensor size in APIConfig."""
    DEFAULT_TIMEOUT = 1800
    # try:
    #     tensor_shape = None
    #     if api_config.args and hasattr(api_config.args[0], 'shape'):
    #         tensor_shape = api_config.args[0].shape
    #     elif api_config.kwargs:
    #         for value in api_config.kwargs.values():
    #             if hasattr(value, 'shape'):
    #                 tensor_shape = value.shape
    #                 break
    #     if tensor_shape and isinstance(tensor_shape, (list, tuple)):
    #         num_elements = np.prod(tensor_shape)
    #         timeout = min(3600, max(300, 0.0001 * num_elements))
    #         return timeout
    # except Exception as e:
    #     pass
    return DEFAULT_TIMEOUT


def run_test_case(
    api_config_str: str, test_class: type, test_amp: bool, gpu_id: int
) -> bool:
    """Run a single test case for the given API configuration."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    write_to_log("checkpoint", api_config_str)
    print(f"{datetime.now()} test begin: {api_config_str}", flush=True)
    try:
        api_config = APIConfig(api_config_str)
    except Exception as err:
        print(f"[config parse error] {api_config_str} {str(err)}", flush=True)
        return False

    case = test_class(api_config, test_amp)
    try:
        case.test()
        return True
    except Exception as err:
        print(f"[test error] {api_config_str} {str(err)}", flush=True)
        return False
    finally:
        case.clear_tensor()
        del case
        del api_config


def main():
    set_start_method("spawn", force=True)
    executor = None

    def cleanup_handler(*args):
        cleanup(executor)
        sys.exit(1)

    atexit.register(lambda: cleanup(executor))
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    parser = argparse.ArgumentParser(description="API Test")
    parser.add_argument("--api_config_file", default="")
    parser.add_argument("--api_config", default="")
    parser.add_argument("--paddle_only", action="store_true")
    parser.add_argument("--paddle_cinn", action="store_true")
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--test_amp", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--num_workers_per_gpu", type=int, default=2)
    options = parser.parse_args()

    test_class = APITestAccuracy
    if options.paddle_only:
        test_class = APITestPaddleOnly
    elif options.paddle_cinn:
        test_class = APITestCINNVSDygraph
    elif options.accuracy:
        test_class = APITestAccuracy

    if options.api_config:
        # Single config execution
        print(f"Test begin: {options.api_config}", flush=True)
        try:
            api_config = APIConfig(options.api_config)
        except Exception as err:
            print(f"[config parse error] {options.api_config} {str(err)}", flush=True)
            return

        case = test_class(api_config, options.test_amp)
        try:
            case.test()
        except Exception as err:
            print(f"[test error] {options.api_config}: {err}", flush=True)
        finally:
            case.clear_tensor()
            del case
            del api_config
    elif options.api_config_file:
        # Batch execution
        finish_configs = read_log("checkpoint")
        print(len(finish_configs), "cases have been tested.", flush=True)
        try:
            with open(options.api_config_file, "r") as f:
                api_configs = {line.strip() for line in f if line.strip()}
        except FileNotFoundError:
            print(
                f"Error: api config file {options.api_config_file} not found",
                flush=True,
            )
            return

        api_configs = sorted(api_configs - finish_configs)
        print(len(api_configs), "cases will be tested.", flush=True)

        if options.num_gpus > 0:
            # Multi GPUs execution
            num_gpus = options.num_gpus
            num_workers = num_gpus * options.num_workers_per_gpu
            print(
                f"Using {num_gpus} GPU(s) via {num_workers} worker processes.",
                flush=True,
            )
            executor = ProcessPoolExecutor(max_workers=num_workers)
            try:
                futures = []
                for i, config in enumerate(api_configs):
                    future = executor.submit(
                        run_test_case, config, test_class, options.test_amp, i % num_gpus
                    )
                    futures.append((future, config))
                for future, config, timeout in futures:
                    timeout = estimate_timeout(config)
                    try:
                        future.result(timeout=timeout)
                    except TimeoutError:
                        write_to_log("timeout", config)
                        print(
                            f"[TIMEOUT] {config} exceeded {timeout}s, terminating...",
                            flush=True,
                        )
                        process = future._process
                        try:
                            if process.is_alive():
                                process.terminate()
                                process.join(timeout=2)
                                if process.is_alive():
                                    process.kill()
                                    process.join()
                        except Exception as e:
                            print(f"[TIMEOUT] {config} exceeded {timeout}s", flush=True)
                            pass
                    except Exception as exc:
                        write_to_log("crash", config)
                        print(f"[FATAL] Worker process crashed: {exc}", flush=True)
            finally:
                cleanup(executor)
        else:
            # Single GPU execution
            for config in api_configs:
                run_test_case(config, test_class, options.test_amp, gpu_id=0)


if __name__ == "__main__":
    main()

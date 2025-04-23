import argparse
import math
import os
import signal
import sys
from concurrent.futures import TimeoutError, as_completed
from datetime import datetime
from multiprocessing import Lock, Manager, set_start_method

import psutil
from pebble import ProcessPool

from tester import (APIConfig, APITestAccuracy, APITestCINNVSDygraph,
                    APITestPaddleOnly)
from tester.api_config.log_writer import (DIR_PATH, aggregate_logs, read_log,
                                          set_engineV2, write_to_log)


def cleanup(pool):
    print(f"{datetime.now()} Cleanup started", flush=True)
    if pool is not None:
        try:
            if pool.active:
                pool.stop()
                pool.join(timeout=5)
        except Exception as e:
            print(f"{datetime.now()} Error shutting down executor: {e}", flush=True)
    print(f"{datetime.now()} Cleanup completed", flush=True)


def get_notsupport_config():
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
        ]
    ]
    configs = set()
    for file_name in not_support_files:
        file_path = os.path.join(DIR_PATH, file_name)
        try:
            with open(file_path, "r") as f:
                configs.update(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            pass
    return configs


def estimate_timeout(api_config) -> float:
    """Estimate timeout based on tensor size in APIConfig."""
    TIMEOUT_STEPS = (
        # (1e4, 10),
        # (1e5, 30),
        (1e6, 60),
        (1e7, 300),
        (1e8, 1800),
        (float("inf"), 3600),
    )
    try:
        api_config = APIConfig(api_config)
        first = None
        if api_config.args:
            first = api_config.args[0]
        elif api_config.kwargs:
            first = next(iter(api_config.kwargs.values()))
        if first is not None and hasattr(first, "shape"):
            total_elements = math.prod(first.shape)
            for threshold, timeout in TIMEOUT_STEPS:
                if total_elements <= threshold:
                    return timeout
    except Exception:
        pass
    return TIMEOUT_STEPS[-1][1]


def init_worker(gpu_worker_list, lock, num_gpus, num_workers_per_gpu):
    set_engineV2()
    my_pid = os.getpid()
    assigned_gpu = -1

    try:
        with lock:
            for gpu_id in range(num_gpus):
                workers = gpu_worker_list[gpu_id]
                if len(workers) < num_workers_per_gpu:
                    workers.append(my_pid)
                    assigned_gpu = gpu_id
                    break
        if assigned_gpu == -1:
            with lock:
                for gpu_id in range(num_gpus):
                    workers = gpu_worker_list[gpu_id]
                    dead_workers = [
                        pid for pid in workers if not psutil.pid_exists(pid)
                    ]
                    if dead_workers:
                        for pid in dead_workers:
                            workers.remove(pid)
                        workers.append(my_pid)
                        assigned_gpu = gpu_id
                        break
        if assigned_gpu == -1:
            raise RuntimeError(f"Worker {my_pid} could not be assigned a GPU.")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
        print(
            f"{datetime.now()} Worker PID: {my_pid}, Assigned GPU ID: {assigned_gpu}",
            flush=True,
        )
    except Exception as e:
        print(
            f"{datetime.now()} Worker {my_pid} initialization failed: {e}", flush=True
        )
        raise


def run_test_case(api_config_str, test_class, test_amp):
    """Run a single test case for the given API configuration."""
    write_to_log("checkpoint", api_config_str)
    print(f"{datetime.now()} {os.getpid()} test begin: {api_config_str}", flush=True)
    try:
        api_config = APIConfig(api_config_str)
    except Exception as err:
        print(f"[config parse error] {api_config_str} {str(err)}", flush=True)
        return

    case = test_class(api_config, test_amp)
    try:
        case.test()
    except Exception as err:
        print(f"[test error] {api_config_str} {str(err)}", flush=True)
    finally:
        case.clear_tensor()


def main():
    print(f"Main process id: {os.getpid()}")
    set_start_method("spawn")

    parser = argparse.ArgumentParser(description="API Test")
    parser.add_argument("--api_config_file", default="")
    parser.add_argument("--api_config", default="")
    parser.add_argument(
        "--paddle_only",
        default=False,
    )
    parser.add_argument(
        "--paddle_cinn",
        default=False,
    )
    parser.add_argument(
        "--accuracy",
        default=False,
    )
    parser.add_argument(
        "--test_amp",
        default=False,
    )
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
                api_configs = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            print(
                f"Error: api config file {options.api_config_file} not found",
                flush=True,
            )
            return

        api_configs = sorted(api_configs - finish_configs)
        all_cases = len(api_configs)
        fail_case = 0
        print(all_cases, "cases will be tested.", flush=True)

        if options.num_gpus > 0:
            # Multi GPUs execution
            BATCH_SIZE = 16384
            num_gpus = options.num_gpus
            num_workers_per_gpu = options.num_workers_per_gpu
            total_workers = num_gpus * num_workers_per_gpu
            print(
                f"Using {num_gpus} GPU(s) with {num_workers_per_gpu} worker(s) per GPU. Total workers: {total_workers}",
                flush=True,
            )

            manager = Manager()
            gpu_worker_list = manager.list([manager.list() for _ in range(num_gpus)])
            lock = Lock()

            pool = ProcessPool(
                max_workers=total_workers,
                initializer=init_worker,
                initargs=[gpu_worker_list, lock, num_gpus, num_workers_per_gpu],
            )

            def cleanup_handler(*args):
                cleanup(pool)
                sys.exit(1)

            signal.signal(signal.SIGINT, cleanup_handler)
            signal.signal(signal.SIGTERM, cleanup_handler)

            try:
                for batch_start in range(0, len(api_configs), BATCH_SIZE):
                    batch = api_configs[batch_start : batch_start + BATCH_SIZE]
                    futures = {}
                    for config in batch:
                        timeout = estimate_timeout(config)
                        future = pool.schedule(
                            run_test_case,
                            [config, test_class, options.test_amp],
                            timeout=timeout,
                        )
                        futures[future] = config

                    for future in as_completed(futures):
                        config = futures[future]
                        try:
                            future.result()
                        except TimeoutError as exc:
                            write_to_log("timeout", config)
                            print(
                                f"[timeout] Test case timed out for {config}: {exc}",
                                flush=True,
                            )
                            fail_case += 1
                        except Exception as exc:
                            write_to_log("crash", config)
                            print(
                                f"[fatal] Worker crashed for {config}: {exc}",
                                flush=True,
                            )
                            fail_case += 1
                    aggregate_logs(mkdir=True)
                print(f"{all_cases} cases tested, {fail_case} failed", flush=True)
                pool.close()
                pool.join()
            except Exception as e:
                print(f"Unexpected error: {e}", flush=True)
                cleanup(pool)
            finally:
                aggregate_logs()
        else:
            # Single GPU execution
            for config in api_configs:
                run_test_case(config, test_class, options.test_amp)


if __name__ == "__main__":
    main()

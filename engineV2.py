import argparse
import math
import os
import signal
import sys
from concurrent.futures import TimeoutError, as_completed
from datetime import datetime
from multiprocessing import Lock, Manager, set_start_method
from typing import TYPE_CHECKING

import psutil
from pebble import ProcessPool

if TYPE_CHECKING:
    from tester import (
        APIConfig,
        APITestAccuracy,
        APITestCINNVSDygraph,
        APITestPaddleOnly,
    )

from tester.api_config.log_writer import (aggregate_logs, read_log,
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


def estimate_timeout(api_config) -> float:
    """Estimate timeout based on tensor size in APIConfig."""
    TIMEOUT_STEPS = (
        # (1e4, 10),
        # (1e5, 30),
        (1e6, 90),
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


def init_worker_gpu(gpu_worker_list, lock, num_gpus, num_workers_per_gpu):
    set_engineV2()
    my_pid = os.getpid()

    try:
        with lock:
            min_workers = float("inf")
            assigned_gpu = -1
            for gpu_id in range(num_gpus):
                workers = gpu_worker_list[gpu_id]
                if len(workers) < min_workers:
                    min_workers = len(workers)
                    assigned_gpu = gpu_id

            if assigned_gpu != -1 and min_workers < num_workers_per_gpu:
                gpu_worker_list[assigned_gpu].append(my_pid)
            else:
                for gpu_id in range(num_gpus):
                    workers = gpu_worker_list[gpu_id]
                    workers[:] = [pid for pid in workers if psutil.pid_exists(pid)]
                    if len(workers) < num_workers_per_gpu:
                        workers.append(my_pid)
                        assigned_gpu = gpu_id
                        break

        if assigned_gpu == -1:
            raise RuntimeError(f"Worker {my_pid} could not be assigned a GPU.")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

        import paddle
        import torch

        from tester import (APIConfig, APITestAccuracy, APITestCINNVSDygraph,
                            APITestPaddleOnly)

        globals()["torch"] = torch
        globals()["paddle"] = paddle
        globals()["APIConfig"] = APIConfig
        globals()["APITestAccuracy"] = APITestAccuracy
        globals()["APITestCINNVSDygraph"] = APITestCINNVSDygraph
        globals()["APITestPaddleOnly"] = APITestPaddleOnly

        def signal_handler(*args):
            torch.cuda.empty_cache()
            paddle.device.cuda.empty_cache()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print(
            f"{datetime.now()} Worker PID: {my_pid}, Assigned GPU ID: {assigned_gpu}",
            flush=True,
        )
    except Exception as e:
        print(
            f"{datetime.now()} Worker {my_pid} initialization failed: {e}", flush=True
        )
        raise


def run_test_case(api_config_str, options):
    """Run a single test case for the given API configuration."""
    write_to_log("checkpoint", api_config_str)
    print(
        f"{datetime.now()} GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} {os.getpid()} test begin: {api_config_str}",
        flush=True,
    )
    try:
        api_config = APIConfig(api_config_str)
    except Exception as err:
        print(f"[config parse error] {api_config_str} {str(err)}", flush=True)
        return

    test_class = APITestAccuracy
    if options.paddle_only:
        test_class = APITestPaddleOnly
    elif options.paddle_cinn:
        test_class = APITestCINNVSDygraph
    elif options.accuracy:
        test_class = APITestAccuracy

    case = test_class(api_config, options.test_amp)
    try:
        case.test()
    except Exception as err:
        print(f"[test error] {api_config_str} {str(err)}", flush=True)
    finally:
        case.clear_tensor()
        del case


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
    parser.add_argument("--num_workers_per_gpu", type=int, default=1)
    # parser.add_argument("--num_cpus", type=int, default=0)
    options = parser.parse_args()

    # if options.num_gpus > 0 and options.num_cpus > 0:
    #     raise ValueError("Cannot use both --num_gpus and --num_cpus at the same time.")

    if options.api_config:
        # Single config execution
        from tester import (APIConfig, APITestAccuracy, APITestCINNVSDygraph,
                            APITestPaddleOnly)

        print(f"Test begin: {options.api_config}", flush=True)
        try:
            api_config = APIConfig(options.api_config)
        except Exception as err:
            print(f"[config parse error] {options.api_config} {str(err)}", flush=True)
            return

        test_class = APITestAccuracy
        if options.paddle_only:
            test_class = APITestPaddleOnly
        elif options.paddle_cinn:
            test_class = APITestCINNVSDygraph
        elif options.accuracy:
            test_class = APITestAccuracy

        case = test_class(api_config, options.test_amp)
        try:
            case.test()
        except Exception as err:
            print(f"[test error] {options.api_config}: {err}", flush=True)
        finally:
            case.clear_tensor()
            del case
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
        all_case = len(api_configs)
        fail_case = 0
        print(all_case, "cases will be tested.", flush=True)

        if options.num_gpus > 0:
            # Multi GPUs
            set_engineV2()
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
                initializer=init_worker_gpu,
                initargs=[gpu_worker_list, lock, num_gpus, num_workers_per_gpu],
            )

            from tester import APIConfig

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
                            [config, options],
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
                print(f"{all_case} cases tested, {fail_case} failed.", flush=True)
                pool.close()
                pool.join()
            except Exception as e:
                print(f"Unexpected error: {e}", flush=True)
                cleanup(pool)
            finally:
                aggregate_logs()
        # elif options.num_cpus > 0:
        #     # Multi CPUs
        #     from tester import (
        #         APIConfig,
        #         APITestAccuracy,
        #         APITestCINNVSDygraph,
        #         APITestPaddleOnly,
        #     )

        #     BATCH_SIZE = 16384
        #     num_cpus = options.num_cpus
        #     print(f"Using {num_cpus} CPU(s).", flush=True)

        #     pool = ProcessPool(max_workers=num_cpus)

        #     def cleanup_handler(*args):
        #         cleanup(pool)
        #         sys.exit(1)

        #     signal.signal(signal.SIGINT, cleanup_handler)
        #     signal.signal(signal.SIGTERM, cleanup_handler)

        #     try:
        #         for batch_start in range(0, len(api_configs), BATCH_SIZE):
        #             batch = api_configs[batch_start : batch_start + BATCH_SIZE]
        #             futures = {}
        #             for config in batch:
        #                 timeout = estimate_timeout(config)
        #                 future = pool.schedule(
        #                     run_test_case,
        #                     [config, options],
        #                     timeout=timeout,
        #                 )
        #                 futures[future] = config

        #             for future in as_completed(futures):
        #                 config = futures[future]
        #                 try:
        #                     future.result()
        #                 except TimeoutError as exc:
        #                     write_to_log("timeout", config)
        #                     print(
        #                         f"[timeout] Test case timed out for {config}: {exc}",
        #                         flush=True,
        #                     )
        #                     fail_case += 1
        #                 except Exception as exc:
        #                     write_to_log("crash", config)
        #                     print(
        #                         f"[fatal] Worker crashed for {config}: {exc}",
        #                         flush=True,
        #                     )
        #                     fail_case += 1
        #             aggregate_logs(mkdir=True)
        #         print(f"{all_cases} cases tested, {fail_case} failed", flush=True)
        #         pool.close()
        #         pool.join()
        #     except Exception as e:
        #         print(f"Unexpected error: {e}", flush=True)
        #         cleanup(pool)
        #     finally:
        #         aggregate_logs()
        else:
            # Single worker
            from tester import (APIConfig, APITestAccuracy,
                                APITestCINNVSDygraph, APITestPaddleOnly)

            globals()["APIConfig"] = APIConfig
            globals()["APITestAccuracy"] = APITestAccuracy
            globals()["APITestCINNVSDygraph"] = APITestCINNVSDygraph
            globals()["APITestPaddleOnly"] = APITestPaddleOnly

            for config in api_configs:
                run_test_case(config, options)
            print(f"{all_case} cases tested.", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()

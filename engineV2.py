import argparse
import errno
import gc
import math
import os
import signal
import sys
import time
from concurrent.futures import TimeoutError, as_completed
from datetime import datetime
from multiprocessing import Lock, Manager, cpu_count, set_start_method
from typing import TYPE_CHECKING

import pynvml
from pebble import ProcessExpired, ProcessPool

if TYPE_CHECKING:
    from tester import (
        APIConfig,
        APITestAccuracy,
        APITestCINNVSDygraph,
        APITestPaddleOnly,
        APITestPaddleGPUPerformance,
        APITestTorchGPUPerformance,
        APITestPaddleTorchGPUPerformance,
        APITestAccuracyStable,
    )
    import torch
    import paddle

from tester.api_config.log_writer import *

os.environ["FLAGS_use_system_allocator"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


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
    # TIMEOUT_STEPS = (
    #     (1e4, 10),
    #     (1e5, 30),
    #     (1e6, 90),
    #     (1e7, 300),
    #     (1e8, 1800),
    #     (float("inf"), 3600),
    # )
    # try:
    #     api_config = APIConfig(api_config)
    #     first = None
    #     if api_config.args:
    #         first = api_config.args[0]
    #     elif api_config.kwargs:
    #         first = next(iter(api_config.kwargs.values()))
    #     if first is not None and hasattr(first, "shape"):
    #         total_elements = math.prod(first.shape)
    #         for threshold, timeout in TIMEOUT_STEPS:
    #             if total_elements <= threshold:
    #                 return timeout
    # except Exception:
    #     pass
    # return TIMEOUT_STEPS[-1][1]
    return 1800


def validate_gpu_options(options) -> tuple:
    """Validate and normalize GPU-related options."""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()
    if device_count == 0:
        raise ValueError("No GPUs found")
    if options.gpu_ids:
        try:
            gpu_ids = [int(id) for id in options.gpu_ids.split(",") if id.strip()]
        except ValueError as e:
            raise ValueError(
                f"Invalid gpu_ids: {options.gpu_ids} (int expected)"
            ) from None
        if len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError(f"Invalid gpu_ids: {options.gpu_ids} (duplicates)")
        gpu_ids = sorted(list(set(gpu_ids)))
        if len(gpu_ids) > 1 and -1 in gpu_ids:
            raise ValueError(f"Invalid gpu_ids: {options.gpu_ids} (-1 allowed only)")
        if gpu_ids != [-1] and not all(0 <= id < device_count for id in gpu_ids):
            raise ValueError(
                f"Invalid gpu_ids: {options.gpu_ids} (valid range [0, {device_count}))"
            )
    else:
        gpu_ids = [-1]
    if (
        options.num_gpus < -1
        or options.num_gpus == 0
        or options.num_gpus > device_count
    ):
        raise ValueError(f"Invalid num_gpus: {options.num_gpus}")
    if options.num_gpus == -1:
        options.num_gpus = device_count if gpu_ids == [-1] else len(gpu_ids)
    if gpu_ids == [-1]:
        gpu_ids = list(range(options.num_gpus))
    elif len(gpu_ids) != options.num_gpus:
        raise ValueError(f"num_gpus {options.num_gpus} mismatches gpu_ids {gpu_ids}")
    if options.num_workers_per_gpu < -1 or options.num_workers_per_gpu == 0:
        raise ValueError(f"Invalid num_workers_per_gpu: {options.num_workers_per_gpu}")
    if options.required_memory <= 0:
        raise ValueError(f"Invalid required_memory: {options.required_memory}")
    return tuple(gpu_ids)


def parse_bool(value):
    if isinstance(value, str):
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        elif value in ["false", "0", "no", "n"]:
            return False
    else:
        raise ValueError(f"Invalid boolean value: {value} parsed from command line")


def check_gpu_memory(
    gpu_ids, num_workers_per_gpu, required_memory
):  # required_memory in GB
    assert isinstance(gpu_ids, tuple) and len(gpu_ids) > 0
    available_gpus = []
    max_workers_per_gpu = {}

    pynvml.nvmlInit()
    try:
        for gpu_id in gpu_ids:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                total_memory = int(mem_info.total) / (1024**3)  # Bytes to GB
                used_memory = int(mem_info.used) / (1024**3)  # Bytes to GB
                free_memory = total_memory - used_memory

                max_workers = int(free_memory // required_memory)
                if max_workers >= 1:
                    available_gpus.append(gpu_id)
                    max_workers_per_gpu[gpu_id] = (
                        max_workers
                        if num_workers_per_gpu == -1
                        else min(max_workers, num_workers_per_gpu)
                    )
            except pynvml.NVMLError as e:
                print(f"[WARNING] Failed to check GPU {gpu_id}: {str(e)}", flush=True)
                continue

    finally:
        pynvml.nvmlShutdown()

    return available_gpus, max_workers_per_gpu


def init_worker_gpu(
    gpu_worker_list, lock, available_gpus, max_workers_per_gpu, options
):
    if options.log_dir:
        set_test_log_path(options.log_dir)
    set_engineV2()
    my_pid = os.getpid()

    def pid_exists(pid):
        try:
            os.kill(pid, 0)
            return True
        except OSError as e:
            return e.errno == errno.EPERM

    try:
        with lock:
            assigned_gpu = -1
            max_available_slots = -1
            for gpu_id in available_gpus:
                workers = gpu_worker_list[gpu_id]
                workers[:] = [pid for pid in workers if pid_exists(pid)]
                available_slots = max_workers_per_gpu[gpu_id] - len(workers)
                if available_slots > max_available_slots:
                    max_available_slots = available_slots
                    assigned_gpu = gpu_id

            if assigned_gpu == -1:
                raise RuntimeError(f"Worker {my_pid} could not be assigned a GPU.")

            gpu_worker_list[assigned_gpu].append(my_pid)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

        import paddle
        import torch

        globals()["torch"] = torch
        globals()["paddle"] = paddle

        from tester import (APIConfig, APITestAccuracy, APITestAccuracyStable,
                            APITestCINNVSDygraph, APITestPaddleGPUPerformance,
                            APITestPaddleOnly,
                            APITestPaddleTorchGPUPerformance,
                            APITestTorchGPUPerformance)

        test_classes = {
            "APIConfig": APIConfig,
            "APITestAccuracy": APITestAccuracy,
            "APITestCINNVSDygraph": APITestCINNVSDygraph,
            "APITestPaddleOnly": APITestPaddleOnly,
            "APITestPaddleGPUPerformance": APITestPaddleGPUPerformance,
            "APITestTorchGPUPerformance": APITestTorchGPUPerformance,
            "APITestPaddleTorchGPUPerformance": APITestPaddleTorchGPUPerformance,
            "APITestAccuracyStable": APITestAccuracyStable,
        }
        globals().update(test_classes)

        def signal_handler(*args):
            torch.cuda.empty_cache()
            paddle.device.cuda.empty_cache()
            restore_stdio()
            close_process_files()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        if options.test_cpu:
            paddle.device.set_device("cpu")

        redirect_stdio()

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
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_id = int(cuda_visible.split(",")[0])

    write_to_log("checkpoint", api_config_str)
    print(
        f"{datetime.now()} GPU {gpu_id} {os.getpid()} test begin: {api_config_str}",
        flush=True,
    )

    pynvml.nvmlInit()
    try:
        while True:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_memory = int(mem_info.free) / (1024**3)  # Bytes to GB

            if free_memory >= options.required_memory:
                break

            print(
                f"{datetime.now()} GPU {gpu_id} Free: {free_memory:.1f} GB, "
                f"Required: {options.required_memory:.1f} GB. ",
                "Waiting for available memory...",
                flush=True,
            )
            time.sleep(60)
    finally:
        pynvml.nvmlShutdown()

    try:
        api_config = APIConfig(api_config_str)
    except Exception as err:
        print(f"[config parse error] {api_config_str} {str(err)}", flush=True)
        return

    option_to_class = {
        "paddle_only": APITestPaddleOnly,
        "paddle_cinn": APITestCINNVSDygraph,
        "accuracy": APITestAccuracy,
        "paddle_gpu_performance": APITestPaddleGPUPerformance,
        "torch_gpu_performance": APITestTorchGPUPerformance,
        "paddle_torch_gpu_performance": APITestPaddleTorchGPUPerformance,
        "accuracy_stable": APITestAccuracyStable,
    }
    test_class = next(
        (cls for opt, cls in option_to_class.items() if getattr(options, opt, False)),
        APITestAccuracy,  # default fallback
    )

    if options.accuracy:
        case = test_class(
            api_config,
            test_amp=options.test_amp,
            atol=options.atol,
            rtol=options.rtol,
            test_tol=options.test_tol,
        )
    else:
        case = test_class(api_config, test_amp=options.test_amp)
    try:
        case.test()
    except Exception as err:
        if "CUDA out of memory" in str(err) or "Out of memory error" in str(err):
            os._exit(99)
        if "CUDA error" in str(err) or "memory corruption" in str(err):
            os._exit(1)
        print(f"[test error] {api_config_str}: {err}", flush=True)
        raise
    finally:
        del test_class, api_config, case
        gc.collect()
        if not any(
            getattr(options, opt)
            for opt in (
                "paddle_gpu_performance",
                "torch_gpu_performance",
                "paddle_torch_gpu_performance",
            )
        ):
            torch.cuda.empty_cache()
            paddle.device.cuda.empty_cache()


def main():
    start_time = time.time()
    print(f"Main process id: {os.getpid()}")
    set_start_method("spawn")

    parser = argparse.ArgumentParser(description="API Test")
    parser.add_argument("--api_config_file", default="")
    parser.add_argument(
        "--api_config_file_pattern",
        default="",
        help="Pattern to match multiple config files (e.g., 'tester/api_config/api_config_support2torch_*.txt')",
    )
    parser.add_argument("--api_config", default="")
    parser.add_argument(
        "--paddle_only",
        type=parse_bool,
        default=False,
        help="test paddle api only to figure out whether the api is supported",
    )
    parser.add_argument(
        "--paddle_cinn",
        type=parse_bool,
        default=False,
        help="test paddle api in dynamic graph mode and cinn mode",
    )
    parser.add_argument(
        "--accuracy",
        type=parse_bool,
        default=False,
        help="test paddle api to corespoding torch api",
    )
    parser.add_argument(
        "--paddle_gpu_performance",
        type=parse_bool,
        default=False,
        help="test paddle api performance",
    )
    parser.add_argument(
        "--torch_gpu_performance",
        type=parse_bool,
        default=False,
        help="test torch api performance",
    )
    parser.add_argument(
        "--paddle_torch_gpu_performance",
        type=parse_bool,
        default=False,
        help="test paddle and torch api performance",
    )
    parser.add_argument(
        "--accuracy_stable",
        type=parse_bool,
        default=False,
        help="test paddle api to corespoding torch api steadily",
    )
    parser.add_argument(
        "--test_amp",
        type=parse_bool,
        default=False,
        help="Whether to test in auto mixed precision (AMP) mode",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="Number of GPUs to use, -1 to use all available",
    )
    parser.add_argument(
        "--num_workers_per_gpu",
        type=int,
        default=1,
        help="Number of workers per GPU, -1 to maximize based on memory",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help="Comma-separated list of GPU IDs to use (e.g., 0,1,2), -1 for all available",
    )
    parser.add_argument(
        "--required_memory",
        type=float,
        default=10.0,
        help="Required memory per worker in GB",
    )
    parser.add_argument(
        "--test_cpu",
        type=parse_bool,
        default=False,
        help="Whether to test CPU mode",
    )
    parser.add_argument("--use_cached_numpy", type=bool, default=False)
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Log directory",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for accuracy tests",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for accuracy tests",
    )
    parser.add_argument(
        "--test_tol",
        type=parse_bool,
        default=False,
        help="Whether to test tolerance range in accuracy",
    )
    options = parser.parse_args()
    print(f"Options: {vars(options)}", flush=True)

    mode = [
        options.accuracy,
        options.paddle_only,
        options.paddle_cinn,
        options.paddle_gpu_performance,
        options.torch_gpu_performance,
        options.paddle_torch_gpu_performance,
        options.accuracy_stable,
    ]
    if len([m for m in mode if m is True]) != 1:
        print(
            "Specify only one test mode:"
            "--accuracy,"
            "--paddle_only,"
            "--paddle_cinn,"
            "--paddle_gpu_performance,"
            "--torch_gpu_performance,"
            "--paddle_torch_gpu_performance"
            "--accuracy_stable"
            " to True.",
            flush=True,
        )
        return
    if options.test_tol and not options.accuracy:
        print(f"--test_tol takes effect when --accuracy is True.", flush=True)
    os.environ["USE_CACHED_NUMPY"] = str(options.use_cached_numpy)

    if options.log_dir:
        set_test_log_path(options.log_dir)

    if options.api_config:
        # Single config execution
        from tester import (APIConfig, APITestAccuracy, APITestAccuracyStable,
                            APITestCINNVSDygraph, APITestPaddleGPUPerformance,
                            APITestPaddleOnly,
                            APITestPaddleTorchGPUPerformance,
                            APITestTorchGPUPerformance)
        
        # set log_writer
        set_engineV2()

        options.api_config = options.api_config.strip()
        print(f"{datetime.now()} test begin: {options.api_config}", flush=True)
        try:
            api_config = APIConfig(options.api_config)
        except Exception as err:
            print(f"[config parse error] {options.api_config} {str(err)}", flush=True)
            return

        option_to_class = {
            "paddle_only": APITestPaddleOnly,
            "paddle_cinn": APITestCINNVSDygraph,
            "accuracy": APITestAccuracy,
            "paddle_gpu_performance": APITestPaddleGPUPerformance,
            "torch_gpu_performance": APITestTorchGPUPerformance,
            "paddle_torch_gpu_performance": APITestPaddleTorchGPUPerformance,
            "accuracy_stable": APITestAccuracyStable,
        }
        test_class = next(
            (
                cls
                for opt, cls in option_to_class.items()
                if getattr(options, opt, False)
            ),
            APITestAccuracy,  # default fallback
        )

        if options.accuracy:
            case = test_class(
                api_config,
                test_amp=options.test_amp,
                atol=options.atol,
                rtol=options.rtol,
                test_tol=options.test_tol,
            )
        else:
            case = test_class(api_config, test_amp=options.test_amp)
        try:
            case.test()
        except Exception as err:
            print(f"[test error] {options.api_config}: {err}", flush=True)
        finally:
            case.clear_tensor()
            del case
    elif options.api_config_file or options.api_config_file_pattern:
        # validate GPU options
        gpu_ids = validate_gpu_options(options)

        # get config files
        if options.api_config_file_pattern:
            import glob

            config_files = []
            patterns = options.api_config_file_pattern.split(",")
            for pattern in patterns:
                pattern = pattern.strip()
                config_files.extend(glob.glob(pattern))
            if not config_files:
                print(
                    f"No config files found: {options.api_config_file_pattern}",
                    flush=True,
                )
                return
            config_files.sort()
            print("Config files to be tested:", flush=True)
            for i, config_file in enumerate(config_files, 1):
                print(f"{i}. {config_file}", flush=True)
        else:
            if not os.path.exists(options.api_config_file):
                print(f"No config file found: {options.api_config_file}", flush=True)
                return
            config_files = [options.api_config_file]

        # when engineV2 was interrupted, resume from .tmp dir
        aggregate_logs()

        # read checkpoint
        finish_configs = read_log("checkpoint")
        print(len(finish_configs), "cases in checkpoint.", flush=True)

        api_config_count = 0
        api_configs = set()
        for config_file in config_files:
            try:
                with open(config_file, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    api_config_count += len(lines)
                    api_configs.update(lines)
            except Exception as e:
                print(f"Failed to read config file {config_file}: {e}", flush=True)
                return
        print(api_config_count, "cases in total.", flush=True)
        dup_case = api_config_count - len(api_configs)
        if dup_case > 0:
            print(dup_case, "cases are duplicates and removed.", flush=True)

        api_config_count = len(api_configs)
        api_configs = sorted(api_configs - finish_configs)
        all_case = len(api_configs)
        fail_case = 0
        finish_case = api_config_count - all_case
        if finish_case:
            print(finish_case, "cases already tested.", flush=True)
        print(all_case, "cases will be tested.", flush=True)
        del api_config_count, dup_case, finish_case

        # validate GPU memory
        available_gpus, max_workers_per_gpu = check_gpu_memory(
            gpu_ids, options.num_workers_per_gpu, options.required_memory
        )
        if not available_gpus:
            print(
                f"No GPUs with sufficient memory available. Current memory constraint is {options.required_memory} GB.",
                flush=True,
            )
            return

        total_workers = sum(max_workers_per_gpu.values())
        print(
            f"Using {len(available_gpus)} GPU(s) with max workers per GPU: {max_workers_per_gpu}. Total workers: {total_workers}.",
            flush=True,
        )

        if options.test_cpu:
            print(f"Using {cpu_count()} CPU(s) for paddle in CPU mode.", flush=True)

        # set log_writer
        set_engineV2()

        # initialize process pool
        manager = Manager()
        gpu_worker_list = manager.dict(
            {gpu_id: manager.list() for gpu_id in available_gpus}
        )
        lock = Lock()

        pool = ProcessPool(
            max_workers=total_workers,
            initializer=init_worker_gpu,
            initargs=[
                gpu_worker_list,
                lock,
                available_gpus,
                max_workers_per_gpu,
                options,
            ],
        )

        def cleanup_handler(*args):
            cleanup(pool)
            sys.exit(1)

        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)

        # batch test
        try:
            BATCH_SIZE = 20000
            i = 0
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
                        i += 1
                        print(f"[{i}/{all_case}] Testing {config}", flush=True)
                        future.result()
                        print(f"[info] Test case succeeded for {config}", flush=True)
                    except TimeoutError as err:
                        write_to_log("timeout", config)
                        print(
                            f"[error] Test case timed out for {config}: {err}",
                            flush=True,
                        )
                        fail_case += 1
                    except ProcessExpired as err:
                        if err.exitcode == 99:
                            write_to_log("oom", config)
                            print(
                                f"[error] CUDA out of memory for {config}",
                                flush=True,
                            )
                        else:
                            write_to_log("crash", config)
                            print(
                                f"[fatal] Worker crashed for {config}: {err}",
                                flush=True,
                            )
                        fail_case += 1
                    except Exception as err:
                        print(
                            f"[warn] Test case failed for {config}: {err}",
                            flush=True,
                        )
                aggregate_logs()
            print(f"{all_case} cases tested, {fail_case} failed.", flush=True)
            pool.close()
            pool.join()
        except Exception as e:
            print(f"Unexpected error: {e}", flush=True)
            cleanup(pool)
            total_time = time.time() - start_time
            print(f"Test time: {round(total_time/60, 3)} minutes.", flush=True)
        finally:
            log_counts = aggregate_logs(end=True)
            print_log_info(all_case, log_counts)
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Test time: {round(total_time/60, 3)} minutes.", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()

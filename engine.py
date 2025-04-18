import argparse
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from datetime import datetime
from multiprocessing import Queue as MPQueue
from queue import Queue

from tester import (APIConfig, APITestAccuracy, APITestCINNVSDygraph,
                    APITestPaddleOnly)
from tester.api_config.log_writer import *


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

def run_test_case(api_config_str, options):
    """Run a single test case for the given API configuration."""
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
        if "CUDA error" in str(err) or "memory corruption" in str(err) or "CUDA out of memory" in str(err):
            raise
        print(f"[test error] {api_config_str} {str(err)}", flush=True)
        return False
    finally:
        case.clear_tensor()
        del case
        del api_config
    return True

def run_tests_on_gpu(task_queue, options, gpu_id):
    """Run tests on a single GPU with dynamic task scheduling."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    thread_task_queue = Queue()
    lock = Lock()

    def worker():
        while True:
            with lock:
                if thread_task_queue.empty():
                    break
                config = thread_task_queue.get()
            run_test_case(config, options)
            thread_task_queue.task_done()

    while not task_queue.empty():
        try:
            config = task_queue.get_nowait()
            thread_task_queue.put(config)
        except:
            break

    with ThreadPoolExecutor(max_workers=options.num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(options.num_threads)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as err:
                if "CUDA error" in str(err) or "memory corruption" in str(err) or "CUDA out of memory" in str(err):
                    raise


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
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--num_threads',
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

        if options.num_gpus > 1:
            # Multi GPUs execution
            task_queue = MPQueue()
            for config in api_configs:
                task_queue.put(config)
            with ProcessPoolExecutor(max_workers=options.num_gpus) as executor:
                futures = [executor.submit(run_tests_on_gpu, task_queue, options, gpu_id) for gpu_id in range(options.num_gpus)]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as err:
                        if "CUDA error" in str(err) or "memory corruption" in str(err) or "CUDA out of memory" in str(err):
                            raise
        else:
            # Single GPU execution
            task_queue = MPQueue()
            for config in api_configs:
                task_queue.put(config)
            run_tests_on_gpu(task_queue, options, gpu_id=0)

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

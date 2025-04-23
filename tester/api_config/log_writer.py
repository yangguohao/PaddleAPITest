import os
from glob import glob
import shutil

# 日志文件路径
DIR_PATH = os.path.dirname(os.path.realpath(__file__))[
    0 : os.path.dirname(os.path.realpath(__file__)).index("PaddleAPITest") + 13
]
TEST_LOG_PATH = os.path.join(DIR_PATH, "tester/api_config/test_log")
TMP_LOG_PATH = os.path.join(TEST_LOG_PATH, ".tmp")
os.makedirs(TMP_LOG_PATH, exist_ok=True)

# 日志类型和对应的文件
LOG_PREFIXES = {
    "accuracy_error": "api_config_accuracy_error",
    "checkpoint": "checkpoint",
    "crash": "api_config_crash",
    "paddle_error": "api_config_paddle_error",
    "paddle_to_torch_failed": "api_config_paddle_to_torch_failed",
    "pass": "api_config_pass",
    "timeout": "api_config_timeout",
    "torch_error": "api_config_torch_error",
}


def get_log_file(log_type: str):
    """获取指定日志类型和PID对应的日志文件路径"""
    pid = os.getpid()
    prefix = LOG_PREFIXES.get(log_type)
    return os.path.join(TMP_LOG_PATH, f"{prefix}_{pid}.txt")


def write_to_log(log_type, line):
    """添加单条日志到当前进程的日志文件"""
    if log_type not in LOG_PREFIXES:
        raise ValueError(f"Invalid log type: {log_type}")
    line = line.strip()
    if not line:
        return
    file_path = get_log_file(log_type)
    try:
        with open(file_path, "a") as f:
            f.write(line + "\n")
    except Exception as err:
        print(f"Error writing to {file_path}: {err}", flush=True)


def read_log(log_type):
    """读取文件所有行，返回集合"""
    if log_type not in LOG_PREFIXES:
        raise ValueError(f"Invalid log type: {log_type}")
    file_path = LOG_PREFIXES[log_type]
    try:
        with open(file_path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()
    except Exception as err:
        print(f"Error reading {file_path}: {err}", flush=True)
        return set()


def aggregate_logs():
    """聚合所有相同类型的日志文件"""
    for prefix in LOG_PREFIXES.values():
        pattern = os.path.join(TMP_LOG_PATH, f"{prefix}_*.txt")
        log_files = glob(pattern)
        if not log_files:
            continue

        all_lines = set()
        for file_path in log_files:
            try:
                with open(file_path, "r") as f:
                    all_lines.update(line.strip() for line in f if line.strip())
            except Exception as err:
                print(f"Error reading {file_path}: {err}", flush=True)

        aggregated_file = os.path.join(TEST_LOG_PATH, f"{prefix}.txt")
        try:
            with open(aggregated_file, "w") as f:
                f.writelines(f"{line}\n" for line in sorted(all_lines))
        except Exception as err:
            print(f"Error writing to {aggregated_file}: {err}", flush=True)

    try:
        shutil.rmtree(TMP_LOG_PATH)
    except OSError:
        pass

import atexit
import os
import threading
from collections import defaultdict

from filelock import FileLock

# 日志文件路径
DIR_PATH = os.path.dirname(os.path.realpath(__file__))[
    0 : os.path.dirname(os.path.realpath(__file__)).index("PaddleAPITest") + 13
]
TEST_LOG_PATH = os.path.join(DIR_PATH, "tester/api_config/test_log")
os.makedirs(TEST_LOG_PATH, exist_ok=True)

# 日志类型和对应的文件
LOG_FILES = {
    "accuracy_error": os.path.join(TEST_LOG_PATH, "api_config_accuracy_error.txt"),
    "checkpoint": os.path.join(TEST_LOG_PATH, "checkpoint.txt"),
    "crash": os.path.join(TEST_LOG_PATH, "api_config_crash.txt"),
    "paddle_error": os.path.join(TEST_LOG_PATH, "api_config_paddle_error.txt"),
    "paddle_to_torch_failed": os.path.join(
        TEST_LOG_PATH, "api_config_paddle_to_torch_failed.txt"
    ),
    "pass": os.path.join(TEST_LOG_PATH, "api_config_pass.txt"),
    "timeout": os.path.join(TEST_LOG_PATH, "api_config_timeout.txt"),
    "torch_error": os.path.join(TEST_LOG_PATH, "api_config_torch_error.txt"),
}

# 日志缓存
_log_buffer = defaultdict(list)
_buffer_lock = defaultdict(lambda: threading.Lock())
_buffer_size_limit = 10


def _write_to_file(log_type, lines):
    """将缓存中的内容写入文件"""
    file_path = LOG_FILES[log_type]
    lock_path = f"{file_path}.lock"
    try:
        with FileLock(lock_path):
            with open(file_path, "a") as f:
                f.writelines(f"{line}\n" for line in lines)
    except Exception as err:
        print(f"Error writing to {file_path}: {err}")


def write_to_log(log_type, line):
    """添加单条日志到缓存，达到批量大小后写入文件"""
    if log_type not in LOG_FILES:
        raise ValueError(f"Invalid log type: {log_type}")
    line = line.strip()
    if not line:
        return
    if log_type == "crash" or log_type == "timeout":
        _write_to_file(log_type, [line])
    lines_to_write = []
    with _buffer_lock[log_type]:
        _log_buffer[log_type].append(line)
        if len(_log_buffer[log_type]) >= _buffer_size_limit:
            lines_to_write = _log_buffer[log_type]
            _log_buffer[log_type] = []
    if lines_to_write:
        _write_to_file(log_type, lines_to_write)


def read_log(log_type):
    """读取文件所有行，返回集合"""
    if log_type not in LOG_FILES:
        raise ValueError(f"Invalid log type: {log_type}")
    file_path = LOG_FILES[log_type]
    lock_path = f"{file_path}.lock"
    try:
        with FileLock(lock_path):
            with open(file_path, "r") as f:
                return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()
    except Exception as err:
        print(f"Error reading {file_path}: {err}")
        return set()


def flush_buffer():
    """在程序退出时写入所有剩余缓存的日志"""
    for log_type in LOG_FILES.keys():
        lines_to_write = []
        with _buffer_lock[log_type]:
            if _log_buffer[log_type]:
                lines_to_write = _log_buffer[log_type]
                _log_buffer[log_type] = []
        if lines_to_write:
            _write_to_file(log_type, lines_to_write)
        lock_path = f"{LOG_FILES[log_type]}.lock"
        try:
            os.remove(lock_path)
        except Exception:
            pass


atexit.register(flush_buffer)

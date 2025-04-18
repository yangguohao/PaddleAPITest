import os
import threading
from collections import defaultdict
from multiprocessing import Lock

# 日志文件路径
DIR_PATH = os.path.dirname(os.path.realpath(__file__))[
    0 : os.path.dirname(os.path.realpath(__file__)).index("PaddleAPITest") + 13
]
TEST_LOG_PATH = os.path.join(DIR_PATH, "tester/api_config/test_log")
os.makedirs(TEST_LOG_PATH, exist_ok=True)

# 日志类型和对应的文件
LOG_FILES = {
    "checkpoint": os.path.join(TEST_LOG_PATH, "checkpoint.txt"),
    "accuracy_error": os.path.join(TEST_LOG_PATH, "api_config_accuracy_error.txt"),
    "paddle_error": os.path.join(TEST_LOG_PATH, "api_config_paddle_error.txt"),
    "torch_to_paddle_failed": os.path.join(
        TEST_LOG_PATH, "api_config_paddle_to_torch_failed.txt"
    ),
    "pass": os.path.join(TEST_LOG_PATH, "api_config_pass.txt"),
    "torch_error": os.path.join(TEST_LOG_PATH, "api_config_torch_error.txt"),
    "not_support": os.path.join(
        DIR_PATH, "tester/api_config/api_config_merged_not_support.txt"
    ),
}
# 批量写入的条数阈值
_BATCH_SIZE = 10


# 进程安全的文件锁
_file_locks = {}
for log_type in LOG_FILES:
    _file_locks[log_type] = Lock()

# 线程安全的日志缓冲
_thread_local = threading.local()


def _get_log_buffer(log_type):
    if not hasattr(_thread_local, "log_buffer"):
        _thread_local.log_buffer = defaultdict(list)
        _thread_local.buffer_locks = defaultdict(threading.Lock)
    return _thread_local.log_buffer[log_type], _thread_local.buffer_locks[log_type]


def _write_lines(log_type, lines):
    """写入多行到指定文件"""
    file_path = LOG_FILES[log_type]
    try:
        with _file_locks[log_type]:
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
    lines_to_write = []
    log_buffer, buffer_lock = _get_log_buffer(log_type)
    with buffer_lock:
        log_buffer.append(line)
        if len(log_buffer) >= _BATCH_SIZE:
            lines_to_write = list(log_buffer)
            log_buffer[:] = []
    if lines_to_write:
        _write_lines(log_type, lines_to_write)


def read_log(log_type):
    """读取文件所有行，返回集合"""
    if log_type not in LOG_FILES:
        raise ValueError(f"Invalid log type: {log_type}")

    file_path = LOG_FILES[log_type]
    try:
        with _file_locks[log_type]:
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
        log_buffer, buffer_lock = _get_log_buffer(log_type)
        with buffer_lock:
            if log_buffer:
                lines_to_write = list(log_buffer)
                log_buffer[:] = []
        if lines_to_write:
            _write_lines(log_type, lines_to_write)

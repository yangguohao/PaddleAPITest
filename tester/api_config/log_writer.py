import atexit
import os
from collections import defaultdict
from threading import Lock

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
    "torch_to_paddle_faild": os.path.join(
        TEST_LOG_PATH, "api_config_paddle_to_torch_faild.txt"
    ),
    "pass": os.path.join(TEST_LOG_PATH, "api_config_pass.txt"),
    "torch_error": os.path.join(TEST_LOG_PATH, "api_config_torch_error.txt"),
    "not_support": os.path.join(DIR_PATH, "tester/api_config/api_config_merged_not_support.txt"),
}

# 线程安全的锁，每个文件一个
_locks = defaultdict(Lock)

# 批量写入的条数阈值
_BATCH_SIZE = 5

# 缓存待写入的日志条目，格式为 {log_type: [lines]}
_log_buffer = defaultdict(list)


def _write_lines(log_type, lines):
    """写入多行到指定文件（内部函数）"""
    if log_type not in LOG_FILES:
        raise ValueError(f"Invalid log type: {log_type}")

    file_path = LOG_FILES[log_type]
    with _locks[file_path]:
        with open(file_path, "a") as f:
            f.writelines(f"{line}\n" for line in lines)


def write_to_log(log_type, line):
    """添加单条日志到缓存，达到批量大小后写入文件"""
    if log_type not in LOG_FILES:
        raise ValueError(f"Invalid log type: {log_type}")

    _log_buffer[log_type].append(line)
    if len(_log_buffer[log_type]) >= _BATCH_SIZE:
        _write_lines(log_type, _log_buffer[log_type])
        _log_buffer[log_type].clear()


def read_log(log_type):
    """读取文件所有行，返回集合"""
    if log_type not in LOG_FILES:
        raise ValueError(f"Invalid log type: {log_type}")

    file_path = LOG_FILES[log_type]
    try:
        with _locks[file_path]:
            with open(file_path, "r") as f:
                return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()


def _flush_buffer():
    """在程序退出时写入所有剩余缓存的日志"""
    for log_type, lines in _log_buffer.items():
        if lines:
            _write_lines(log_type, lines)
            _log_buffer[log_type].clear()


# 注册退出时清空缓存
atexit.register(_flush_buffer)

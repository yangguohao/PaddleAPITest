import os
import shutil
from pathlib import Path
from tester import get_cfg

# 日志文件路径
DIR_PATH = Path(__file__).resolve()
while DIR_PATH.name != "PaddleAPITest":
    DIR_PATH = DIR_PATH.parent
TEST_LOG_PATH = DIR_PATH / "tester/api_config/test_log"
TEST_LOG_PATH.mkdir(parents=True, exist_ok=True)
TMP_LOG_PATH = TEST_LOG_PATH / ".tmp"

# 日志类型和对应的文件
LOG_PREFIXES = {
    "checkpoint": "checkpoint",
    "pass": "api_config_pass",
    "paddle_error": "api_config_paddle_error",
    "torch_error": "api_config_torch_error",
    "paddle_to_torch_failed": "api_config_paddle_to_torch_failed",
    "accuracy_error": "api_config_accuracy_error",
    "timeout": "api_config_timeout",
    "crash": "api_config_crash",
}

is_engineV2 = False


def set_engineV2():
    global is_engineV2
    is_engineV2 = True
    TMP_LOG_PATH.mkdir(exist_ok=True)


def get_log_file(log_type: str):
    """获取指定日志类型和PID对应的日志文件路径"""
    global is_engineV2
    prefix = LOG_PREFIXES.get(log_type)
    id = get_cfg().id
    if not is_engineV2:
        return TEST_LOG_PATH / f"{prefix + id}.txt"
    pid = os.getpid()
    return TMP_LOG_PATH / f"{prefix}_{pid}.txt"


def write_to_log(log_type, line):
    """添加单条日志到当前进程的日志文件"""
    if log_type not in LOG_PREFIXES:
        raise ValueError(f"Invalid log type: {log_type}")
    line = line.strip()
    if not line:
        return
    file_path = get_log_file(log_type)
    try:
        with file_path.open("a") as f:
            f.write(line + "\n")
    except Exception as err:
        print(f"Error writing to {file_path}: {err}", flush=True)


def read_log(log_type):
    """读取文件所有行，返回集合"""
    if log_type not in LOG_PREFIXES:
        raise ValueError(f"Invalid log type: {log_type}")
    id = get_cfg().id
    file_path = TEST_LOG_PATH / f"{LOG_PREFIXES[log_type] + id}.txt"
    try:
        with file_path.open("r") as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()
    except Exception as err:
        print(f"Error reading {file_path}: {err}", flush=True)
        return set()


def aggregate_logs(mkdir=False):
    """聚合所有相同类型的日志文件"""
    if not TMP_LOG_PATH.exists():
        if mkdir:
            TMP_LOG_PATH.mkdir(exist_ok=True)
        return
    for prefix in LOG_PREFIXES.values():
        log_files = list(TMP_LOG_PATH.glob(f"{prefix}_*.txt"))
        if not log_files:
            continue

        all_lines = set()
        for file_path in log_files:
            try:
                with file_path.open("r") as f:
                    all_lines.update(line.strip() for line in f if line.strip())
            except Exception as err:
                print(f"Error reading {file_path}: {err}", flush=True)

        aggregated_file = TEST_LOG_PATH / f"{prefix}.txt"
        try:
            with aggregated_file.open("a") as f:
                f.writelines(f"{line}\n" for line in sorted(all_lines))
        except Exception as err:
            print(f"Error writing to {aggregated_file}: {err}", flush=True)

    try:
        shutil.rmtree(TMP_LOG_PATH)
    except OSError:
        pass
    if mkdir:
        TMP_LOG_PATH.mkdir(exist_ok=True)

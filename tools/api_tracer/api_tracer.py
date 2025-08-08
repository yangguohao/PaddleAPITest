import os
import signal
import sys
from typing import Any, List, Set, Union

from config_serializer import ConfigSerializer
from framework_dialect import FrameworkDialect, TracingHook


class APITracer:

    _valid_kwargs: Set[str] = {
        "merge_output",
        "record_stack",
        "stack_format",
        "disable_torch_api_list",
    }

    def __init__(
        self,
        dialect: str,
        output_path: str = "trace_output",
        levels: Union[int, List] = 0,
        **kwargs: Any,
    ):
        """
        初始化 API 追踪器

        Args:
        - dialect (str): 指定抓取的框架方言, 例如 "torch"
        - output_path (str, optional): 输出文件的路径, 默认为 "trace_output"
        - levels (Union[int, List], optional): 抓取配置的粒度, 可以是单个整数或整数列表, 默认为 0

        Kwargs:
        - merge_output (bool, optional): 是否合并输出, 默认为 False
        - record_stack (bool, optional): 是否记录堆栈信息, 默认为 False
        - stack_format (str, optional): 堆栈信息的格式, 默认为 "short"
        - disable_torch_api_list (bool, optional): 是否禁用 Torch API 列表, 默认为 False
        """
        if invalid := set(kwargs) - self._valid_kwargs:
            raise ValueError(f"Invalid keyword arguments: {sorted(invalid)}")
        if "stack_format" in kwargs:
            record_stack = kwargs.get("record_stack", False)
            stack_format = kwargs.get("stack_format", "short")
            if record_stack and stack_format not in ["full", "short", "api"]:
                raise ValueError(
                    f"Invalid stack_format: {stack_format}. It should be one of ['full', 'short', 'api']"
                )

        os.makedirs(output_path, exist_ok=True)
        levels = levels if isinstance(levels, list) else [levels]

        self.dialect = FrameworkDialect.get_dialect(dialect)
        self.serializer = ConfigSerializer(self.dialect, output_path, levels, **kwargs)
        self.hooks: List[TracingHook] = self.dialect.get_hooks(
            self.serializer, levels, **kwargs
        )
        self._is_tracing = False

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(
            f"[APITracer] API tracer initialized for '{self.dialect.get_framework_name()}' "
            f"in levels {levels}, output path: {output_path}"
        )
        print(f"[APITracer] Kwargs: {kwargs}")

    def _signal_handler(self, signum, frame):
        print(f"[APITracer] Received signal {signum}, stopping trace...")
        self.stop()
        sys.exit(1)

    def start(self):
        """启动抓取"""
        if self._is_tracing:
            print("[APITracer] Tracing is already active.")
            return

        print("[APITracer] Starting API trace...")
        self.serializer.open()
        for hook in self.hooks:
            hook.install()
        self._is_tracing = True
        print("[APITracer] Tracing is now ACTIVE.")

    def stop(self):
        """停止抓取并恢复"""
        if not self._is_tracing:
            print("[APITracer] Tracing is not active.")
            return

        print("[APITracer] Stopping API trace...")
        for hook in reversed(self.hooks):
            hook.uninstall()
        self.serializer.close()
        self._is_tracing = False
        print("[APITracer] Tracing stopped and all APIs have been restored.")
        self.serializer.get_apis_and_configs()

    def __enter__(self):
        """进入上下文管理器"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """退出上下文管理器"""
        self.stop()

import os
import signal
import sys
from typing import List, Set, Union

from .config_serializer import ConfigSerializer
from .framework_dialect import FrameworkDialect, TracingHook


class APITracer:

    _valid_kwargs: Set[str] = {"disable_torch_api_list"}

    def __init__(
        self,
        dialect: str,
        output_path: str = "trace_output",
        levels: Union[int, List] = 0,
        merge_output: bool = False,
        **kwargs,
    ):
        os.makedirs(output_path, exist_ok=True)

        if invalid := set(kwargs) - self._valid_kwargs:
            raise ValueError(f"Invalid keyword arguments: {sorted(invalid)}")

        levels = levels if isinstance(levels, list) else [levels]

        self.dialect = FrameworkDialect.get_dialect(dialect)
        self.serializer = ConfigSerializer(
            self.dialect, output_path, levels, merge_output
        )
        self.hooks: List[TracingHook] = self.dialect.get_hooks(
            self.serializer, levels, **kwargs
        )
        self._is_tracing = False

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(
            f"[APITracer] API tracer initialized for '{self.dialect.get_framework_name()}' "
            f"in {'merged ' if merge_output else ''}levels {levels}. Output path: {output_path}."
        )

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

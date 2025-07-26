import os
from typing import List

from config_serializer import ConfigSerializer
from framework_dialect import FrameworkDialect, TracingHook


class APITracer:

    def __init__(self, dialect: str, output_path: str = "./trace_output"):
        os.makedirs(output_path, exist_ok=True)
        self.dialect = FrameworkDialect.get_dialect(dialect)
        self.serializer = ConfigSerializer(
            self.dialect, os.path.join(output_path, "api_trace.yaml")
        )
        self.hooks: List[TracingHook] = self.dialect.get_hooks(self.serializer)
        self._is_tracing = False

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

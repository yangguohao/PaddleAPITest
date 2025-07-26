import functools
import importlib
import os
import traceback
from typing import Any, Dict, Optional, Tuple

import yaml

from .config_serializer import ConfigSerializer
from .framework_dialect import FrameworkDialect


class APITracer:
    """API抓取器(猴子补丁)"""

    def __init__(self, dialect: str, output_path: str = "./trace_output"):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.dialect = FrameworkDialect.get_dialect(dialect)
        self.serializer = ConfigSerializer(
            self.dialect, output_path + "/api_trace.yaml"
        )
        self._original_apis: Dict[str, Any] = {}
        self._is_tracing = False

    def _get_api_parent_and_name(
        self, api_name: str
    ) -> Tuple[Optional[Any], Optional[str]]:
        """获取API的父模块和函数名"""
        parts = api_name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:i])
            try:
                module = importlib.import_module(module_name)
                parent_obj = module
                for attr in parts[i:-1]:
                    parent_obj = getattr(parent_obj, attr)
                func_name = parts[-1]
                return parent_obj, func_name
            except (ImportError, AttributeError):
                continue
        return None, None

    def _create_wrapper(self, api_name: str, original_api: Any):
        """为API创建包装器: 拦截调用, 记录信息, 执行原始调用"""

        @functools.wraps(original_api)
        def wrapper(*args, **kwargs):
            if not self._is_tracing:
                return original_api(*args, **kwargs)
            output = original_api(*args, **kwargs)
            try:
                self.serializer.dump_call(api_name, args, kwargs, output)
            except Exception as e:
                print(f"[APITracer] Error during serialization of '{api_name}': {e}")
            return output

        return wrapper

    def start(self):
        """启动抓取"""
        print("[APITracer] Starting API trace...")
        self.serializer.open()

        print("[APITracer] Discovering APIs using the provided dialect...")
        api_list = self.dialect.discover_apis() + self.dialect.discover_custom_ops()

        with open(os.path.join(self.output_path, "api_list.yaml"), "w") as f:
            yaml.dump(api_list, f)

        print(f"[APITracer] Attempting to patch {len(api_list)} APIs...")
        patched_apis = 0
        skipped_apis = 0
        for api_name in api_list:
            if api_name in self._original_apis:
                # print(f"[APITracer] Skipping {api_name}: Already patched")
                continue

            parent_obj, func_name = self._get_api_parent_and_name(api_name)
            if not parent_obj or not func_name or not hasattr(parent_obj, func_name):
                print(f"[APITracer] Skipping {api_name}: Could not resolve path.")
                continue

            try:
                original_api = getattr(parent_obj, func_name)
                wrapper = None

                if isinstance(original_api, property):
                    if original_api.fget and original_api.fset:
                        wrapped_getter = self._create_wrapper(
                            f"{api_name}.fget", original_api.fget
                        )
                        wrapper = property(
                            wrapped_getter,
                            original_api.fset,
                            original_api.fdel,
                            original_api.__doc__,
                        )
                elif isinstance(original_api, (classmethod, staticmethod)):
                    original_func = original_api.__func__
                    wrapped_func = self._create_wrapper(api_name, original_func)
                    wrapper = type(original_api)(wrapped_func)
                elif callable(original_api):
                    wrapper = self._create_wrapper(api_name, original_api)

                if wrapper:
                    self._original_apis[api_name] = original_api
                    setattr(parent_obj, func_name, wrapper)
                    patched_apis += 1
            except (TypeError, AttributeError) as e:
                error_msg = str(e).lower()
                if (
                    "immutable type" in error_msg
                    or "can't set attribute" in error_msg
                    or "read-only" in error_msg
                ):
                    # print(f"[APITracer] Skip non-writable API '{api_name}'.")
                    skipped_apis += 1
                else:
                    print(f"[APITracer] Could not patch {api_name}: {e}")
            except Exception as e:
                print(f"[APITracer] Could not patch {api_name}: {e}")
            finally:
                if api_name in self._original_apis:
                    del self._original_apis[api_name]

        print(
            f"[APITracer] Successfully patched {patched_apis} APIs. Skipped {skipped_apis} non-writable APIs."
        )

        with open(os.path.join(self.output_path, "api_list_wrap.yaml"), "w") as f:
            yaml.dump(list(self._original_apis.keys()), f)

        self._is_tracing = True
        print("[APITracer] Tracing is now ACTIVE.")

    def stop(self):
        """停止抓取并恢复"""
        print("[APITracer] Stopping API trace...")
        for api_name, original_api in self._original_apis.items():
            parent_obj, func_name = self._get_api_parent_and_name(api_name)
            if parent_obj and func_name:
                try:
                    setattr(parent_obj, func_name, original_api)
                except Exception as e:
                    print(f"[APITracer] Error restoring API '{api_name}': {e}")

        self._original_apis.clear()
        self.serializer.close()
        print("[APITracer] All patched APIs have been restored.")

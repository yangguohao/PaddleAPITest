import abc
import functools
import importlib
import inspect
import pkgutil
import traceback
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import torch
from torch.utils._python_dispatch import TorchDispatchMode

if TYPE_CHECKING:
    from config_serializer import ConfigSerializer


class TracingHook(abc.ABC):
    """钩子的抽象基类"""

    @abc.abstractmethod
    def install(self):
        pass

    @abc.abstractmethod
    def uninstall(self):
        pass


# UNUSED for torch
class SetattrHook(TracingHook):
    def __init__(self, dialect: "FrameworkDialect", serializer: "ConfigSerializer"):
        self.dialect = dialect
        self.serializer = serializer
        self._original_apis: Dict[str, Any] = {}
        self._module_cache: Dict[str, Any] = {}

    def _get_api_parent_and_name(
        self, api_name: str
    ) -> Tuple[Optional[Any], Optional[str]]:
        parts = api_name.split(".")
        if len(parts) < 2:
            return None, None

        for i in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:i])
            try:
                if module_name in self._module_cache:
                    module = self._module_cache[module_name]
                else:
                    module = importlib.import_module(module_name)
                    self._module_cache[module_name] = module

                parent_obj = module
                for attr in parts[i:-1]:
                    parent_obj = getattr(parent_obj, attr)
                return parent_obj, parts[-1]
            except (ImportError, AttributeError):
                continue
        return None, None

    def _create_wrapper(self, api_name: str, original_api: Any):

        @functools.wraps(original_api)
        def wrapper(*args, **kwargs):
            output = original_api(*args, **kwargs)
            self.serializer.dump_call(api_name, args, kwargs)
            return output

        return wrapper

    def install(self):
        api_list = self.dialect.discover_apis() + self.dialect.discover_custom_ops()

        # with open(os.path.join(os.path.dirname(__file__), "trace_output", "api_list.yaml"), "w") as f:
        #     yaml.dump(api_list, f)

        print(f"[SetattrHook] Attempting to patch {len(api_list)} APIs...")
        patched_count, skipped_count = 0, 0
        for api_name in api_list:
            if api_name in self._original_apis:
                # print(f"[SetattrHook] Skipping {api_name}: Already patched")
                continue

            parent_obj, func_name = self._get_api_parent_and_name(api_name)
            if not parent_obj or not func_name or not hasattr(parent_obj, func_name):
                print(f"[SetattrHook] Skipping {api_name}: Could not resolve path.")
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
                    setattr(parent_obj, func_name, wrapper)
                    self._original_apis[api_name] = original_api
                    patched_count += 1
            except (TypeError, AttributeError) as e:
                error_msg = str(e).lower()
                if (
                    "immutable type" in error_msg
                    or "can't set attribute" in error_msg
                    or "read-only" in error_msg
                ):
                    # print(f"[SetattrHook] Skip non-writable API '{api_name}'.")
                    skipped_count += 1
                else:
                    print(f"[SetattrHook] Could not patch {api_name}: {e}")
            except Exception as e:
                print(f"[SetattrHook] Could not patch {api_name}: {e}")

        print(
            f"[SetattrHook] Patched {patched_count} APIs. Skipped {skipped_count} non-writable APIs."
        )

        # with open(os.path.join(os.path.dirname(__file__), "trace_output", "api_list_wrap.yaml"), "w") as f:
        #     yaml.dump(list(self._original_apis.keys()), f)

    def uninstall(self):
        print(f"[SetattrHook] Restoring {len(self._original_apis)} patched APIs...")
        for api_name, original_api in self._original_apis.items():
            parent_obj, func_name = self._get_api_parent_and_name(api_name)
            if parent_obj and func_name:
                try:
                    setattr(parent_obj, func_name, original_api)
                except Exception as e:
                    print(f"[SetattrHook] Error restoring API '{api_name}': {e}")
        self._original_apis.clear()
        self._module_cache.clear()
        print("[SetattrHook] Restoration complete.")


# SetattrHook can not hook torch C API
class TorchFunctionModeTracer(torch.overrides.TorchFunctionMode):
    def __init__(self, serializer: "ConfigSerializer"):
        self.serializer = serializer
        self.ignored_functions = torch.overrides.get_ignored_functions()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in self.ignored_functions:
            return func(*args, **kwargs)

        output = func(*args, **kwargs)

        api_name = torch.overrides.resolve_name(func)
        if not api_name:
            if hasattr(func, "__module__"):
                api_name = f"{func.__module__}.{func.__name__}"
            elif hasattr(func, "__objclass__"):
                api_name = f"{func.__objclass__.__module__}.{func.__objclass__.__name__}.{func.__name__}"
            else:
                api_name = f"unknown.{func.__name__}"
                print(f"Unknown func: {func}, type: {type(func)}")

        self.serializer.dump_call(api_name, args, kwargs)
        return output


class TorchFunctionHook(TracingHook):
    def __init__(self, serializer: "ConfigSerializer"):
        self.tracing_mode = TorchFunctionModeTracer(serializer)

    def install(self):
        print(f"[TorchFunctionHook] Enabling __torch_function__ tracing mode...")
        self.tracing_mode.__enter__()
        print("[TorchFunctionHook] Mode enabled.")

    def uninstall(self):
        print("[TorchFunctionHook] Disabling __torch_function__ tracing mode...")
        self.tracing_mode.__exit__(None, None, None)
        print("[TorchFunctionHook] Mode disabled.")


class TorchDispatchModeTracer(TorchDispatchMode):
    def __init__(self, serializer: "ConfigSerializer"):
        self.serializer = serializer

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        output = func(*args, **kwargs)
        api_name = func.name()
        self.serializer.dump_call(api_name, args, kwargs)
        return output


class TorchDispatchHook(TracingHook):
    def __init__(self, serializer: "ConfigSerializer"):
        self.tracing_mode = TorchDispatchModeTracer(serializer)

    def install(self):
        print(f"[TorchDispatchHook] Enabling __torch_dispatch__ tracing mode...")
        self.tracing_mode.__enter__()
        print("[TorchDispatchHook] Mode enabled.")

    def uninstall(self):
        print("[TorchDispatchHook] Disabling __torch_dispatch__ tracing mode...")
        self.tracing_mode.__exit__(None, None, None)
        print("[TorchDispatchHook] Mode disabled.")


class FrameworkDialect(abc.ABC):
    """框架方言的抽象基类"""

    @abc.abstractmethod
    def get_framework_name(self) -> str:
        """返回框架名称"""
        raise NotImplementedError

    def discover_apis(self) -> List[str]:
        """返回框架API列表"""
        return []

    def discover_custom_ops(self) -> List[str]:
        """返回自定义算子API列表"""
        return []

    @abc.abstractmethod
    def serialize_special_type(self, item: Any) -> Optional[Dict]:
        """序列化框架所特有的数据类型"""
        raise NotImplementedError

    @abc.abstractmethod
    def format_special_type(self, item: Any) -> str:
        """格式化框架所特有的数据类型"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_hooks(
        self, serializer: "ConfigSerializer", level: Union[int, List]
    ) -> List[TracingHook]:
        """获取跟踪钩子, 用于在API调用时进行记录"""
        raise NotImplementedError

    @classmethod
    def get_dialect(cls, framework_name: str) -> "FrameworkDialect":
        dialect_map = {"torch": PyTorchDialect}
        dialect_class = dialect_map.get(framework_name)
        if not dialect_class:
            raise ValueError(f"Unsupported framework: {framework_name}")
        return dialect_class()


class PyTorchDialect(FrameworkDialect):
    """PyTorch方言实现"""

    MODULE_BLACKLIST = {
        "torch._decomp",
        "torch._dynamo",
        "torch._functorch",
        "torch._inductor",
        "torch._ops",
        "torch.ao",
        "torch.backends",
        "torch.compiler",
        "torch.contrib",
        "torch.distributed",
        "torch.fx",
        "torch.jit",
        "torch.onnx",
        "torch.testing",
        "torch.utils",
    }

    IGNORE_ATTRIBUTES = {
        "__bytes__",
        "__class__",
        "__delattr__",
        "__delete__",
        "__dict__",
        "__dir__",
        "__doc__",
        "__format__",
        "__get__",
        "__getattribute__",
        "__hash__",
        "__init__",
        "__init_subclass__",
        "__module__",
        "__new__",
        "__prepare__",
        "__repr__",
        "__set__",
        "__setattr__",
        "__sizeof__",
        "__slots__",
        "__str__",
        "__weakref__",
    }

    IGNORE_CLASSES = {
        "torch._utils.CallbackRegistry",
        "torch.cuda._gpu_trace.CallbackRegistry",
        "torch.cuda._sanitizer.StreamSynchronizations",
        "torch.cuda._sanitizer._TensorsAccessed",
        "torch.xpu._gpu_trace.CallbackRegistry",
    }

    def get_framework_name(self) -> str:
        return "torch"

    # UNUSED in TensorTracerMode
    def discover_apis(self) -> List[str]:
        """使用pkgutil遍历torch包"""
        print(
            f"[{self.__class__.__name__}] Discovering APIs for '{self.get_framework_name()}'..."
        )

        base_module = importlib.import_module(self.get_framework_name())
        api_set: Set[str] = set()
        modules = {base_module}

        if hasattr(base_module, "__path__"):
            for module_info in pkgutil.walk_packages(
                base_module.__path__, prefix=base_module.__name__ + "."
            ):
                if any(
                    module_info.name.startswith(pattern)
                    for pattern in self.MODULE_BLACKLIST
                ):
                    # print(
                    #     f"[Discovery Info] Skipping blacklisted module: {module_info.name}"
                    # )
                    continue
                try:

                    sub_module = importlib.import_module(module_info.name)
                    modules.add(sub_module)
                except Exception as e:
                    print(
                        f"[{self.__class__.__name__}] Could not import module {module_info.name}: {e}"
                    )
                    continue

        print(
            f"[{self.__class__.__name__}] Discovered {len(modules)} modules to inspect."
        )

        for module in modules:
            for member_name in dir(module):
                full_name = f"{module.__name__}.{member_name}"
                try:
                    obj = getattr(module, member_name)
                    if (
                        not hasattr(obj, "__module__")
                        or not obj.__module__
                        or not obj.__module__.startswith("torch")
                    ):
                        continue
                    if full_name in self.IGNORE_CLASSES:
                        continue
                    if callable(obj) and not inspect.isclass(obj):
                        api_set.add(full_name)
                    elif inspect.isclass(obj):
                        for cls_member_name, cls_member in inspect.getmembers(obj):
                            if cls_member_name in self.IGNORE_ATTRIBUTES:
                                continue
                            full_cls_name = f"{full_name}.{cls_member_name}"
                            if inspect.ismethod(cls_member) or inspect.isfunction(
                                cls_member
                            ):
                                api_set.add(full_cls_name)
                            elif isinstance(cls_member, (staticmethod, classmethod)):
                                api_set.add(full_cls_name)
                            elif isinstance(cls_member, property):
                                if cls_member.fget and cls_member.fset:
                                    api_set.add(full_cls_name)
                            elif isinstance(cls_member, partial):
                                if hasattr(
                                    cls_member.func, "__module__"
                                ) and cls_member.func.__module__.startswith("torch"):
                                    api_set.add(full_cls_name)
                            elif (
                                hasattr(cls_member, "__isabstractmethod__")
                                and cls_member.__isabstractmethod__
                            ):
                                api_set.add(full_cls_name)
                except Exception as e:
                    traceback.print_exc()
                    print(
                        f"[Discovery Warning] Could not inspect {module.__name__}.{member_name}: {e}"
                    )
                    continue

        api_list = sorted(list(api_set))
        print(f"[{self.__class__.__name__}] Discovered {len(api_list)} native APIs.")
        return api_list

    # UNUSED in TensorTracerMode
    def discover_custom_ops(self) -> List[str]:
        # TODO(@cangtianhuang): implemente me
        return []

    def serialize_special_type(self, item: Any) -> Optional[Dict]:
        if isinstance(item, torch.Tensor):
            return {
                "type": "torch.Tensor",
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "device": str(item.device),
            }
        if isinstance(item, torch.dtype):
            return {"type": "torch.dtype", "value": str(item)}
        if isinstance(item, torch.device):
            return {"type": "torch.device", "value": str(item)}
        if isinstance(item, torch.memory_format):
            return {"type": "torch.memory_format", "value": str(item)}
        # TODO(@cangtianhuang): add more serialization logic here
        return None

    def format_special_type(self, item: Dict) -> Optional[str]:
        if item["type"] == "torch.Tensor":
            return f'Tensor({item["shape"]}, "{item["dtype"].replace("torch.", "")}")'
        if item["type"] == "torch.dtype":
            return f'"{item["value"].replace("torch.", "")}"'
        if item["type"] == "torch.device":
            return f'"{item["value"]}"'
        if item["type"] == "torch.memory_format":
            return f'"{item["value"]}"'
        # TODO(@cangtianhuang): add more formatting logic here
        return None

    def get_hooks(self, serializer, level) -> List[TracingHook]:
        if level == 0:
            return [TorchFunctionHook(serializer)]
        if level == 1:
            return [TorchDispatchHook(serializer)]
        if level == -1:
            return [SetattrHook(self, serializer)]  # kept but not used
        if level == [0, 1]:
            return [
                TorchFunctionHook(serializer),
                TorchDispatchHook(serializer),
            ]
        raise ValueError(f"Invalid level: {level}")

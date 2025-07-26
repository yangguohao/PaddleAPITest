import abc
from functools import partial
import importlib
import inspect
import pkgutil
import traceback
from typing import Any, Dict, List, Optional, Set

import torch  # PyTorchDialect


class FrameworkDialect(abc.ABC):
    """框架方言的抽象基类"""

    @abc.abstractmethod
    def get_framework_name(self) -> str:
        """返回框架名称"""
        raise NotImplementedError

    @abc.abstractmethod
    def discover_apis(self) -> List[str]:
        """返回框架API列表"""
        raise NotImplementedError

    @abc.abstractmethod
    def discover_custom_ops(self) -> List[str]:
        """返回自定义算子API列表"""
        raise NotImplementedError

    @abc.abstractmethod
    def serialize_special_type(self, item: Any) -> Optional[Dict]:
        """序列化框架所特有的数据类型"""
        raise NotImplementedError

    @classmethod
    def get_dialect(cls, framework_name: str) -> "FrameworkDialect":
        framework_name_map = {
            "torch": PyTorchDialect,
        }
        if framework_name not in framework_name_map:
            raise ValueError(f"Unsupported framework: {framework_name}")
        return framework_name_map[framework_name]()


class PyTorchDialect(FrameworkDialect):
    """PyTorch方言实现"""

    MODULE_BLACKLIST = {
        "torch._dynamo.config",
        "torch._dynamo.test_case",
        "torch._dynamo.test_minifier_common",
        "torch._functorch.config",
        "torch._inductor.config",
        "torch._inductor.test_case",
        "torch.ao.pruning._experimental",
        "torch.backends._coreml.preprocess",
        "torch.compiler.config",
        "torch.contrib._tensorboard_vis",
        "torch.distributed._tools.sac_ilp",
        "torch.distributed.elastic.rendezvous",
        "torch.fx.experimental._config",
        "torch.onnx._internal.exporter",
        "torch.onnx._internal.fx",
        "torch.testing._internal",
        "torch.utils._cxx_pytree",
        "torch.utils.model_dump",
        "torch.utils.serialization.config",
        "torch.utils.tensorboard",
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

    def get_framework_name(self) -> str:
        return "torch"

    def discover_apis(self) -> List[str]:
        """使用pkgutil遍历torch包"""
        print(
            f"[{self.__class__.__name__}] Starting to discover APIs for '{self.get_framework_name()}'..."
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
                        f"[Discovery Warning] Could not import module {module_info.name}: {e}"
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

    def discover_custom_ops(self) -> List[str]:
        # TODO(@cangtianhuang): implemente me
        return []

    def serialize_special_type(self, item: Any) -> Optional[Dict]:
        if isinstance(item, torch.Tensor):
            return {
                "_type": "torch.Tensor",
                "shape": list(item.shape),
                "dtype": str(item.dtype).replace("torch.", ""),
                "device": str(item.device),
            }
        if isinstance(item, torch.dtype):
            return {"_type": "torch.dtype", "value": str(item).replace("torch.", "")}
        if isinstance(item, torch.device):
            return {"_type": "torch.device", "value": str(item)}
        if isinstance(item, torch.memory_format):
            return {"_type": "torch.memory_format", "value": str(item)}
        # TODO(@cangtianhuang): add more serialization logic here
        return None

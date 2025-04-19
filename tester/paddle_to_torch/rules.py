import re
import types
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ConvertResult:
    """Paddle2PyTorch 的转换结果数据类, 封装 API 转换结果，提供成功/失败的构造方法

    Attributes:
        paddle_api (str): Paddle API 名称
        is_supported (bool): 是否支持转换, 默认为 True
        code (Optional[List[str]]): 转换后的代码列表
        compiled_code (Optional[types.CodeType]): 预编译后的代码对象
        output_var (Optional[str]): 输出变量名，默认值 None 表示 result 保存最后的输出值
        error_message (Optional[str]): 错误信息, 仅当 is_supported = False 时有效

    Methods:
        success(paddle_api, code, output_var): 创建成功转换结果
        error(paddle_api, message): 创建失败转换结果
    """

    paddle_api: str
    is_supported: bool = True
    code: Optional[List[str]] = (
        None  # ["_tmp_0 = torch.add(x, y)", "_tmp_1 = torch.mul(_tmp_0, z)"]
    )
    compiled_code: Optional[types.CodeType] = field(default=None, repr=False)
    output_var: Optional[str] = None  # "_tmp_1"
    error_message: Optional[str] = None

    @staticmethod
    def success(
        paddle_api: str, code: List[str], output_var: Optional[str] = None
    ) -> "ConvertResult":
        return ConvertResult(paddle_api, code=code, output_var=output_var)

    @staticmethod
    def error(paddle_api: str, message: str) -> "ConvertResult":
        return ConvertResult(paddle_api, is_supported=False, error_message=message)


class BaseRule(ABC):
    """转换规则的抽象基类"""

    @abstractmethod
    def apply(self, paddle_api: str) -> ConvertResult:
        """
        将 Paddle API 调用转换为 PyTorch 等效代码形式
        code 中可包含输入变量的占位符(如 {input}、{x}), 这些变量将被自动填充为 torch tensor

        Args:
            paddle_api (str): Paddle API 名称

        Returns:
            ConvertResult: 包含代码和输出变量的 ConvertResult 对象, 或错误信息
        """
        pass

    def _format_args(self, args: List, kwargs: OrderedDict) -> str:
        """
        将参数格式化为调用字符串的辅助方法

        Args:
            args (List): 位置参数列表
            kwargs (Dict): 关键字参数字典

        Returns:
            str: 格式化后的调用字符串
        """
        PLACEHOLDER_PATTERN: re.Pattern = re.compile(r"\{([^{}]+)\}")

        def replacer(match):
            placeholder = match.group(1)
            if placeholder.isdigit():
                return f"_tmp_{placeholder}"
            elif placeholder.replace("_", "").isalnum():
                return placeholder
            return match.group(0)

        formatted_args = []
        for arg in args:
            if isinstance(arg, str):
                arg = PLACEHOLDER_PATTERN.sub(replacer, arg)
            formatted_args.append(str(arg))

        formatted_kwargs = OrderedDict()
        for key, value in kwargs.items():
            if isinstance(value, str):
                value = PLACEHOLDER_PATTERN.sub(replacer, value)
            formatted_kwargs[key] = str(value)

        args_str = ", ".join(formatted_args)
        kwargs_str = ", ".join(f"{k}={v}" for k, v in formatted_kwargs.items())
        return ", ".join(filter(None, [args_str, kwargs_str]))

    def read_mapping(self, mapping: Dict):
        """
        预处理，根据传入的 json 配置初始化成员变量

        Args:
            mapping (Dict): 包含 json 配置的字典

        Returns:
            None
        """
        if "Rule" in mapping:
            return
        self.direct_mapping: bool = not mapping.get("composite_steps")
        if self.direct_mapping:
            if "torch_api" not in mapping:
                raise ValueError("Missing required field 'torch_api' in the mapping.")
            self.torch_api: str = mapping["torch_api"]
            self.args_map: OrderedDict = mapping.get("paddle_torch_args_map", {})
            self.torch_args: List = mapping.get("torch_args", [])
            self.torch_kwargs: OrderedDict = mapping.get("torch_kwargs", OrderedDict())
        else:
            self.composite_steps: List = mapping.get("composite_steps", [])
            for step in self.composite_steps:
                if "torch_api" not in step:
                    raise ValueError(
                        f"Missing required field 'torch_api' in composite step: {step}"
                    )


class GenericRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        code = []
        if self.direct_mapping:  # 直接映射
            is_tensor_method = paddle_api.startswith("paddle.Tensor.")
            is_inplace = (
                paddle_api.endswith("_") and not paddle_api.endswith("__")
            ) or paddle_api == "paddle.Tensor.__setitem__"

            code.append("_kwargs = {}")
            if self.args_map:
                code.append("for paddle_param, torch_param in {")
                for paddle_param, torch_param in self.args_map.items():
                    code.append(f"    '{paddle_param}': '{torch_param}',")
                code.append("}.items():")
                code.append("    if paddle_param in locals():")
                code.append("        _kwargs[torch_param] = locals()[paddle_param]")

            if is_tensor_method:
                torch_method = self.torch_api.replace("torch.Tensor.", "")
                code.append("_tmp_tensor = args[0]")
                formatted_args = self._format_args(self.torch_args, self.torch_kwargs)
                if formatted_args:
                    code.append(
                        f"result = _tmp_tensor.{torch_method}({formatted_args}, **_kwargs)"
                    )
                else:
                    code.append(f"result = _tmp_tensor.{torch_method}(**_kwargs)")
                if is_inplace:
                    code.append("result = _tmp_tensor")
            else:
                formatted_args = self._format_args(self.torch_args, self.torch_kwargs)
                if formatted_args:
                    code.append(
                        f"result = {self.torch_api}({formatted_args}, **_kwargs)"
                    )
                else:
                    code.append(f"result = {self.torch_api}(**_kwargs)")
                if is_inplace:
                    code.append(
                        "result = args[0] if args else next(iter(kwargs.values()))"
                    )
            return ConvertResult.success(paddle_api, code)
        else:  # 简单组合映射
            for i, step in enumerate(self.composite_steps):
                torch_args = step.get("torch_args", [])
                torch_kwargs = step.get("torch_kwargs", OrderedDict())
                formatted_args = self._format_args(torch_args, torch_kwargs)
                code.append(f"_tmp_{i} = {step['torch_api']}({formatted_args})")
            return ConvertResult.success(
                paddle_api, code, f"_tmp_{len(self.composite_steps) - 1}"
            )


class ErrorRule(BaseRule):
    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def apply(self, paddle_api: str) -> ConvertResult:
        return ConvertResult.error(paddle_api, self.message)


# a


# b
class BroadcastTensorsRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        code = ["result = torch.broadcast_tensors(*input)"]
        return ConvertResult.success(paddle_api, code, "result")


# c


# d


# e


# f


# g


# h


# i


# j


# k


# l


# m


# n


# o


# p


# q


# r


# s


# t


# u


# v


# w


# x


# y


# z


__all__ = [  # type: ignore
    cls.__name__
    for cls in globals().values()
    if isinstance(cls, type) and issubclass(cls, BaseRule) and cls != BaseRule
]

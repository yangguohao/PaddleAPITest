import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ConvertResult:
    """Paddle2PyTorch 的转换结果数据类, 封装 API 转换结果，提供成功/失败的构造方法
    
    Attributes:
        paddle_api (str):
            Paddle API 名称

        is_supported (bool):
            是否支持转换, 默认为 True

        code (Optional[List[str]]):
            转换后的代码列表，支持显式/隐式赋值
            1. 显式赋值（用户定义变量名）:
                ["t = torch.add(x, y)", "result = torch.mul(t, z)"]
            2. 隐式赋值（自动生成中间变量）:
                ["torch.add(x, y)", "torch.mul(0, z)"]
                自动生成中间变量 _tmp_i (i 表示第 i 行的运行结果), 后续代码支持使用占位符表示(如 {0})
        
        output_var (Optional[str]):
            输出变量名，可以包含占位符 {n}
            示例: "{0}" 表示最终结果 result 保存 _tmp_0 变量值
            默认值 None 表示 result 保存最后的输出值
        
        error_message (Optional[str]):
            错误信息, 仅当 is_supported = False 时有效

    Methods:
        success(paddle_api, code, output_var): 创建成功转换结果
        error(paddle_api, message): 创建失败转换结果
    """
    paddle_api: str
    is_supported: bool = True

    code: Optional[List[str]] = None  # ["_tmp_0 = torch.add(x, y)", "_tmp_1 = torch.mul(_tmp_0, z)"]
    output_var: Optional[str] = None  # "_tmp_1"
    error_message: Optional[str] = None

    @staticmethod
    def success(paddle_api: str, code: List[str], output_var: Optional[str] = None) -> 'ConvertResult':
        return ConvertResult(paddle_api, code=code, output_var=output_var)

    @staticmethod
    def error(paddle_api: str, message: str) -> 'ConvertResult':
        return ConvertResult(paddle_api, is_supported=False, error_message=message)

class BaseRule(ABC):
    """转换规则的抽象基类"""
    def __init__(self):
        self.cached_results: Dict[str, ConvertResult] = {}

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
        PLACEHOLDER_PATTERN: re.Pattern = re.compile(r'\{([^{}]+)\}')

        def replace_placeholders(s: str) -> str:
            placeholders = re.findall(PLACEHOLDER_PATTERN, s)
            for placeholder in placeholders:
                if placeholder.isdigit():
                    s = s.replace(f"{{{placeholder}}}", f"_tmp_{placeholder}")
                elif placeholder.replace('_', '').isalnum():
                    s = s.replace(f"{{{placeholder}}}", placeholder)
            return s

        formatted_args = []
        for arg in args:
            if isinstance(arg, str):
                arg = replace_placeholders(arg)
            formatted_args.append(str(arg))

        formatted_kwargs = OrderedDict()
        for key, value in kwargs.items():
            if isinstance(value, str):
                value = replace_placeholders(value)
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
        self.direct_mapping: bool = "composite_steps" not in mapping or not mapping["composite_steps"]
        self.min_input_args: int = mapping.get("min_input_args", 0)
        if self.direct_mapping:
            if "torch_api" not in mapping:
                raise ValueError("Missing required field 'torch_api' in the mapping.")
            self.torch_api: str = mapping.get("torch_api", "")
            self.args_map: OrderedDict = mapping.get("paddle_torch_args_map", {})
            self.torch_args: List = mapping.get("torch_args", [])
            self.torch_kwargs: OrderedDict = mapping.get("torch_kwargs", OrderedDict())
        else:
            self.composite_steps: List = mapping.get("composite_steps", [])

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import List, Dict, Optional

@dataclass
class ConvertResult:
    """Paddle2PyTorch 的转换结果数据类, 封装 API 转换结果，提供成功/失败的构造方法
    
    Attributes:
        code (Optional[List[str]]):
            转换后的代码片段列表，支持显式/隐式赋值
            1. 显式赋值（用户定义变量名）:
                ["t = torch.add(x, y)", "result = torch.mul(t, z)"]
            2. 隐式赋值（自动生成中间变量）:
                ["torch.add(x, y)", "torch.mul({0}, z)"]
                自动生成中间变量 _tmp_i (i 表示第 i 行的运行结果), 后续代码支持使用占位符表示(如 {0})
        
        output_vars (Optional[Dict[str, str]]):
            输出变量映射字典, 格式为 {"用户变量": "内部变量"}
            示例: {"result": "_tmp_0"} 表示最终结果 result 保存_tmp_0 变量值
            默认值 None 表示使用 result 变量保存最后的输出值
        
        is_supported (bool):
            是否支持转换, 默认为 True
        
        error_message (Optional[str]):
            错误信息, 仅当 is_supported = False 时有效

    Methods:
        success(code, output_vars): 创建成功转换结果
        error(message): 创建失败转换结果
    """
    code: Optional[List[str]] = None  # ["torch.add(x, y)", "torch.mul({0}, z)"]
    output_vars: Optional[Dict[str, str]] = None  # {"result": "{1}"}

    is_supported: bool = True
    error_message: Optional[str] = None

    @staticmethod
    def success(code: List[str], output_vars: Optional[Dict[str, str]] = None) -> 'ConvertResult':
        return ConvertResult(code=code, output_vars=output_vars)

    @staticmethod
    def error(message: str) -> 'ConvertResult':
        return ConvertResult(is_supported=False, error_message=message)

class BaseRule(ABC):
    """转换规则的抽象基类"""
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, paddle_api: str, paddle_args: List, paddle_kwargs: Dict) -> ConvertResult:
        """
        将 Paddle API 调用转换为 PyTorch 等效代码形式
        code 中可包含中间变量占位符(如 {0}、{1}), 这些变量将在后续处理时被自动替换

        Args:
            paddle_api (str): Paddle API 名称
            paddle_args (List): Paddle API 调用的位置参数列表
            paddle_kwargs (Dict): Paddle API 调用的关键字参数字典

        Returns:
            ConvertResult: 包含代码和输出变量的 ConvertResult 对象, 或错误信息
        """
        pass

    def _format_args(self, args: List, kwargs: Dict) -> str:
        """
        将参数格式化为调用字符串的辅助方法

        Args:
            args (List): 位置参数列表
            kwargs (Dict): 关键字参数字典

        Returns:
            str: 格式化后的调用字符串
        """
        args_str = ", ".join(str(arg) for arg in args)
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"{args_str}{', ' if args_str and kwargs_str else ''}{kwargs_str}"
    
    def _preprocess(self, mapping: Dict):
        self.direct_mapping = "composite_steps" not in mapping or not mapping["composite_steps"]
        self.paddle_arg_list = mapping.get("paddle_arg_list", [])
        self.min_input_args = mapping.get("min_input_args", 0)
        if self.direct_mapping:
            self.torch_api = mapping.get("torch_api")
            self.args_map = mapping.get("paddle_torch_args_map", {})
            self.torch_args = mapping.get("torch_args", [])
            self.torch_kwargs = mapping.get("torch_kwargs", {})
        else:
            pass # decode


    def _postprocess(self, code: List[str], output_vars: Optional[Dict[str, str]]) -> tuple[List[str], Dict[str, str]]:
        """
        对代码块进行后处理, 分配中间变量, 替换占位符

        Args:
            code (List[str]): 需要处理的代码行列表
            output_vars (Optional[Dict[str, str]]): 输出变量名及其对应内部变量名的字典

        Returns:
            tuple[List[str], Dict[str, str]]: 处理后的代码行列表和输出变量字典

        Raises:
            ValueError: 若占位符未定义、输出变量字典的内部变量未找到，则抛出异常
        """
        if not code:
            return [], {}

        # Detects "var = expr" or "a, b = expr" or "(a, b) = expr"
        ASSIGNMENT_PATTERN: re.Pattern = re.compile(r'^\s*(?:\((.*?)\)|([^=]+?))\s*=\s*(.+)$')
        # Detects "{num}"
        PLACEHOLDER_PATTERN: re.Pattern = re.compile(r'\{(\d+)\}')

        final_code = []
        intermediate_vars = []

        # First pass: process all lines and collect intermediate variables
        for i, line in enumerate(code):
            line = line.strip()
            if not line:
                continue
            # Check for explicit assignment
            if match := ASSIGNMENT_PATTERN.match(line):
                vars_part = match.group(1) or match.group(2)
                vars_in_line = [v.strip() for v in re.split(r'\s*,\s*', vars_part) if v.strip()]
                intermediate_vars.extend(vars_in_line)
                final_code.append(line)
            else:
                var_name = f"_tmp_{i}"
                intermediate_vars.append(var_name)
                final_code.append(f"{var_name} = {line}")

        # Second pass: replace placeholders with actual variable names
        for i, line in enumerate(final_code):
            try:
                placeholders = re.findall(r'\{(\d+)\}', line)
                for placeholder in placeholders:
                    idx = int(placeholder)
                    var_name = f"_tmp_{idx}"
                    if var_name not in intermediate_vars:
                        raise ValueError(f"Placeholder {{{idx}}} is not defined")
                    line = line.replace(f"{{{idx}}}", f"_tmp_{idx}")
                final_code[i] = line
            except ValueError as e:
                raise ValueError(f"Error processing line {i+1}: {str(e)}") from e

        # Use provided output_vars or default to last variable as "result"
        final_output_vars = {}
        if output_vars is not None:
            for var_name in output_vars.values():
                if match := re.fullmatch(r'\{(\d+)\}', var_name):
                    idx = match.group(1)
                    var_name = f"_tmp_{idx}"
                if var_name not in intermediate_vars:
                    raise ValueError(f"Output variable '{var_name}' not found in intermediate variables")
                final_output_vars[var_name] = var_name
        else:
            if not final_code:
                final_output_vars["result"] = None
            else:
                last_line = final_code[-1]
                if match := ASSIGNMENT_PATTERN.match(last_line):
                    vars_part = match.group(1) or match.group(2)
                    last_line_vars = [v.strip() for v in re.split(r'\s*,\s*', vars_part) if v.strip()]
                else:
                    last_line_vars = [intermediate_vars[-1]]

                if len(last_line_vars) == 1:
                    final_output_vars["result"] = last_line_vars[0]
                else:
                    final_output_vars["result"] = f"({', '.join(last_line_vars)})"

        return final_code, final_output_vars
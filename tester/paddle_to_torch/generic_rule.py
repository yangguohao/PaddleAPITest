from typing import Dict, List
from .base_rule import ConvertResult, BaseRule

class GenericRule(BaseRule):
    def __init__(self, mapping: Dict):
        super().__init__()
        self.torch_api = mapping.get("torch_api")
        self.args_map = mapping.get("paddle_torch_args_map", {})
        self.min_input_args = mapping.get("min_input_args", 0)
        self.composite_steps = mapping.get("composite_steps", []) # 支持简单组合映射

    def apply(self, paddle_api: str, paddle_args: List, paddle_kwargs: Dict) -> ConvertResult:
        if len(paddle_args) + len(paddle_kwargs) < self.min_input_args:
            return ConvertResult.error(f"{paddle_api}至少需要{self.min_input_args}个参数")
        if len(paddle_args) > len(self.args_map):
            return ConvertResult.error(f"{paddle_api}的参数数量超过了映射规则的最大值")

        if not self.composite_steps:  # 直接映射
            torch_args = []
            torch_kwargs = {
                self.args_map[paddle_param]: value
                for paddle_param, value in zip(
                    list(self.args_map.keys())[:len(paddle_args)], 
                    paddle_args
                )
            }
            for paddle_param, value in paddle_kwargs.items():
                if paddle_param not in self.args_map:
                    return ConvertResult.error(f"{paddle_api}的参数{paddle_param}没有对应的参数映射")
                torch_param = self.args_map.get(paddle_param)
                torch_kwargs[torch_param] = value

            code = [f"{self.torch_api}({self._format_args(torch_args, torch_kwargs)})"]
            return ConvertResult.success(code)

        else:  # 简单组合映射
            code = []
            for step in self.composite_steps:
                call = f"{step["api"]}({self._format_args(step.get("args", []), step.get("kwargs", {}))})"
                code.append(call)
            return ConvertResult.success(code)


class ErrorRule(BaseRule):
    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def apply(self, paddle_api: str, paddle_args: List, paddle_kwargs: Dict) -> ConvertResult:
        return ConvertResult.error(self.message)

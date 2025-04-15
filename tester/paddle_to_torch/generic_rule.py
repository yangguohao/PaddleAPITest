from collections import OrderedDict

from .base_rule import BaseRule, ConvertResult


class GenericRule(BaseRule):
    def __init__(self):
        super().__init__()

    def apply(self, paddle_api: str) -> ConvertResult:
        code = []
        if self.direct_mapping:  # 直接映射
            kwargs_code = ["_kwargs = {}"]
            for paddle_param, torch_param in self.args_map.items():
                kwargs_code.append(f"if '{paddle_param}' in locals():")
                kwargs_code.append(f"    _kwargs['{torch_param}'] = {paddle_param}")
            code.extend(kwargs_code)
            if formatted_args := self._format_args(self.torch_args, self.torch_kwargs):
                code.append(f"_tmp_0 = {self.torch_api}({formatted_args}, **_kwargs)")
            else:
                code.append(f"_tmp_0 = {self.torch_api}(**_kwargs)")
            return ConvertResult.success(paddle_api, code, "_tmp_0")
        else:  # 简单组合映射
            for i, step in enumerate(self.composite_steps):
                torch_args = step.get("torch_args", [])
                torch_kwargs = step.get("torch_kwargs", OrderedDict())
                kwargs_code = [f"_kwargs_{i} = {{}}"]
                for paddle_param, torch_param in torch_kwargs.items():
                    kwargs_code.append(
                        f"if '{paddle_param}' in locals():"
                    )
                    kwargs_code.append(f"    _kwargs_{i}['{torch_param}'] = {paddle_param}")

                if formatted_args := self._format_args(torch_args, torch_kwargs):
                    code.append(f"_tmp_{i} = {step["api"]}({formatted_args}, **kwargs_{i})")
                else:
                    code.append(f"_tmp_{i} = {step['api']}(**kwargs_{i})")
            return ConvertResult.success(paddle_api, code, f"_tmp_{len(self.composite_steps) - 1}")


class ErrorRule(BaseRule):
    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def apply(self, paddle_api: str) -> ConvertResult:
        return ConvertResult.error(paddle_api, self.message)

from collections import OrderedDict

from .base_rule import BaseRule, ConvertResult


class GenericRule(BaseRule):
    def __init__(self):
        super().__init__()

    def apply(self, paddle_api: str) -> ConvertResult:
        code = []
        if self.direct_mapping:  # 直接映射
            is_tensor_method = paddle_api.startswith("paddle.Tensor.")
            is_inplace = (
                paddle_api.endswith("_") and not paddle_api.endswith("__")
            ) or paddle_api == "paddle.Tensor.__setitem__"

            kwargs_code = ["_kwargs = {}"]
            for paddle_param, torch_param in self.args_map.items():
                kwargs_code.append(f"if '{paddle_param}' in locals():")
                kwargs_code.append(f"    _kwargs['{torch_param}'] = {paddle_param}")
            code.extend(kwargs_code)

            if is_tensor_method:
                torch_method = self.torch_api.replace("torch.Tensor.", "")
                code.append(f"_tmp_tensor = args[0]")
                if formatted_args := self._format_args(
                    self.torch_args, self.torch_kwargs
                ):
                    code.append(
                        f"result = _tmp_tensor.{torch_method}({formatted_args}, **_kwargs)"
                    )
                else:
                    code.append(f"result = _tmp_tensor.{torch_method}(**_kwargs)")
                if is_inplace:
                    code.append(f"result = _tmp_tensor")
            else:
                if formatted_args := self._format_args(
                    self.torch_args, self.torch_kwargs
                ):
                    code.append(
                        f"result = {self.torch_api}({formatted_args}, **_kwargs)"
                    )
                else:
                    code.append(f"result = {self.torch_api}(**_kwargs)")
                if is_inplace:
                    code.append(
                        f"result = args[0] if args else next(iter(kwargs.values()))"
                    )
            return ConvertResult.success(paddle_api, code)
        else:  # 简单组合映射
            for i, step in enumerate(self.composite_steps):
                torch_args = step.get("torch_args", [])
                torch_kwargs = step.get("torch_kwargs", OrderedDict())
                code.append(
                    f"_tmp_{i} = {step["torch_api"]}({self._format_args(torch_args, torch_kwargs)})"
                )
            return ConvertResult.success(
                paddle_api, code, f"_tmp_{len(self.composite_steps) - 1}"
            )


class ErrorRule(BaseRule):
    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def apply(self, paddle_api: str) -> ConvertResult:
        return ConvertResult.error(paddle_api, self.message)

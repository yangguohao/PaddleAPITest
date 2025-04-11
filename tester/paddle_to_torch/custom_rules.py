from typing import Dict, List
from .base_rule import ConvertResult, BaseRule

__all__ = [
    cls.__name__ 
    for cls in globals().values() 
    if isinstance(cls, type) and 
       issubclass(cls, BaseRule) and 
       cls != BaseRule
]

# a

# b
class BroadcastTensorsRule(BaseRule):
    def __init__(self, paddle_api: str):
        super().__init__(paddle_api)
    
    def apply(self, paddle_args: List, paddle_kwargs: Dict) -> ConvertResult:
        if len(paddle_args) != 1:
            return ConvertResult.error(
                f"paddle.broadcast_tensors 需要 1 个位置参数（输入张量列表），但传入了 {len(paddle_args)} 个"
            )
        input_tensors = paddle_args[0]
        if not isinstance(input_tensors, (list, tuple)):
            return ConvertResult.error(
                f"paddle.broadcast_tensors 的输入必须是 list 或 tuple，但传入了 {type(input_tensors)}"
            )
        tensor_args_str = ", ".join([str(tensor) for tensor in input_tensors])
        torch_code = f"torch.broadcast_tensors({tensor_args_str})"
        return ConvertResult.success(code=[torch_code])

# c

# d

# e
class ErrorRule(BaseRule):
    def __init__(self, paddle_api: str, message: str):
        super().__init__(paddle_api)
        self.message = message

    def apply(self, paddle_args: List, paddle_kwargs: Dict) -> ConvertResult:
        return ConvertResult.error(self.message)
    
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

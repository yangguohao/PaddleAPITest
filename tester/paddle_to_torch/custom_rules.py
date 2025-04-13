from typing import Dict, List
from .base_rule import ConvertResult, BaseRule

__all__ = [  # type: ignore
    cls.__name__ 
    for cls in globals().values() 
    if isinstance(cls, type) and 
       issubclass(cls, BaseRule) and 
       cls != BaseRule
]

# a

# b
class BroadcastTensorsRule(BaseRule):
    def __init__(self):
        super().__init__()
    
    def apply(self, paddle_api: str, paddle_args: List, paddle_kwargs: Dict) -> ConvertResult:
        input_tensors = paddle_args[0]
        if not isinstance(input_tensors, (list, tuple)):
            return ConvertResult.error(
                f"paddle.broadcast_tensors 的输入必须是 list 或 tuple, 但传入了 {type(input_tensors)}"
            )
        tensor_args_str = ", ".join([str(tensor) for tensor in input_tensors])
        torch_code = f"torch.broadcast_tensors({tensor_args_str})"
        return ConvertResult.success(code=[torch_code])

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

from .base_rule import BaseRule, ConvertResult

# a

# b
class BroadcastTensorsRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        code = [
            "result = torch.broadcast_tensors(*input)"
        ]
        return ConvertResult.success(paddle_api, code)

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
    if isinstance(cls, type) and 
       issubclass(cls, BaseRule) and 
       cls != BaseRule
]

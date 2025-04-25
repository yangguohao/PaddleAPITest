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

    def _format_arg(self, arg) -> str:
        """
        将参数格式化为调用字符串的辅助方法

        Args:
            arg: 待格式化的参数

        Returns:
            str: 格式化后的参数
        """
        PLACEHOLDER_PATTERN: re.Pattern = re.compile(r"\{([^{}]+)\}")

        def replacer(match):
            placeholder = match.group(1)
            if placeholder.isdigit():
                return f"_tmp_{placeholder}"
            elif placeholder.replace("_", "").isalnum():
                return placeholder
            return match.group(0)

        if isinstance(arg, str):
            arg = PLACEHOLDER_PATTERN.sub(replacer, arg)
        return str(arg)

    def read_mapping(self, mapping: Dict):
        """
        预处理，根据传入的 json 配置初始化成员变量

        Args:
            mapping (Dict): 包含 json 配置的字典

        Returns:
            None
        """
        self.mapping: Dict = mapping
        if "Rule" in mapping:
            if "torch_api" in mapping:
                self.torch_api: str = mapping["torch_api"]
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
            if is_tensor_method:
                code.append("_tmp_tensor = next(iter(kwargs.values()))")
            is_inplace = (
                paddle_api.endswith("_") and not paddle_api.endswith("__")
            ) or paddle_api == "paddle.Tensor.__setitem__"

            code.append("_args = []")
            if self.torch_args:
                for arg in self.torch_args:
                    code.append(f"_args.extend([{self._format_arg(arg)}])")
            code.append("_kwargs = {}")
            if self.torch_kwargs:
                for key, value in self.torch_kwargs.items():
                    code.append(f"_kwargs['{key}'] = {self._format_arg(value)}")
            if self.args_map:
                code.append("for paddle_param, torch_param in {")
                for paddle_param, torch_param in self.args_map.items():
                    code.append(f"    '{paddle_param}': '{torch_param}',")
                code.append("}.items():")
                code.append("    if paddle_param in locals():")
                code.append("        _kwargs[torch_param] = locals()[paddle_param]")

            if is_tensor_method:
                torch_method = self.torch_api.replace("torch.Tensor.", "")
                if is_inplace:
                    code.append(f"_tmp_tensor.{torch_method}(*_args, **_kwargs)")
                    code.append("result = _tmp_tensor")
                else:
                    code.append(
                        f"result = _tmp_tensor.{torch_method}(*_args, **_kwargs)"
                    )
            else:
                if is_inplace:
                    code.append(f"{self.torch_api}(*_args, **_kwargs)")
                    code.append("result = next(iter(kwargs.values()))")
                else:
                    code.append(f"result = {self.torch_api}(*_args, **_kwargs)")
            return ConvertResult.success(paddle_api, code)
        else:  # 简单组合映射
            for i, step in enumerate(self.composite_steps):
                code.append(f"_args_{i} = []")
                for arg in step.get("torch_args", []):
                    code.append(f"_args_{i}.extend([{self._format_arg(arg)}])")
                code.append(f"_kwargs_{i} = {{}}")
                for key, value in step.get("torch_kwargs", {}).items():
                    code.append(f"_kwargs_{i}['{key}'] = {self._format_arg(value)}")
                code.append(
                    f"_tmp_{i} = {step['torch_api']}(*_args_{i}, **_kwargs_{i})"
                )
            code.append(f"result = _tmp_{len(self.composite_steps) - 1}")
            return ConvertResult.success(paddle_api, code)


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
class CropRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
ndim = x.dim()
offsets = locals().get('offsets')
shape = locals().get('shape')
if offsets is None:
    offsets = [0] * ndim
elif isinstance(offsets, (list, tuple)):
    offsets = [o.item() if isinstance(o, torch.Tensor) else int(o) for o in offsets]
elif isinstance(offsets, torch.Tensor):
    offsets = offsets.tolist()
if shape is None:
    shape = [x.size(i) - offsets[i] for i in range(ndim)]
elif isinstance(shape, (list, tuple)):
    shape = [s.item() if isinstance(s, torch.Tensor) else int(s) for s in shape]
elif isinstance(shape, torch.Tensor):
    shape = shape.tolist()
shape = [x.size(i) - offsets[i] if s == -1 else s for i, s in enumerate(shape)]
slices = [slice(offsets[i], offsets[i] + shape[i]) for i in range(ndim)]
result = x[slices]
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code, "result")


class CumRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        torch_api = paddle_api.replace("paddle.", "torch.")
        impl = f"""
axis = locals().get('axis')
if axis is None:
    x = x.flatten()
    axis = 0
dtype = locals().get('dtype', 'int64')
if dtype is not None:
    dtype = getattr(torch, dtype)
result = {torch_api}(input=x, dim=axis)
result.values.to(dtype)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class CumprodRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = f"""
dim = locals().get('dim')
if dim is None:
    x = x.flatten()
    axis = 0
dtype = locals().get('dtype')
if dtype is not None:
    dtype = getattr(torch, dtype)
result = torch.cumprod(input=x, dim=dim, dtype=dtype)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


# d


# e
class EmptyRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
if isinstance(shape, torch.Tensor):
    size_list = shape.tolist()
elif isinstance(shape, (list, tuple)):
    size_list = []
    for s in shape:
        if isinstance(s, torch.Tensor):
            size_list.append(s.item())
        else:
            size_list.append(s)
result = torch.empty(*size_list)     
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)

class ExpandRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
result = x.expand(*shape)  
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)

class ExpandasRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
result = x.expand_as(y)  
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


# f


# g
# class GetItemRule(BaseRule):
#     def apply(self, paddle_api: str) -> ConvertResult:
#         generic_rule = GenericRule()
#         generic_rule.read_mapping(self.mapping)
#         result = generic_rule.apply(paddle_api)
#         if result.is_supported and result.code:
#             result.code.append("result = result.item()")
#         return result
class GatherRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        #抽取对应维度的tensor直接进行stack操作
        impl = """
x = locals().get('x')
index = locals().get('index')
axis = locals().get('axis', 0)
if len(index.shape) == 0:
    result = torch.squeeze(torch.narrow(x, axis, index, 1),axis)
else:
    ans = []
    for i in index:
        temp = torch.narrow(x, axis, i.reshape([]), 1)
        ans.append(torch.squeeze(temp, axis))
    result = torch.stack(ans,axis)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)

class Gather_ndRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
class _func:
    def func(self,x,index):
        if index.dim() == 1:
            temp = x
            for i in range(index.numel()):
                temp = torch.narrow(temp, 0, index[i].reshape([]), 1)
                temp = torch.squeeze(temp, 0)
            return temp
        ans = []
        for i in index:
            ans.append(self.func(x, i))
        return torch.stack(ans, 0)
f = _func()
x = locals().get('x')
index = locals().get('index')
result = f.func(x,index)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)

class Gather_treeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
parents = locals().get('parents')
ids = locals().get('ids')
result = torch.empty(ids.shape)
max_time = ids.shape[0]
batch_size = ids.shape[1]
beam_size = ids.shape[2]
for batch in range(batch_size):
    for beam in range(beam_size):
        result[max_time-1,batch,beam] = ids[max_time-1,batch,beam]
        pa = parents[max_time-1,batch,beam]
        for step in range(max_time-2,-1,-1):
            result[step,batch,beam] = ids[step,batch,pa]
            pa = parents[step,batch,pa]
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)        

# h


# i
class IndexSelectRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
_kwargs = {}
for paddle_param, torch_param in {
    'x': 'input',
    'index': 'index',
    'axis': 'dim'
}.items():
    if paddle_param in locals():
        _kwargs[torch_param] = locals()[paddle_param]
    else:
        _kwargs[torch_param] = 0
_kwargs['index'] = torch.squeeze(_kwargs['index'])
result = torch.index_select( **_kwargs)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code, "result")

# j


# k


# l


# m


# n


# o


# p
class Put_along_axisRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
input = locals().get('arr')
dim = locals().get('axis')
index = locals().get('indices')
src = locals().get('values')
reduce = locals().get('reduce', 'assign')
if reduce == 'add':
    reduce = 'sum'
if reduce == 'multiply':
    reduce = 'prod'
include_self = locals().get('include_self', True)
broadcast = locals().get('broadcast', True)
flag = True
if broadcast == True:
    for i in range(len(input.shape)):
        if input.shape[i] < indices.shape[i]:
            flag = False
    if flag == False:
        src, index= torch.broadcast_tensor(src, index)
    else:
        broadcast_shape_list = list(input.shape)
        broadcast_shape_list[dim] = list(index.shape)[dim]
        broadcast_shape = tuple(broadcast_shape_list)
        index = torch.broadcast_to(index, broadcast_shape)
        src = torch.broadcast_to(src, broadcast_shape)
if reduce == 'assign':
    result = torch.scatter(input, dim, index, src)
else:
    result = torch.scatter_reduce(input, dim, index, src, reduce, include_self=include_self)  
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code, "result")
# q


# r
class Roi_aignRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
import torchvision
_kwargs = {}
for paddle_param, torch_param in {
    'x': 'input',
    'output_size': 'output_size',
    'spatial_scale': 'spatial_scale',
    'sampling_ratio': 'sampling_ratio',
    'aligned': 'aligned'
}.items():
    if paddle_param in locals():
        _kwargs[torch_param] = locals()[paddle_param]
boxes = locals().get('boxes')
boxnum = locals().get('boxes_num')
ans = []
end = 0
for i in range(boxnum.shape[0]):
    begin = end
    end = end + int(boxnum[i])
    ans.append(boxes[begin:end,])
result = torchvision.ops.roi_align( **_kwargs, boxes = ans)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code, "result")
    
class Roi_poolRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
import torchvision
_kwargs = {}
for paddle_param, torch_param in {
    'x': 'input',
    'output_size': 'output_size',
    'spatial_scale': 'spatial_scale'
}.items():
    if paddle_param in locals():
        _kwargs[torch_param] = locals()[paddle_param]
    else:
        _kwargs[torch_param] = 1.0
boxes = locals().get('boxes')
boxnum = locals().get('boxes_num')
ans = []
end = 0
for i in range(boxnum.shape[0]):
    begin = end
    end = end + int(boxnum[i])
    ans.append(boxes[begin:end,])
"""
        code = impl.splitlines()
        code.append(f"result = {self.torch_api}(boxes = ans, **_kwargs)")
        return ConvertResult.success(paddle_api, code, "result")

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

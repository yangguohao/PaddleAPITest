import re
import types
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class Code:
    """Paddle2PyTorch 转换代码数据类，封装转换后的可执行代码，自动预编译

    Attributes:
        preprocess: 预处理代码，在核心逻辑前执行
        core: 核心逻辑代码，应包含 Torch API
        postprocess: 后处理代码，在核心逻辑后执行
        preprocess_compiled: 预编译的预处理代码
        core_compiled: 预编译的核心逻辑代码
        postprocess_compiled: 预编译的后处理代码
    """

    preprocess: List[str] = field(default_factory=list)
    core: List[str] = field(default_factory=list)
    postprocess: List[str] = field(default_factory=list)

    preprocess_compiled: Optional[types.CodeType] = field(init=False, default=None)
    core_compiled: Optional[types.CodeType] = field(init=False, default=None)
    postprocess_compiled: Optional[types.CodeType] = field(init=False, default=None)

    def __post_init__(self):
        """自动编译代码"""
        self.preprocess_compiled = self._compile(self.preprocess)
        self.core_compiled = self._compile(self.core)
        self.postprocess_compiled = self._compile(self.postprocess)

    @classmethod
    def _compile(cls, code_lines: List[str]) -> Optional[types.CodeType]:
        """代码编译方法"""
        if not code_lines:
            return None
        try:
            return compile("\n".join(code_lines), "<string>", "exec")
        except SyntaxError as e:
            return None

    def is_valid(self) -> bool:
        """检查代码是否编译成功"""
        return all(
            compiled is not None or not code
            for compiled, code in [
                (self.preprocess_compiled, self.preprocess),
                (self.core_compiled, self.core),
                (self.postprocess_compiled, self.postprocess),
            ]
        )


@dataclass
class ConvertResult:
    """Paddle2PyTorch 转换结果数据类, 封装 API 转换结果，提供成功/失败的构造方法

    Attributes:
        paddle_api (str): Paddle API 名称
        is_supported (bool): 是否支持转换, 默认为 True
        is_torch_corresponding: 是否与 Torch API 对应，默认为 True
        code (Optional[Code]): 转换后的代码数据对象
        output_var (Optional[str]): 输出变量名，默认值 None 表示 result 保存最后的输出值
        error_message (Optional[str]): 错误信息, 仅当 is_supported = False 时有效

    Methods:
        success(paddle_api, code, output_var): 创建成功转换结果
        error(paddle_api, message): 创建失败转换结果
    """

    paddle_api: str
    is_supported: bool = True
    is_torch_corresponding: bool = True

    code: Optional[Code] = None
    output_var: Optional[str] = None
    error_message: Optional[str] = None

    @classmethod
    def success(
        cls,
        paddle_api: str,
        code: Union[Code, List[str]],
        output_var: str = "result",
        is_torch_corresponding: bool = True,
    ) -> "ConvertResult":
        code_obj = Code(core=code) if isinstance(code, list) else code
        if not code_obj.is_valid():
            return cls.error(paddle_api, "Invalid code.")

        if len(code_obj.core) > 4:
            print(
                f"Warning: The core code of {paddle_api} is too complex.",
                flush=True,
            )

        return cls(
            paddle_api,
            code=code_obj,
            output_var=output_var,
            is_torch_corresponding=is_torch_corresponding,
        )

    @classmethod
    def error(cls, paddle_api: str, message: str) -> "ConvertResult":
        return cls(paddle_api, is_supported=False, error_message=message)


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

    @classmethod
    def _format_arg(cls, arg) -> str:
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
            self.torch_api: str = mapping.get("torch_api", "")
            self.args_map: OrderedDict = mapping.get("paddle_torch_args_map", {})
            self.torch_args: List = mapping.get("torch_args", [])
            self.torch_kwargs: OrderedDict = mapping.get("torch_kwargs", OrderedDict())
            self.is_attribute: bool = mapping.get("is_attribute", False)
            self.defaults: Dict = mapping.get("set_defaults", {})
        else:
            self.composite_steps: List = mapping.get("composite_steps", [])
            for step in self.composite_steps:
                if "torch_api" not in step:
                    raise ValueError(
                        f"Missing required field 'torch_api' in composite step: {step}"
                    )

    def apply_generic(self):
        # if "torch_api" in self.mapping:
        #     self.torch_api: str = self.mapping.get("torch_api", "")
        defaults_code = []
        if "set_defaults" in self.mapping:
            defaults = self.mapping.get("set_defaults", {})
            for default_name, default_value in defaults.items():
                defaults_code.append(
                    f"{default_name} = locals().get('{default_name}', {default_value})"
                )
        map_code = []
        if "torch_args" in self.mapping:
            args = self.mapping.get("torch_args", [])
            map_code.append("_args = []")
            for arg in args:
                map_code.append(f"_args.extend([{self._format_arg(arg)}])")
        if "torch_kwargs" in self.mapping or "paddle_torch_args_map" in self.mapping:
            map_code.append("_kwargs = {}")
        if "torch_kwargs" in self.mapping:
            kwargs = self.mapping.get("torch_kwargs", {})
            for key, value in kwargs.items():
                map_code.append(f"_kwargs['{key}'] = {self._format_arg(value)}")
        if "paddle_torch_args_map" in self.mapping:
            args_map = self.mapping.get("paddle_torch_args_map", {})
            map_code.append("for paddle_param, torch_param in {")
            for paddle_param, torch_param in args_map.items():
                map_code.append(f"    '{paddle_param}': '{torch_param}',")
            map_code.append("}.items():")
            map_code.append("    if paddle_param in locals():")
            map_code.append("        _kwargs[torch_param] = locals()[paddle_param]")
        return defaults_code, map_code


class GenericRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        code = []
        if self.direct_mapping:  # 直接映射
            for default_name, default_value in self.defaults.items():
                code.append(
                    f"{default_name} = locals().get('{default_name}', {default_value})"
                )
            is_tensor_method = paddle_api.startswith("paddle.Tensor.")
            if is_tensor_method:
                if not self.torch_api.startswith("torch.Tensor."):
                    return ConvertResult.error(
                        paddle_api,
                        "The torch api should start with 'torch.Tensor.' when direct mapping a paddle api that starts with 'paddle.Tensor.'",
                    )
                code.append(
                    "_tmp_tensor = args[0] if args else next(iter(kwargs.values()))"
                )
                if self.is_attribute:
                    code.append(f"result = _tmp_tensor.{self.torch_api.split('.')[-1]}")
                    return ConvertResult.success(paddle_api, code)
            is_inplace = (
                paddle_api.endswith("_") and not paddle_api.endswith("__")
            ) or paddle_api == "paddle.Tensor.__setitem__"

            code.append("_args = args[1:]")
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
    def __init__(self, message: str = "Error Rule"):
        super().__init__()
        self.message = message

    def apply(self, paddle_api: str) -> ConvertResult:
        return ConvertResult.error(paddle_api, self.message)


# a
class Adaptive_log_softmax_with_lossRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
def scatter_nd(index, updates, shape):
    output = torch.zeros(shape, dtype=updates.dtype).to(updates.device)
    if index.numel() == 0:
        result = output + updates
    else:
        flat_index = index.view(-1, index.size(-1))
        flat_updates = updates.reshape(flat_index.size(0), *updates.shape[index.dim()-1:])
        for i in range(flat_index.size(0)):
            idx_tuple = tuple(flat_index[i])
            output[idx_tuple] += flat_updates[i]
        result = output   
    return result
        
input = locals().get('input')
label = locals().get('label')
head_weight = locals().get('head_weight')
tail_weight = locals().get('tail_weights')
cutoffs = locals().get('cutoffs')
head_bias = locals().get('head_bias', None)

target_dim = label.dim()

is_batched = target_dim > 0
input = input if is_batched else input.unsqueeze(0)
label = label if is_batched else label.unsqueeze(0)

used_rows = 0
batch_size = label.shape[0]

output = torch.zeros([batch_size], dtype = input.dtype)
gather_inds = torch.empty([batch_size], dtype = label.dtype)

cutoff_values = [0, *cutoffs]
for i in range(len(cutoff_values) - 1):
    index1 = cutoff_values[i]
    index2 = cutoff_values[i + 1]
    label_mask = (label >= index1) & (label < index2)
    row_indices = label_mask.nonzero().squeeze()
    if row_indices.numel() == 0:
        continue
    
    if i == 0:
        scatter_output = scatter_nd(
            index = torch.unsqueeze(row_indices, 1),
            updates = torch.masked_select(label, label_mask),
            shape = gather_inds.shape
        )
        gather_inds = scatter_output
    else:
        relative_label = label[label_mask] - index1
        input_subset = input.index_select(index = row_indices, dim = 0)
        cluster_output = torch.nn.functional.linear(
            input = input_subset, weight = tail_weight[i-1][0].t()
        )
        cluster_output = torch.nn.functional.linear(
            input = cluster_output, weight = tail_weight[i-1][1].t()
        )

        cluster_index = cutoffs[0] + i - 1
        
        gather_inds = torch.index_fill(
            gather_inds, 0, row_indices, cluster_index
        )

        cluster_logprob = torch.nn.functional.log_softmax(
            cluster_output, dim = 1
        )

        local_logprob = torch.gather(
            cluster_logprob, dim = 1, index = relative_label.unsqueeze(1)
        )
        
        scatter_output = scatter_nd(
            row_indices.unsqueeze(1), local_logprob.squeeze(1), output.shape
        )
        output = (
            output * (scatter_output == 0).float()
            + scatter_output
        )
    used_rows += row_indices.numel()
if head_bias is not None:
    head_output = torch.nn.functional.linear(
        input = input, weight = head_weight.t(), bias = head_bias
    )
else:
    head_output = torch.nn.functional.linear(
        input = input, weight = head_weight.t()
    )    
head_logprob = torch.nn.functional.log_softmax(head_output, dim = 1)
output += torch.gather(
    head_logprob, dim = 1, index = gather_inds.unsqueeze(1)
).squeeze()
loss = (-output).mean()

if not is_batched:
    output = output.squeeze(0)
    
result = [output, loss]
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code, "result")


class PoolRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        head_code, map_code = self.apply_generic()
        func1 = """
kernel_size = tuple(kernel_size) if isinstance(kernel_size, list) else kernel_size
stride = tuple(stride) if isinstance(stride, list) else stride

def _get_same_padding_1d(input_size, kernel_size, stride):
    if stride is None:
        stride = kernel_size
    output_size = (input_size + stride - 1) // stride
    total_pad = max(0, (output_size - 1) * stride + kernel_size - input_size)
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    return pad_left, pad_right

if isinstance(padding, str):
    if padding.upper() == "VALID":
        padding = 0
    elif padding.upper() == "SAME":
        input_size = x.shape[2]
        pad_left, pad_right = _get_same_padding_1d(input_size, kernel_size, stride)
        padding = pad_left # 对称填充
        if pad_left != pad_right:  # 非对称填充
            x = torch.nn.functional.pad(x, (pad_left, pad_right))
            padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 1:  # [pad]
        padding = tuple(padding)
    elif len(padding) == 2:  # [pad_left, pad_right]
        pad_left, pad_right = padding
        x = torch.nn.functional.pad(x, (pad_left, pad_right))
        padding = 0
"""
        func2 = """
kernel_size = tuple(kernel_size) if isinstance(kernel_size, list) else kernel_size
stride = tuple(stride) if isinstance(stride, list) else stride
if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
            
def _get_same_padding_2d(input_size, kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride)
    output_size_h = (input_size[0] + stride[0] - 1) // stride[0]
    output_size_w = (input_size[1] + stride[1] - 1) // stride[1]
    total_pad_h = max(0, (output_size_h - 1) * stride[0] + kernel_size[0] - input_size[0])
    total_pad_w = max(0, (output_size_w - 1) * stride[1] + kernel_size[1] - input_size[1])
    pad_h = (total_pad_h // 2, total_pad_h - total_pad_h // 2)
    pad_w = (total_pad_w // 2, total_pad_w - total_pad_w // 2)
    return pad_h, pad_w

if isinstance(padding, str):
    if padding == "VALID":
        padding = 0
    elif padding == "SAME":
        input_size = (x.shape[2], x.shape[3])
        pad_h, pad_w = _get_same_padding_2d(input_size, kernel_size, stride)
        padding = (pad_h[0], pad_w[0]) # 对称填充
        if pad_h[0] != pad_h[1] or pad_w[0] != pad_w[1]: # 非对称填充
            x = torch.nn.functional.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1]))
            padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 2: # [pad_height, pad_width]
        padding = tuple(padding)
    elif len(padding) == 4:
        if all(isinstance(p, (list, tuple)) for p in padding): # Paddle 的 4D 填充格式(NCHW 或 NHWC)
            if data_format == "NCHW":
                pad_top, pad_bottom = padding[2]
                pad_left, pad_right = padding[3]
            else:  # NHWC
                pad_top, pad_bottom = padding[1]
                pad_left, pad_right = padding[2]
        else: # [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]
            pad_top, pad_bottom, pad_left, pad_right = padding
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        padding = 0
"""
        func3 = """
kernel_size = tuple(kernel_size) if isinstance(kernel_size, list) else kernel_size
stride = tuple(stride) if isinstance(stride, list) else stride
if data_format == 'NDHWC':
    x = x.permute(0, 4, 1, 2, 3)
        
def _get_same_padding_3d(input_size, kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 3
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride,) * 3
    output_size_d = (input_size[0] + stride[0] - 1) // stride[0]
    output_size_h = (input_size[1] + stride[1] - 1) // stride[1]
    output_size_w = (input_size[2] + stride[2] - 1) // stride[2]
    total_pad_d = max(0, (output_size_d - 1) * stride[0] + kernel_size[0] - input_size[0])
    total_pad_h = max(0, (output_size_h - 1) * stride[1] + kernel_size[1] - input_size[1])
    total_pad_w = max(0, (output_size_w - 1) * stride[2] + kernel_size[2] - input_size[2])
    pad_d = (total_pad_d // 2, total_pad_d - total_pad_d // 2)
    pad_h = (total_pad_h // 2, total_pad_h - total_pad_h // 2)
    pad_w = (total_pad_w // 2, total_pad_w - total_pad_w // 2)
    return pad_d, pad_h, pad_w

if isinstance(padding, str):
    if padding == "VALID":
        padding = 0
    elif padding == "SAME":
        input_size = (x.shape[2], x.shape[3], x.shape[4])  # (D, H, W)
        pad_d, pad_h, pad_w = _get_same_padding_3d(input_size, kernel_size, stride)
        padding = (pad_d[0], pad_h[0], pad_w[0]) # 对称填充
        if pad_d[0] != pad_d[1] or pad_h[0] != pad_h[1] or pad_w[0] != pad_w[1]: # 非对称填充
            x = torch.nn.functional.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1], pad_d[0], pad_d[1]))
            padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 3:  # [pad_depth, pad_height, pad_width]
        max_pad = [kernel_size[i] // 2 for i in range(3)]
        if any(p > m for p, m in zip(padding, max_pad)):
            pad_d, pad_h, pad_w = padding
            x = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d))
            padding = 0
        else:
            padding = tuple(padding)
    elif len(padding) == 6:  # [front, back, top, bottom, left, right]
        pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right = padding
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        padding = 0
    elif len(padding) == 5: # Paddle 的 5D 填充格式
        if data_format == "NCDHW":
            pad_front, pad_back = padding[2]
            pad_top, pad_bottom = padding[3]
            pad_left, pad_right = padding[4]
        else: # NDHWC
            pad_front, pad_back = padding[1]
            pad_top, pad_bottom = padding[2]
            pad_left, pad_right = padding[3]
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        padding = 0
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        impl2 = """
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        impl3 = """
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        if paddle_api.endswith("_pool1d"):
            code = head_code + func1.splitlines() + map_code + core.splitlines()
        elif paddle_api.endswith("_pool2d"):
            code = (
                head_code
                + func2.splitlines()
                + map_code
                + core.splitlines()
                + impl2.splitlines()
            )
        elif paddle_api.endswith("_pool3d"):
            code = (
                head_code
                + func3.splitlines()
                + map_code
                + core.splitlines()
                + impl3.splitlines()
            )
        else:
            return ConvertResult.error(
                paddle_api, f"Unsupported pooling api: {paddle_api}"
            )
        return ConvertResult.success(paddle_api, code)


# b
class BroadcastShapeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
x_shape = locals().get('x_shape')
y_shape = locals().get('y_shape')
result = torch.broadcast_shapes(x_shape, y_shape)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code, "result")


class BroadcastTensorsRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        code = ["result = torch.broadcast_tensors(*input)"]
        return ConvertResult.success(paddle_api, code, "result")


class BatchNormRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        head_code, map_code = self.apply_generic()
        impl1 = """
if locals().get('data_format') == 'NHWC':
    x = x.permute(0, 3, 1, 2)
if 'running_mean' in locals():
    running_mean.requires_grad = False
if 'running_var' in locals():
    running_var.requires_grad = False
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        impl2 = """
if locals().get('data_format') == 'NHWC':
    result = result.permute(0, 2, 3, 1)
"""
        code = (
            head_code
            + impl1.splitlines()
            + map_code
            + core.splitlines()
            + impl2.splitlines()
        )
        return ConvertResult.success(paddle_api, code)


# c
class CorrcoefRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
rowvar = locals().get('rowvar',True)
if rowvar:
    result = torch.corrcoef(x)
else:
    x = x.t()
    result = torch.corrcoef(x).t()
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code, "result")

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


class ClassCenterSampleRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
unique_pos_classes = torch.unique(label)
num_pos_classes = unique_pos_classes.size(0)
if num_pos_classes >= num_samples:
    sampled_classes = unique_pos_classes
    remapped_label = torch.zeros_like(label)
    for new_idx, old_class in enumerate(sampled_classes):
        remapped_label[label == old_class] = new_idx
else:
    all_classes = torch.arange(num_classes, device=label.device)
    neg_classes = all_classes[~torch.isin(all_classes, unique_pos_classes)]
    num_neg_needed = num_samples - num_pos_classes
    if num_neg_needed > 0:
        if neg_classes.numel() >= num_neg_needed:
            selected_neg = neg_classes[torch.randperm(neg_classes.size(0))[:num_neg_needed]]
        else:
            selected_neg = neg_classes
        sampled_classes = torch.cat([unique_pos_classes, selected_neg])
    else:
        sampled_classes = unique_pos_classes
    remapped_label = torch.zeros_like(label)
    for new_idx, old_class in enumerate(sampled_classes):
        remapped_label[label == old_class] = new_idx
result = (remapped_label, sampled_classes)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class Conv1dTransposeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        head_code, map_code = self.apply_generic()
        impl1 = """
crop = None
if bias is not None:
    out_channels = weight.size(1) * groups
    bias = bias.expand(out_channels)
stride = stride[0] if isinstance(stride, (list, tuple)) else stride
output_padding = output_padding[0] if isinstance(output_padding, (list, tuple)) else output_padding
dilation = dilation[0] if isinstance(dilation, (list, tuple)) else dilation
output_size = output_size[0] if isinstance(output_size, (list, tuple)) else output_size
if data_format == "NLC":
    x = x.transpose(1, 2)
if isinstance(padding, str):
    if padding.upper() == "SAME":
        kernel_size = weight.size(-1)
        padding = (dilation * (kernel_size - 1)) // 2
    elif padding.upper() == "VALID":
        padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 1:
        padding = padding[0]
    elif len(padding) == 2:
        crop = padding
        padding = 0
    elif len(padding) == 3:
        crop = padding[1] if data_format == "NLC" else padding[2]
        padding = 0
if output_size is not None:
    L_in = x.size(-1)
    kernel_size = weight.size(-1)
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    output_padding = output_size - L_out
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        impl2 = """
if crop:
    result = result[:, :, crop[0]:result.size(-1) - crop[1]]
if data_format == "NLC":
    result = result.transpose(1, 2)
"""
        code = (
            head_code
            + impl1.splitlines()
            + map_code
            + core.splitlines()
            + impl2.splitlines()
        )
        return ConvertResult.success(paddle_api, code)


class Conv2dTransposeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        head_code, map_code = self.apply_generic()
        impl1 = """
crop = None
if bias is not None:
    out_channels = weight.size(1) * groups
    bias = bias.expand(out_channels)
stride = tuple(stride) if isinstance(stride, list) else stride
output_padding = tuple(output_padding) if isinstance(output_padding, list) else output_padding
dilation = tuple(dilation) if isinstance(dilation, list) else dilation
if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
if isinstance(padding, str):
    if padding.upper() == "SAME":
        padding = []
        for i in range(2):
            dilation_i = dilation[i] if isinstance(dilation, tuple) else dilation
            kernel_size = weight.size(2 + i)
            padding.append((dilation_i * (kernel_size - 1)) // 2)
        padding = tuple(padding)
    elif padding.upper() == "VALID":
        padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 2:
        padding = tuple(padding)
    elif len(padding) == 4 and all(isinstance(pad, int) for pad in padding):
        crop = padding
        padding = 0
    elif len(padding) == 4:
        crop = []
        if data_format == "NHWC":
            for i in range(1, 3):
                crop.extend(padding[i])
        else:
            for i in range(2, 4):
                crop.extend(padding[i])
        padding = 0
if output_size is not None:
    output_padding = []
    for i in range(2):
        L_in = x.size(2 + i)
        kernel_size = weight.size(2 + i)
        stride_i = stride[i] if isinstance(stride, tuple) else stride
        padding_i = padding[i] if isinstance(padding, tuple) else padding
        dilation_i = dilation[i] if isinstance(dilation, tuple) else dilation
        L_out = (L_in - 1) * stride_i - 2 * padding_i + dilation_i * (kernel_size - 1) + 1
        output_padding.append(output_size[i] - L_out)
    output_padding = tuple(output_padding)
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        impl2 = """
if crop:
    result = result[:, :, crop[0]:result.size(-1) - crop[1], crop[2]:result.size(-2) - crop[3]]
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        code = (
            head_code
            + impl1.splitlines()
            + map_code
            + core.splitlines()
            + impl2.splitlines()
        )
        return ConvertResult.success(paddle_api, code)


class Conv3dTransposeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        head_code, map_code = self.apply_generic()
        impl1 = """
crop = None
if bias is not None:
    out_channels = weight.size(1) * groups
    bias = bias.expand(out_channels)
stride = tuple(stride) if isinstance(stride, list) else stride
output_padding = tuple(output_padding) if isinstance(output_padding, list) else output_padding
dilation = tuple(dilation) if isinstance(dilation, list) else dilation
if data_format == "NDHWC":
    x = x.permute(0, 4, 1, 2, 3)
if isinstance(padding, str):
    if padding.upper() == "SAME":
        padding = []
        for i in range(3):
            dilation_i = dilation[i] if isinstance(dilation, tuple) else dilation
            kernel_size = weight.size(2 + i)
            padding.append((dilation_i * (kernel_size - 1)) // 2)
        padding = tuple(padding)
    elif padding.upper() == "VALID":
        padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 3:
        padding = tuple(padding)
    elif len(padding) == 6:
        crop = padding
        padding = 0
    elif len(padding) == 5:
        crop = []
        if data_format == "NDHWC":
            for i in range(1, 4):
                crop.extend(padding[i])
        else:
            for i in range(2, 5):
                crop.extend(padding[i])
        padding = 0
if output_size is not None:
    output_padding = []
    for i in range(3):
        L_in = x.size(2 + i)
        kernel_size = weight.size(2 + i)
        stride_i = stride[i] if isinstance(stride, tuple) else stride
        padding_i = padding[i] if isinstance(padding, tuple) else padding
        dilation_i = dilation[i] if isinstance(dilation, tuple) else dilation
        L_out = (L_in - 1) * stride_i - 2 * padding_i + dilation_i * (kernel_size - 1) + 1
        output_padding.append(output_size[i] - L_out)
    output_padding = tuple(output_padding)
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        impl2 = """
if crop:
    result = result[:, :, crop[0]:result.size(-3) - crop[1], crop[2]:result.size(-2) - crop[3], crop[4]:result.size(-1) - crop[5]]
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        code = (
            head_code
            + impl1.splitlines()
            + map_code
            + core.splitlines()
            + impl2.splitlines()
        )
        return ConvertResult.success(paddle_api, code)


class Conv1dRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
if data_format == "NLC":
    x = x.permute(0, 2, 1)
stride = tuple(stride) if isinstance(stride, list) else stride
dilation = tuple(dilation) if isinstance(dilation, list) else dilation
if isinstance(padding, str):
    if padding.lower() == "same":
        padding = "same"
    elif padding.lower() == "valid":
        padding = "valid"
elif isinstance(padding, list):
    if len(padding) == 2:
        pad_left, pad_right = padding
        x = torch.nn.functional.pad(x, (pad_left, pad_right))
        padding = 0
    else:
        padding = tuple(padding)
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        post = """
if data_format == "NLC":
    result = result.permute(0, 2, 1)
"""
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=core.splitlines(),
            postprocess=post.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


class Conv2dRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        head_code, map_code = self.apply_generic()
        impl1 = """
if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
stride = tuple(stride) if isinstance(stride, list) else stride
dilation = tuple(dilation) if isinstance(dilation, list) else dilation
if isinstance(padding, str):
    if padding.lower() == "same":
        padding = "same"
    elif padding.lower() == "valid":
        padding = "valid"
elif isinstance(padding, list):
    if len(padding) == 2:  # [pad_height, pad_width]
        padding = tuple(padding)
    elif len(padding) == 4:
        if all(isinstance(pad, int) for pad in padding): # [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]
            pad_top, pad_bottom, pad_left, pad_right = padding
        else: # Paddle 的 4D 填充格式(NCHW 或 NHWC)
            if data_format == "NCHW":
                pad_top, pad_bottom = padding[2][0], padding[2][1]
                pad_left, pad_right = padding[3][0], padding[3][1]
            else:  # NHWC
                pad_top, pad_bottom = padding[1][0], padding[1][1]
                pad_left, pad_right = padding[2][0], padding[2][1]
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        padding = 0
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        impl2 = """
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        code = (
            head_code
            + impl1.splitlines()
            + map_code
            + core.splitlines()
            + impl2.splitlines()
        )
        return ConvertResult.success(paddle_api, code)


class Conv3dRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        head_code, map_code = self.apply_generic()
        impl1 = """
if data_format == "NDHWC":
    x = x.permute(0, 4, 1, 2, 3)
stride = tuple(stride) if isinstance(stride, list) else stride
dilation = tuple(dilation) if isinstance(dilation, list) else dilation
if isinstance(padding, str):
    if padding.lower() == "same":
        padding = "same"
    elif padding.lower() == "valid":
        padding = "valid"
elif isinstance(padding, list):
    if len(padding) == 3:  # [pad_depth, pad_height, pad_width]
        padding = tuple(padding)
    elif len(padding) == 6:  # [front, back, top, bottom, left, right]
        pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right = padding
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        padding = 0
    elif len(padding) == 5: # Paddle 的 5D 填充格式
        if data_format == "NCDHW":
            pad_front, pad_back = padding[2][0], padding[2][1]
            pad_top, pad_bottom = padding[3][0], padding[3][1]
            pad_left, pad_right = padding[4][0], padding[4][1]
        else: # NDHWC
            pad_front, pad_back = padding[1][0], padding[1][1]
            pad_top, pad_bottom = padding[2][0], padding[2][1]
            pad_left, pad_right = padding[3][0], padding[3][1]
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        padding = 0
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        impl2 = """
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        code = (
            head_code
            + impl1.splitlines()
            + map_code
            + core.splitlines()
            + impl2.splitlines()
        )
        return ConvertResult.success(paddle_api, code)


# d
class DataFormatRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        head_code, map_code = self.apply_generic()
        impl1 = """
if data_format == "NDHWC":
    x = x.permute(0, 4, 1, 2, 3)
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        impl2 = """
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        code = (
            head_code
            + impl1.splitlines()
            + map_code
            + core.splitlines()
            + impl2.splitlines()
        )
        return ConvertResult.success(paddle_api, code)


class Distribute_fpn_proposalsRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
import math
        
def BBoxArea(box, pixel_offset):
    w = box[2] - box[0]
    h = box[3] - box[1]
    if pixel_offset:
        return (w+1) * (h+1)
    else:
        return w * h     

pixel_offset = locals().get('pixel_offset',False)
rois_num = locals().get('rois_num', None)
num_level = max_level - min_level + 1
if rois_num is not None:
    for i in range(1, rois_num.numel()):
        rois_num[i] += rois_num[i-1]
    fpn_rois_lod = torch.concat([torch.tensor([0]), rois_num])
else:
    fpn_rois_lod = torch.tensor([0, fpn_rois.shape[0]])   

size = fpn_rois_lod.numel() - 1
fpn_rois_num = (int)(fpn_rois_lod[size])
# 计算roi所属的level
num_rois_level = torch.zeros([num_level])
target_level = []
for i in range(fpn_rois_lod.numel() - 1):
    fpn_rois_slice = fpn_rois[fpn_rois_lod[i]:fpn_rois_lod[i+1]]
    for rois_data in fpn_rois_slice:
        roi_scale = math.sqrt(BBoxArea(rois_data, pixel_offset))
        tgt_lvl = math.floor(math.log2(roi_scale / refer_scale) + refer_level)
        tgt_lvl = min(max_level, max(tgt_lvl, min_level))
        target_level.append(tgt_lvl)
        num_rois_level[tgt_lvl - min_level] += 1 
# 初始化结果
multi_rois = []
for i in range(num_level):
    multi_rois.append([])
restore_ind = torch.empty(fpn_rois.shape[0], 1)
rois_num_per_level = []
for i in range(num_level):
    rois_num_per_level.append(
        torch.zeros([rois_num.numel()])
    )
# 计算结果
index = 0
for i in range(fpn_rois_lod.numel() - 1):
    fpn_rois_slice = fpn_rois[fpn_rois_lod[i]:fpn_rois_lod[i+1]]
    for rois_data in fpn_rois_slice:
        level = target_level[index]
        if multi_rois[level-min_level] == []:
            multi_rois[level-min_level].append(rois_data)
        else:
            multi_rois[level-min_level].append(rois_data)
        rois_num_per_level[level - min_level][i] += 1
        index += 1
for i in range(num_level):
    if multi_rois[i] == []:
        multi_rois[i] = torch.zeros([0,4])
    else:
        multi_rois[i] = torch.stack(multi_rois[i])
index = 0
for i in range(num_level):
    for j in range(fpn_rois.shape[0]):
        if target_level[j] == i + min_level:
            restore_ind[j] = index
            index += 1

result = (multi_rois, restore_ind, rois_num_per_level)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class DropoutRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
def axis_dropout(x, p, axis, training=True, mode='upscale_in_train'):
    if isinstance(axis, int):
        axis = [axis]
    mask_shape = [x.shape[i] if i in axis else 1 for i in range(x.dim())]
    mask = torch.bernoulli(torch.full(mask_shape, 1-p)).to(x.device)
    if mode == 'upscale_in_train':
        if training:
            return x * mask / (1.0 - p)
        else:
            return x
    elif mode == 'downscale_in_infer':
        if training:
            return x * mask
        else:
            return x * (1.0 - p)
    else:
        raise ValueError(f"Invalid mode: {mode}")
        
x = locals().get('x')
p = locals().get('p')
axis = locals().get('axis')
training = locals().get('training')
mode = locals().get('mode')
result = axis_dropout(x, p, axis, training, mode) if axis is not None else torch.nn.functional.dropout(input=x, p=float(p), training=training)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code, "result")


class Dropout2dRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
x = locals().get('x')
p = locals().get('p')
training = locals().get('training')
data_format = locals().get('data_format')

if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
result = torch.nn.functional.dropout(input=x, p=float(p), training=training)
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code, "result")


class Dropout3dRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
x = locals().get('x')
p = locals().get('p')
training = locals().get('training')
data_format = locals().get('data_format')

if data_format == "NDHWC":
    x = x.permute(0, 4, 1, 2, 3)
result = torch.nn.functional.dropout3d(input=x, p=float(p), training=training)
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code, "result")


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
class FractionalMaxPoolRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        head_code, map_code = self.apply_generic()
        func1 = """
batch_size, C = x.shape[0], x.shape[1]
if locals().get('random_u') is not None:
    random_u = torch.tensor([[[random_u] * 2] * C] * batch_size, dtype=x.dtype, device=x.device)

def compute_kernel_size(x, output_size):
    H_in, W_in = x.shape[2], x.shape[3]
    if isinstance(output_size, int):
        H_out = W_out = output_size
    else:
        H_out, W_out = output_size
    
    def compute_k(input_size, output_size):
        if output_size is None or output_size == input_size:
            return 1  # No pooling
        else:
            return (input_size + output_size - 1) // output_size  # ceil(input_size / output_size)
    
    kH = compute_k(H_in, H_out)
    kW = compute_k(W_in, W_out)
    return (kH, kW)
"""
        func2 = """
batch_size, C = x.shape[0], x.shape[1]
if locals().get('random_u') is not None:
    random_u = torch.tensor([[[random_u] * 3] * C] * batch_size, dtype=x.dtype, device=x.device)

def compute_kernel_size(x, output_size):
    D_in, H_in, W_in = x.shape[2], x.shape[3], x.shape[4]
    if isinstance(output_size, int):
        D_out = H_out = W_out = output_size
    else:
        D_out, H_out, W_out = output_size

    def compute_k(input_size, output_size):
        if output_size is None or output_size == input_size:
            return 1  # No pooling
        else:
            return (input_size + output_size - 1) // output_size  # ceil(input_size / output_size)

    kD = compute_k(D_in, D_out)
    kH = compute_k(H_in, H_out)
    kW = compute_k(W_in, W_out)
    return (kD, kH, kW)  
"""
        impl = """
kernel_size = locals().get('kernel_size')
if kernel_size is None:
    kernel_size = compute_kernel_size(x, output_size)
elif isinstance(kernel_size, list):
    kernel_size = tuple(kernel_size)
if isinstance(output_size, (list, tuple)):
    output_size = tuple([x.shape[i + 2] if size is None else size 
                       for i, size in enumerate(output_size)])
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        code = head_code
        if paddle_api == "paddle.nn.functional.fractional_max_pool2d":
            code += func1.splitlines()
        else:
            code += func2.splitlines()
        code += impl.splitlines() + map_code + core.splitlines()
        return ConvertResult.success(paddle_api, code)


# g


class GatherRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        # 抽取对应维度的tensor直接进行stack操作
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


class GenerateProposalsRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """    
import torchvision
import math
def nms(box1, box2, normaloized):
    if box2[0] > box1[2] or box2[2] < box1[0] or box2[1] > box1[3] or box2[3] < box1[1]:
        return 0
    if normaloized:
        norm = 0
    else:
        norm = 1
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    w = x_max - x_min + norm
    h = y_max - y_min + norm
    area = w * h
    if box1[0] > box1[2] or box1[1] > box1[3]:
        area1 = 0
    else:
        area1 = (box1[2] - box1[0] + norm) * (box1[3] - box1[1] + norm)
    if box2[0] > box2[2] or box2[1] > box2[3]:
        area2 = 0
    else:
        area2 = (box2[2] - box2[0] + norm) * (box2[3] - box2[1] + norm)
    return area / (area1 + area2 - area)
    
pre_nms_top_n = locals().get('pre_nms_top_n', 6000)
post_nms_top_n = locals().get('post_nms_top_n', 1000)
nms_thresh = locals().get('nms_thresh', 0.5)
min_size = locals().get('min_size', 0.1)
eta = locals().get('eta', 1.0)
pixel_offset = locals().get('pixel_offset', False)
return_rois_num = locals().get('return_rois_num', False)

# 初始化结果
rpn_rois = []
rpn_roi_probs = []

# 调整大小
scores = scores.permute(0,2,3,1)
bbox_deltas = bbox_deltas.permute(0,2,3,1)
scores = scores.reshape([scores.shape[0],-1, 1])
bbox_deltas = bbox_deltas.reshape([bbox_deltas.shape[0],-1, 4])
anchors = anchors.reshape([-1,4])
variances = variances.reshape([-1,4])
proposal = torch.empty([scores.shape[0], scores.shape[1],4])

#逐张图片进行处理
for ii in range(scores.shape[0]):
    scores_i = scores[ii]
    bbox_deltas_i = bbox_deltas[ii]
    img_size_i = img_size[ii]
    proposal_i = proposal[ii]

    class cls:
        def __init__(self, scores, index):
            self.scores = scores
            self.index = index
    ind = []
    for j in range(scores_i.numel()):
        c = cls(scores_i[j,0], j)
        ind.append(c)    
    ind = sorted(ind, key = lambda x : x.scores, reverse = True)   
    for j in range(len(ind)):
        ind[j] = ind[j].index    
    if pre_nms_top_n < scores_i.numel():
        ind = torch.tensor(ind[:pre_nms_top_n]).squeeze()
    else:
        ind = torch.tensor(ind).squeeze()
    scores_i = scores_i.index_select(0, ind)
    bbox_deltas_i = bbox_deltas_i.index_select(0, ind)
    anchors_i = anchors.index_select(0, ind)
    variances_i = variances.index_select(0, ind)
    proposal_i = proposal_i.index_select(0, ind)

    #计算候选框的位置 
    if pixel_offset == True:
        offset = 1
    else:
        offset = 0
    for i in range(anchors_i.shape[0]):
        anchor_width = anchors_i[i][2] - anchors_i[i][0] + offset
        anchor_height = anchors_i[i][3] - anchors_i[i][1] + offset
        anchor_center_x = anchors_i[i][0] + 0.5 * anchor_width
        anchor_center_y = anchors_i[i][1] + 0.5 * anchor_height
        bbox_center_x = variances_i[i][0] * bbox_deltas_i[i, 0] * anchor_width + anchor_center_x
        bbox_center_y = variances_i[i][1] * bbox_deltas_i[i, 1] * anchor_height + anchor_center_y
        bbox_width = anchor_width * torch.exp(min(variances_i[i][2] * bbox_deltas_i[i, 2], math.log(1000.0 / 16.0)))
        bbox_height = anchor_height * torch.exp(min(variances_i[i][3] * bbox_deltas_i[i, 3], math.log(1000.0 / 16.0)))

        proposal_i[i,0] = bbox_center_x - 0.5 * bbox_width
        proposal_i[i,1] = bbox_center_y - 0.5 * bbox_height
        proposal_i[i,2] = bbox_center_x + 0.5 * bbox_width - offset
        proposal_i[i,3] = bbox_center_y + 0.5 * bbox_height - offset

    # 将检测框的坐标限定到图像尺寸范围内。
    for i in range(proposal_i.shape[0]):
        proposal_i[i,0] = max(min(float(proposal_i[i,0]), img_size_i[1]), 0)
        proposal_i[i,1] = max(min(float(proposal_i[i,1]), img_size_i[0]), 0)
        proposal_i[i,2] = max(min(float(proposal_i[i,2]), img_size_i[1]), 0)
        proposal_i[i,3] = max(min(float(proposal_i[i,3]), img_size_i[0]), 0)

    # 源码将这里限制为1 如果取消注释 这里将和源码一样
    # min_size = max(min_size,1.)
    #删除面积较小的候选框
    proposal_i = proposal_i.reshape([-1, 4])
    keep = []
    for i in range(proposal_i.shape[0]):
        w = proposal_i[i,2] - proposal_i[i,0]
        h = proposal_i[i,3] - proposal_i[i,1]
        if pixel_offset:
            x_cen = proposal_i[i,0] + 0.5 * w
            y_cen = proposal_i[i,1] + 0.5 * h
            if w >= min_size and h >= min_size and x_cen <= img_size_i[1] and y_cen <= img_size_i[0]:
                keep.append(i)
        elif w >= min_size and h >= min_size:
            keep.append(i)
    keep = torch.tensor(keep).squeeze()
    proposal_i = proposal_i.index_select(0,keep)
    scores_i = scores_i.index_select(0,keep)

    # 通过非极大抑制，选出合适的候选框
    adaptive_threshold = nms_thresh
    nomormalized = not pixel_offset
    selected_index = []
    selected_num = 0
    for num in range(proposal_i.shape[0]):
        flag =True
        for i in selected_index:
            if flag:
                overlap = nms(proposal_i[i], proposal_i[num], nomormalized)
                flag = overlap <= adaptive_threshold
            else:
                break
        if flag:
            selected_index.append(num)
            selected_num += 1
        if flag and eta < 1 and adaptive_threshold > 0.5:
            adaptive_threshold = adaptive_threshold * eta
    if selected_num > post_nms_top_n:
        selected_index = selected_index[:post_nms_top_n]
    proposal_i = proposal_i.index_select(0,torch.tensor(selected_index).squeeze())
    scores_i = scores_i.index_select(0,torch.tensor(selected_index).squeeze())

    #汇集结果
    rpn_rois.append(proposal_i)
    rpn_roi_probs.append(scores_i)

# 返回结果
if return_rois_num:
    num = []
    for i in range(len(rpn_rois)):
        num.append(rpn_rois[i].numel()//4)
    result = (torch.stack(rpn_rois).squeeze(), torch.stack(rpn_roi_probs).squeeze(0), torch.tensor(num).squeeze())
else:
    result = (torch.stack(rpn_rois).squeeze(), torch.stack(rpn_roi_probs).squeeze())
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class GetWindowRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """

def general_gaussian(M, p, sig, sym=True, dtype=torch.float64):
    if M < 1:
        return torch.tensor([], dtype=dtype)
    if M == 1:
        return torch.ones(1, dtype=dtype)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M, dtype=dtype)
    w = n.new_empty(n.shape)
    if not sym and not odd:
        n = n[:-1]
    sig2 = 2 * sig * sig
    w = torch.exp(-torch.pow(n - (M - 1.0) / 2.0, 2.0 * p) / sig2)
    return w

def triang(M, sym=True, dtype=torch.float64):
    if M < 1:
        return torch.tensor([], dtype=dtype)
    if M == 1:
        return torch.ones(1, dtype=dtype)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(1, (M + 1) // 2 + 1, dtype=dtype)
    if M % 2 == 0:
        w = (2 * n - 1.0) / M
        w = torch.cat([w, w.flip(0)])
    else:
        w = 2 * n / (M + 1.0)
        w = torch.cat([w, w[-2::-1]])
    if not sym and not odd:
        w = w[:-1]
    return w

def bohman(M, sym=True, dtype=torch.float64):
    if M < 1:
        return torch.tensor([], dtype=dtype)
    if M == 1:
        return torch.ones(1, dtype=dtype)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    fac = torch.linspace(-1, 1, M, dtype=dtype)
    w = (1 - torch.abs(fac)) * torch.cos(torch.pi * torch.abs(fac)) + 1.0 / torch.pi * torch.sin(torch.pi * torch.abs(fac))
    if not sym and not odd:
        w = w[:-1]
    return w

def tukey(M, alpha=0.5, sym=True, dtype=torch.float64):
    if M < 1:
        return torch.tensor([], dtype=dtype)
    if M == 1:
        return torch.ones(1, dtype=dtype)
    if alpha <= 0:
        return torch.ones(M, dtype=dtype)
    if alpha >= 1:
        return torch.hann_window(M, periodic=not sym, dtype=dtype)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M, dtype=dtype)
    width = int(alpha * (M - 1) / 2.0)
    n1 = n[0:width+1]
    n2 = n[width+1:M-width-1]
    n3 = n[M-width-1:]
    w1 = 0.5 * (1 + torch.cos(torch.pi * (-1 + 2.0*n1/alpha/(M-1))))
    w2 = torch.ones(len(n2), dtype=dtype)
    w3 = 0.5 * (1 + torch.cos(torch.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
    w = torch.cat([w1, w2, w3])
    if not sym and not odd:
        w = w[:-1]
    return w

def taylor(M, nbar=4, sll=30, norm=True, sym=True, dtype=torch.float64):
    if M < 1:
        return torch.tensor([], dtype=dtype)
    if M == 1:
        return torch.ones(1, dtype=dtype)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    B = 10**(sll / 20)
    A = torch.log(B + torch.sqrt(B**2 - 1)) / torch.pi
    sigma2 = nbar**2 / (A**2 + (nbar - 0.5)**2)
    fm = lambda m: torch.prod(torch.tensor([(1 - (m/torch.sqrt(sigma2))**2/(n**2 + (n-0.5)**2)) 
                                          for n in range(1, nbar)], dtype=dtype))
    coefficients = torch.tensor([fm(i) for i in range(nbar)], dtype=dtype)
    n = torch.arange(-(M-1)/2, (M+1)/2, dtype=dtype) * 2/M
    w = coefficients[0]
    for i in range(1, nbar):
        w = w + coefficients[i] * torch.cos(2 * torch.pi * i * torch.arange(M, dtype=dtype) / M)
    if norm:
        w = w / w.max()
    if not sym and not odd:
        w = w[:-1]
    return w

if isinstance(window, tuple):
    window_name, param = window[0], window[1:]
else:
    window_name, param = window, None
fftbins = locals().get('fftbins', True)
dtype = locals().get('dtype', 'float64')
dtype = getattr(torch, dtype)
if window_name == 'hamming':
    window = torch.signal.windows.hamming(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'hann':
    window = torch.signal.windows.hann(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'gaussian':
    window = torch.signal.windows.gaussian(win_length, std=param[0], sym=not fftbins, dtype=dtype)
elif window_name == 'general_gaussian':
    window = general_gaussian(win_length, p=param[0], sig=param[1], sym=not fftbins, dtype=dtype)
elif window_name == 'exponential':
    window = torch.signal.windows.exponential(win_length, center=param[0], tau=param[1], sym=not fftbins, dtype=dtype)
elif window_name == 'triang':
    window = triang(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'bohman':
    window = bohman(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'blackman':
    window = torch.signal.windows.blackman(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'cosine':
    window = torch.signal.windows.cosine(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'tukey':
    window = tukey(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'taylor':
    window = taylor(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'bartlett':
    window = torch.signal.windows.bartlett(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'kaiser':
    window = torch.signal.windows.kaiser(win_length, beta=param[0], sym=not fftbins, dtype=dtype)
elif window_name == 'nuttall':
    window = torch.signal.windows.nuttall(win_length, sym=not fftbins, dtype=dtype)
result = window
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


# h


# i

class IsEmptyRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
result = x.numel() == 0
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)

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


class ItemRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
x = next(iter(kwargs.values()))
args = locals().get('args')
if args:
    if len(args) == 1:
        x = x.flatten()
    result = x[*args].item()
else:
    result = x.item()
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


# j


# k


# l
class LcmRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
x, y = torch.broadcast_tensors(x, y)
x_abs = torch.abs(x)
y_abs = torch.abs(y)
gcd = torch.gcd(x_abs, y_abs)
lcm = torch.zeros_like(gcd)
nonzero_mask = gcd != 0
lcm[nonzero_mask] = (x_abs[nonzero_mask] * y_abs[nonzero_mask]) // gcd[nonzero_mask]
result = torch.abs(lcm)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


# m
class Matrix_transposeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
result = x.transpose(-1, -2)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)
      
class MedianRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
axis = locals().get('axis')
keepdim = locals().get('keepdim', False)
mode = locals().get('mode', 'avg')
if axis is None:
    x_flat = x.flatten()
    length = x_flat.numel()
    if length % 2 == 0 and mode == 'avg':
        sorted_x = torch.sort(x_flat).values
        mid = length // 2
        median = (sorted_x[mid - 1] + sorted_x[mid]) / 2
    else:
        median = torch.median(x_flat)
else:
    if mode == 'avg':
        length = x.shape[axis] if x.ndim > 0 else 1
        if length % 2 == 0:
            sorted_x = torch.sort(x, dim=axis).values
            mid = length // 2
            median = (sorted_x.index_select(axis, torch.tensor([mid - 1])) + 
                      sorted_x.index_select(axis, torch.tensor([mid]))) / 2
            if not keepdim:
                median = median.squeeze(axis)
        else:
            median = torch.median(x, dim=axis, keepdim=keepdim).values
    else:
        median = torch.median(x, dim=axis, keepdim=keepdim)
result = median
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class MultiplexRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
input = locals().get("inputs")
index = locals().get("index")
temp = []
for i in range(index.shape[0]):
    j = index[i].item()
    temp.append(input[j][i])
result = torch.stack(temp)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


# n
class NanmedianRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
axis = locals().get("axis")
keepdim = locals().get("keepdim", False)
mode = locals().get("mode", "avg")

def single_axis_nanmedian(x, axis, keepdim, mode):
    if mode == "avg":
        valid_mask = ~torch.isnan(x)
        if x.ndim == 0:
            valid_x = x.masked_select(valid_mask).reshape(1)
            length = valid_x.numel()
        else:
            valid_x = x.masked_select(valid_mask).reshape(
                *[s if i != axis else -1 for i, s in enumerate(x.shape)]
            )
            length = valid_x.shape[axis]
        if length % 2 == 0:
            sorted_x = torch.sort(valid_x, dim=axis).values
            non_nan_mask = ~torch.isnan(sorted_x)
            sorted_x = sorted_x.masked_select(non_nan_mask).reshape(
                *[s if i != axis else -1 for i, s in enumerate(sorted_x.shape)]
            )
            mid = length // 2
            median = (
                sorted_x.index_select(axis, torch.tensor([mid - 1]))
                + sorted_x.index_select(axis, torch.tensor([mid]))
            ) / 2
            if not keepdim:
                median = median.squeeze(axis)
        else:
            median = torch.nanmedian(x, dim=axis, keepdim=keepdim).values
    else:
        median = torch.nanmedian(x, dim=axis, keepdim=keepdim)
    return median

if axis is None:
    x = x.flatten()
    valid_mask = ~torch.isnan(x)
    valid_x = x[valid_mask]
    length = valid_x.numel()
    if length % 2 == 0 and mode == "avg":
        sorted_x = torch.sort(valid_x).values
        mid = length // 2
        median = (sorted_x[mid - 1] + sorted_x[mid]) / 2
    else:
        median = torch.nanmedian(x)
elif isinstance(axis, int):
    median = single_axis_nanmedian(x, axis, keepdim, mode)
else:
    axes = [ax % x.ndim for ax in axis]
    non_axes = [i for i in range(x.ndim) if i not in axes]
    perm = non_axes + list(axes)
    x_permuted = x.permute(perm)
    non_axes_shape = [x.shape[i] for i in non_axes]
    flattened_size = 1
    for ax in axes:
        flattened_size *= x.shape[ax]
    new_shape = non_axes_shape + [flattened_size]
    x_flat = x_permuted.reshape(new_shape)
    median = single_axis_nanmedian(x_flat, -1, False, mode)
    if mode == "min":
        median = median.values
    if keepdim:
        output_shape = [1 if i in axes else x.shape[i] for i in range(x.ndim)]
        median = median.reshape(output_shape)
result = median
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class NmsRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
import torchvision
class scores_pair:
    def __init__(self, scores, index):
        self.scores = scores
        self.index = index
scores = locals().get('scores', None)
top_k = locals().get('top_k', None)
category_idxs = locals().get('category_idxs', None)
category = locals().get('categories', None)
iou_threshold = locals().get('iou_threshold', 0.3)

# 没有scores时自行生成scores
if scores is None:
    scores = torch.arange(1,0,(0.-1.)/boxes.shape[0])
    scores = scores[:boxes.shape[0]]

# 存在category时, 按照类别进行nms
if category_idxs is not None:
    result = []
    for cls in category:
        sele = []
        for i in range(len(category_idxs)):
            if category_idxs[i] == cls:
                sele.append(i)
        box = boxes.index_select(0, torch.tensor(sele))
        score = scores.index_select(0, torch.tensor(sele))
        result.append(torchvision.ops.nms(box, score, iou_threshold))
    result = torch.concat(result) 
else:
    result = torchvision.ops.nms(boxes, scores, iou_threshold)

# 对结果从大到小进行排序输出
ind = []
scores = scores.index_select(0,result)
for j in range(scores.numel()):
    tmp = scores_pair(scores[j], j)
    ind.append(tmp)
ind = sorted(ind, key = lambda x : x.scores, reverse = True)
for j in range(len(ind)):
    ind[j] = ind[j].index
result = result.index_select(0, torch.tensor(ind))
if top_k is not None:
    result = result[:top_k]
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class NumelRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
num_elements = x.numel()
result = torch.tensor(num_elements, dtype=torch.int64)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


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
if reduce == 'mul':
    reduce = 'prod'
include_self = locals().get('include_self', True)
broadcast = locals().get('broadcast', True)

def infer_broadcast_shape(input, index, dim):
    broadcast_shape_list = list(input.shape)
    broadcast_shape_list[dim] = list(index.shape)[dim]
    broadcast_shape = tuple(broadcast_shape_list)
    for i in range(len(input.shape)):
        if input.shape[i] < index.shape[i]:
            # if indices matrix has larger size than arr matrix, do not broadcast.
            return None
    return broadcast_shape

if broadcast == True:
    broadcast_shape = infer_broadcast_shape(arr, indices, axis)
    if broadcast_shape:
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
class QrRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
mode = locals().get('mode', 'reduced')
result = torch.linalg.qr(x, mode)
if mode == "r":
    result = result[1]
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code, "result")

# r
class RankRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = f"result = torch.tensor(input.dim(),dtype=torch.int64)"
        code = Code(
            core=core.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


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
        code.append(f"result = {self.torch_api}(boxes = ans, **_kwargs)")  # type: ignore
        return ConvertResult.success(paddle_api, code, "result")


# s
class SampleNeighborsRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
eids = locals().get('eids', None)
sample_size = locals().get('sample_size', -1)
return_ids = locals().get('return_eids', False)
out_neighbors = []
out_count = []
out_eids = []
for node in input_nodes:
    start = colptr[node]
    end = colptr[node + 1]        
    neighbors = row[start:end]
    num_neighbors = neighbors.numel()
    edge_ids = torch.arange(start,end,dtype=torch.int64)

    if num_neighbors == 0:
        sampled = torch.tensor([], dtype=row.dtype)
        sampled_eids = torch.tensor([], dtype=torch.int64)
    elif sample_size == -1 or num_neighbors <= sample_size:
        sampled = neighbors
        sampled_eids = edge_ids
    else:
        sampled = neighbors[:sample_size]
        sampled_eids = edge_ids[:sample_size]

    out_neighbors.append(sampled)
    out_count.append(sampled.numel())
    out_eids.append(sampled_eids)

out_neighbors = torch.cat(out_neighbors) if out_neighbors else torch.tensor([], dtype=row.dtype)
out_count = torch.tensor(out_count, dtype=torch.int64)
if return_ids:
    out_eids = eids.index_select(0,torch.cat(out_eids)) if out_eids else torch.tensor([], dtype=eids.dtype)

if return_ids:
    result = (out_neighbors, out_count, out_eids)
else:
    result = (out_neighbors, out_count)

"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class SegmentMaxRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
num = int(segment_ids.max().item()) + 1
ans = torch.full((num,)+data.shape[1:], float('-inf'), dtype = data.dtype)
for idx in range(data.shape[0]): 
    seg_id = segment_ids[idx]
    val = data[idx]
    ans[seg_id][val > ans[seg_id]] = val[val > ans[seg_id]]
ans[ans == float('-inf')] = 0
result = ans
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class ScatterRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
overwrite = locals().get('overwrite', True)
x = x.clone()
index = index.view(-1, 1)
try:
    updates = updates.expand_as(x)
except:
    pass
if not overwrite:
    for i in range(index.shape[0]):
        x[index[i]] = torch.zeros_like(x[index[i]])
    for i in range(index.shape[0]):
        x[index[i]] += updates[i]
else:
    for i in range(index.shape[0]):
        x[index[i]] = updates[i]
result = x    
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class ScatterndRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
output = torch.zeros(shape, dtype=updates.dtype).to(updates.device)
if index.numel() == 0:
    result = output + updates
else:
    flat_index = index.view(-1, index.size(-1))
    flat_updates = updates.reshape(flat_index.size(0), *updates.shape[index.dim()-1:])
    for i in range(flat_index.size(0)):
        idx_tuple = tuple(flat_index[i])
        output[idx_tuple] += flat_updates[i]
    result = output    
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class ScatterndaddRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
x = x.clone()
if index.numel() == 0:
    result = x + updates
else:
    flat_index = index.view(-1, index.size(-1))
    flat_updates = updates.reshape(flat_index.size(0), *updates.shape[index.dim()-1:])
    for i in range(flat_index.size(0)):
        idx_tuple = tuple(flat_index[i])
        x[idx_tuple] += flat_updates[i]
    result = x    
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)

class SortRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
axis = locals().get('axis', -1)
descending = locals().get('descending', False)  
stable = locals().get('stable', False)  
axis = axis if axis >= 0 else x.dim() + axis
result, _ = torch.sort(x, dim=axis, descending=descending, stable=stable)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)

class SortTensorRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
axis = locals().get('axis', -1)
descending = locals().get('descending', False)  
stable = locals().get('stable', False)  
axis = axis if axis >= 0 else x.dim() + axis
result, _ = x.sort(dim=axis, descending=descending, stable=stable)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)

class SlogdetRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
result = torch.linalg.slogdet(x)
result = torch.stack(result,0)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)

# t
class TriangularSolveRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
transpose = locals().get('transpose', False)
upper = locals().get('upper', True)
unitriangular = locals().get('unitriangular', False)
if transpose:
    x = x.transpose(-1,-2)
result = torch.linalg.solve_triangular(x,y,upper=upper,left=True,unitriangular=unitriangular)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)

# u


# v
class ViewRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
result = x.view(shape_or_dtype)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


# w


# x


# y


# z


# __
class __Pow__Rule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
tensor = locals().get('self')
other = locals().get('y')
result = tensor.__pow__(other)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)

class __rshift__Rule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
tensor = locals().get('x')
other = locals().get('y')
is_arithmetic = locals().get('is_arithmetic')
# setting default value for is_arithmetic
if is_arithmetic is None:
    is_arithmetic = True

def logical_right_shift(x: torch.Tensor, y: torch.Tensor):
    mask = (1 << (x.element_size() * 8 - 1)) - 1
    x_arithmetic, mask = x >> y, mask >> (y - 1)
    shifted = torch.where(y >= 1, x_arithmetic & mask, x)
    shifted = torch.where(y < 0, torch.zeros_like(x), shifted)
    return shifted
    
if is_arithmetic:
    result = tensor.__rshift__(other)
else:
    # logical right shift 
    result = logical_right_shift(tensor, other)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)
    
__all__ = [  # type: ignore
    cls.__name__
    for cls in globals().values()
    if isinstance(cls, type) and issubclass(cls, BaseRule) and cls != BaseRule
]

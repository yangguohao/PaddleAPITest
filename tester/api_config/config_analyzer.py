import random
import re
import collections
import paddle
import numpy
import math
import json
import paddle
import inspect
import torch
import copy


USE_CACHED_NUMPY = False
cached_numpy = {}

not_zero_apis = [
    "paddle.Tensor.__div__",
    "paddle.Tensor.__floordiv__",
    "paddle.Tensor.__rdiv__",
    "paddle.Tensor.__rfloordiv__",
    "paddle.Tensor.__rtruediv__",
    "paddle.Tensor.__truediv__",
    "paddle.Tensor.divide",
    "paddle.Tensor.floor_divide",
    "paddle.divide",
    "paddle.floor_divide",
    "paddle.nn.functional.kl_div",
    "paddle.sparse.divide",
    "paddle.Tensor.__mod__",
    "paddle.Tensor.__rmod__",
    "paddle.Tensor.floor_mod",
    "paddle.Tensor.mod",
    "paddle.floor_mod",
    "paddle.mod",
]

class TensorConfig:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.numpy_tensor = None
        self.paddle_tensor = None
        self.torch_tensor = None
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.shape = copy.deepcopy(self.shape)
        result.dtype = copy.deepcopy(self.dtype)
        return result

    def __str__(self):
        return "Tensor("+str(self.shape)+",\""+self.dtype+"\")"
    def __repr__(self):
        return "Tensor("+str(self.shape)+",\""+self.dtype+"\")"

    def convert_dtype_to_torch_type(self, dtype):
        if dtype in ["float32", numpy.float32]:
            return torch.float32
        elif dtype in ['float16', numpy.float16]:
            return torch.float16
        elif dtype in ['float64', numpy.float64]:
            return torch.float64
        elif dtype in ['int16', numpy.int16]:
            return torch.int16
        elif dtype in ['int8', numpy.int8]:
            return torch.int8
        elif dtype in ['bool', numpy.bool_]:
            return torch.bool
        elif dtype in ['bfloat16', numpy.uint16]:
            return torch.bfloat16
        elif dtype in ['uint8', numpy.uint8]:
            return torch.uint8
        elif dtype in ['int32', numpy.int32]:
            return torch.int32
        elif dtype in ['int64', numpy.int64]:
            return torch.int64
        elif dtype in ['complex64', numpy.complex64]:
            return torch.complex64
        elif dtype in ['complex128', numpy.complex128]:
            return torch.complex128
        else:
            raise ValueError(f'Unsupport dtype: {dtype}')

    def numel(self):
        numel = 1
        for i in self.shape:
            numel = numel * i
        return numel

    def get_cached_numpy(self, dtype, shape):
        numel = 1
        for i in shape:
            numel = numel * i

        if dtype in cached_numpy:
            tensor = cached_numpy[dtype][:numel].reshape(shape)
        else:
            if "int" in dtype:
                cached_numpy[dtype] = numpy.random.randint(-65535, 65535, size=4300000000, dtype="int64").astype(dtype)
                tensor = cached_numpy[dtype][:numel].reshape(shape)
            else:
                cached_numpy[dtype] = (numpy.random.random([4300000000]) - 0.5).astype(dtype)
                tensor = cached_numpy[dtype][:numel].reshape(shape)
        return tensor

    def generate_random_axes(self, api_config):
        x_shape = self.get_arg(api_config, 0, "x").shape
        max_dim = max(len(x_shape), 1) # scalar

        if len(self.shape) == 0:
            dim = numpy.random.randint(0, max_dim)
            if numpy.random.rand() > 0.5:
                dim -= max_dim
            return numpy.array(dim, dtype=self.dtype)

        if len(self.shape) == 1:
            dims = numpy.random.choice(max_dim, size=self.shape[0], replace=False)
            mask = numpy.random.rand(self.shape[0]) > 0.5
            dims = numpy.where(mask, dims - max_dim, dims)
            return numpy.array(dims, dtype=self.dtype)

        raise ValueError(
            f"Invalid shape for 'axis' Tensor in {api_config.api_name}. "
            f"Expected a 0-D or 1-D Tensor, but got shape {self.shape}."
        )

    def generate_random_index(self, api_config, allow_none=False):
        axis = self.get_arg(api_config, 2, "axis")
        if axis is None and not allow_none:
            raise ValueError("Axis is None")
        else:
            axis = 0
        x_shape = self.get_arg(api_config, 0, "x").shape
        axis = axis if axis >= 0 else axis + len(x_shape)
        if not (0 <= axis < len(x_shape)):
            raise ValueError(f"Invalid axis {axis} for shape {x_shape}")

        if len(self.shape) >= 1:
            return numpy.random.randint(0, x_shape[axis], size=self.shape, dtype=self.dtype)

        raise ValueError(
            f"Invalid shape for 'index' Tensor in {api_config.api_name}. "
            f"Expected a 0-D or 1-D Tensor, but got shape {self.shape}."
        )

    def get_numpy_tensor(self, api_config, index=None, key=None, **kwargs):
        if index is not None:
            self.index = index
        if key is not None:
            self.key = key

        if self.dtype in ["float8_e5m2", "float8_e4m3fn"]:
            print("Warning ", self.dtype, "not supported")
            return
        if self.numpy_tensor is None:
            if api_config.api_name in not_zero_apis:
                if "int" in self.dtype:
                    self.numpy_tensor = (numpy.random.randint(1, 65535, size=self.shape)).astype(self.dtype)
                else:
                    dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                    self.numpy_tensor = (numpy.random.random(self.shape) + 0.5).astype(dtype)
            # a
            elif api_config.api_name in ["paddle.arange"]:
                tensor_num = 0
                for arg in api_config.args:
                    if "Tensor" in str(arg):
                        tensor_num += 1
                if tensor_num == 3 or "step" in api_config.kwargs:
                    if self.check_arg(api_config,2,"step"):
                        if "int" in self.dtype:
                            x = numpy.random.random()
                            if x > 0.5:
                                self.numpy_tensor = (numpy.random.randint(1, 65535, size=self.shape)).astype(self.dtype)
                            else:
                                self.numpy_tensor = (numpy.random.randint(-65536, -1, size=self.shape)).astype(self.dtype)
                        else:
                            dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                            x = numpy.random.random()
                            if x > 0.5:
                                self.numpy_tensor = (numpy.random.random(self.shape) + 1.0).astype(dtype)
                            else:
                                self.numpy_tensor = (numpy.random.random(self.shape) - 2.0).astype(dtype) 

            elif api_config.api_name in ["paddle.argmax","paddle.argmin"]:  
                if  self.check_arg(api_config, 1, "axis"):
                    arr=self.get_arg(api_config,0,'x')                
                    min_dim = numpy.min(arr.shape)
                    indices = (numpy.random.randint(0, min_dim-1, size=self.numel())).astype("int64")
                    self.numpy_tensor = indices.reshape(self.shape)
                    self.dtype = "int64"

            elif api_config.api_name in ["paddle.atan2"]:
                s1=self.get_arg(api_config,0)
                s2=self.get_arg(api_config,1)
                s1=s1.shape
                s2=s2.shape
                if numpy.max(s1) == 0 and max(s2) == 0:
                    while len(s1)>len(s2):
                        s2.append(0)
                    while len(s2)>len(s1):
                        s1.append(0)
                self.numpy_tensor=numpy.random.random(s1)
            # b
            elif api_config.api_name in ["paddle.bernoulli"]:
                if self.check_arg(api_config, 0, "x"):
                    dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                    self.numpy_tensor = numpy.random.random(self.shape).astype(dtype)
            elif api_config.api_name in ["paddle.bincount"]:
                if self.check_arg(api_config, 0, "x"):
                    if "int" in self.dtype:
                        self.numpy_tensor = numpy.random.randint(0, 65535, size=self.shape).astype(self.dtype)
                    else:
                        raise ValueError(f"The input of paddle.bincount must be of integer type, but the current type is {self.dtype}")
                elif self.check_arg(api_config, 2, "minlength") or self.check_arg(api_config, None, "minlength"):
                    if "int" in self.dtype:
                        self.numpy_tensor = numpy.random.randint(0, 65535, size=self.shape).astype(self.dtype)
                    else:
                        dtype = "int64"
                        self.numpy_tensor = numpy.random.randint(0, 65535, size=self.shape).astype(dtype)
                        self.dtype = dtype
            # c
            elif api_config.api_name in ["paddle.chunk"]:
                if self.check_arg(api_config, 2, "axis"):
                    x_tensor = self.get_arg(api_config, 0, "x")
                    chunks = self.get_arg(api_config, 1, "chunks")
                    valid_axes = []
                    for i, dim_size in enumerate(x_tensor.shape):
                        if dim_size % chunks == 0:
                            valid_axes.append(i)
                    if not valid_axes:
                        valid_axes = [0]
                    chosen_axis = random.choice(valid_axes)
                    if len(self.shape) == 0:  
                        self.numpy_tensor = numpy.array(chosen_axis, dtype=self.dtype)
                    elif len(self.shape) == 1:  
                        if self.shape[0] == 1:
                            self.numpy_tensor = numpy.array([chosen_axis], dtype=self.dtype)
                        else:
                            raise ValueError(
                                f"Invalid shape for 'axis' Tensor in paddle.chunk. "
                                f"Expected a 0-D or 1-D Tensor with 1 element, but got shape {self.shape}."
                            )
                    else:
                        raise ValueError(
                            f"Invalid shape for 'axis' Tensor in paddle.chunk. "
                            f"Expected a 0-D or 1-D Tensor, but got shape {self.shape}."
                        )

            elif api_config.api_name in ["paddle.nn.functional.conv2d_transpose"]:
                if index is not None and index == 0 or key == "x":
                    if not hasattr(api_config, "x"):
                        if "int" in self.dtype:
                            self.numpy_tensor = (numpy.random.randint(-65535, 65535, size=self.shape)).astype(self.dtype)
                        else:
                            dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                            self.numpy_tensor = (numpy.random.random(self.shape) - 0.5).astype(dtype)
                        api_config.x = self.numpy_tensor
                elif index is not None and index == 1 or key =="weight":
                    if not hasattr(api_config, "weight"):
                        if "int" in self.dtype:
                            self.numpy_tensor = (numpy.random.randint(-65535, 65535, size=self.shape)).astype(self.dtype)
                        else:
                            dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                            self.numpy_tensor = (numpy.random.random(self.shape) - 0.5).astype(dtype)
                        api_config.weight = self.numpy_tensor     
                elif index is not None and index == 2 or key =="bias":
                    if not hasattr(api_config, "bias"):
                        if "int" in self.dtype:
                            self.numpy_tensor = (numpy.random.randint(-65535, 65535, size=self.shape)).astype(self.dtype)
                        else:
                            dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                            self.numpy_tensor = (numpy.random.random(self.shape) - 0.5).astype(dtype)
                        api_config.bias = self.numpy_tensor
                elif key == "output_size":
                    if not hasattr(api_config,"bias"):
                        bias = None
                    else:
                        bias = paddle.to_tensor(api_config.bias)
                    if "stride" in api_config.kwargs:
                        stride = api_config.kwargs["stride"]
                    else:
                        stride = 1
                    if "padding" in api_config.kwargs:
                        padding = api_config.kwargs["padding"]
                    else:
                        padding = 0
                    if "dilation" in api_config.kwargs:
                        dilation = api_config.kwargs["dilation"]
                    else:
                        dilation = 1
                    if "groups" in api_config.kwargs:
                        groups = api_config.kwargs["groups"]
                    else:
                        groups = 1
                    if "output_padding" in api_config.kwargs:
                        output_padding = api_config.kwargs["output_padding"]
                    else:
                        output_padding = 0
                    if "data_format" in api_config.kwargs:
                        data_format = api_config.kwargs["data_format"]
                    else:
                        data_format = "NCHW"
                        
                    output_size = paddle.nn.functional.conv2d_transpose(paddle.to_tensor(api_config.x),paddle.to_tensor(api_config.weight),bias = bias, \
                                                                        stride = stride, padding = padding, output_padding = output_padding, \
                                                                        groups = groups, dilation = dilation, data_format = data_format)
                    
                    
                    last = [0,0]
                    last[0] = output_size.shape[data_format.find('H')]
                    last[1] = output_size.shape[data_format.find('W')] 
                    s = [1,1]
                    if isinstance(stride,int):
                        s[0] = stride
                        s[1] = stride
                    else:
                        s = stride
                    self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                    self.numpy_tensor[0] = numpy.random.randint(last[0],last[0]+s[0])
                    self.numpy_tensor[1] = numpy.random.randint(last[1],last[1]+s[1])
                    return self.numpy_tensor
                
            elif api_config.api_name in ["paddle.cumsum"] and self.check_arg(api_config, 1, "axis"):
                # special args[1] tensor init, for the rest reuse default initialization logic
                x_tensor_config = self.get_arg(api_config, 0, "x")
                len_shape = len(x_tensor_config.shape)
                self.numpy_tensor = numpy.random.randint(-len_shape, len_shape, size=self.shape)
                return self.numpy_tensor
            elif api_config.api_name in ["paddle.clip"] and self.check_arg(api_config, 0, "x"):
                # init input tensor x randomly (index == 0 indicates we are init TensorConfig(x).numpy_tennor)
                self.numpy_tensor = self.get_random_numpy_tensor(shape=self.shape, data_type=self.dtype)
                
                # if both min and max need a Tensor instead of None, init min and max at the same TensorConfig numpy tensor init process
                min_config = self.get_arg(api_config, 1, "min")
                max_config = self.get_arg(api_config, 2, "max")
                if (isinstance(min_config, TensorConfig) and isinstance(max_config, TensorConfig)):
                    min_shape = min_config.shape
                    min_dtype = min_config.dtype
                    min = self.get_random_numpy_tensor(shape=min_shape, data_type=min_dtype)

                    max_shape = max_config.shape
                    max_dtype = max_config.dtype
                    max = self.get_random_numpy_tensor(shape=max_shape, data_type=max_dtype, min=min)
                    
                    self.set_tensor_arg_value(api_config, 1, "min", min)
                    self.set_tensor_arg_value(api_config, 2, "max", max)
                elif min_config is not None and max_config is not None:
                    # min and max args are specified but at least one of them is scalar (not a TensorConfig)
                    # according to API DOC, min and max is float|int|Tensor
                    if isinstance(min_config, TensorConfig) and (isinstance(max_config, int) or isinstance(max_config, float)):
                        min_shape = min_config.shape
                        min_dtype = min_config.dtype
                        min = self.get_random_numpy_tensor(shape=min_shape, data_type=min_dtype, max=max_config)
                        self.set_tensor_arg_value(api_config, 1, "min", min)
                    elif (isinstance(max_config, TensorConfig) and (isinstance(min_config, int) or isinstance(min_config, float))):
                        max_shape = max_config.shape
                        max_dtype = max_config.dtype
                        max = self.get_random_numpy_tensor(shape=max_shape, data_type=max_dtype, min=min_config)
                        self.set_tensor_arg_value(api_config, 2, "max", max)
                    # for both min and max are scalar, there is no need to init numpy tensor

                return self.numpy_tensor
            # d
            # e
            elif api_config.api_name in ["paddle.empty"]:
                is_shape_param = False
                if len(api_config.args) > 0:
                    if self.check_arg(api_config, 0, "shape"):
                        is_shape_param = True
                    elif isinstance(api_config.args[0], list):
                        for item in api_config.args[0]:
                            if str(item) == str(self):
                                is_shape_param = True
                                break
                if "shape" in api_config.kwargs:
                    if str(api_config.kwargs["shape"]) == str(self):
                        is_shape_param = True
                    elif isinstance(api_config.kwargs["shape"], list):
                        for item in api_config.kwargs["shape"]:
                            if str(item) == str(self):
                                is_shape_param = True
                                break
                if is_shape_param:
                    if "int" in self.dtype:
                        self.numpy_tensor = numpy.random.randint(1, 10, size=self.shape).astype(self.dtype)
                    else:
                        dtype = "int32"
                        self.numpy_tensor = numpy.random.randint(1, 10, size=self.shape).astype(dtype)
                        self.dtype = dtype 
            elif api_config.api_name in ["paddle.eye"]:
                self.numpy_tensor = numpy.random.randint(0, 2048, size = self.shape)

            elif api_config.api_name in ["paddle.expand","paddle.Tensor.expand"]:
                if key == "shape" or index == 1:
                    d=self.get_arg(api_config, 0, "x")
                    s=d.shape
                    if 'list_index' in kwargs:
                        ind=kwargs['list_index'][0]
                    else:
                        ind=0
                    if len(s)==0 or s[ind]==1:
                        self.numpy_tensor = (numpy.random.randint(1, 127, size=self.shape)).astype(self.dtype)
                    else:
                        if len(self.shape)==0 or self.shape[0]==1:
                            self.numpy_tensor = numpy.array(s[ind])
                        else:
                            self.numpy_tensor = (numpy.random.randint(1, 127, size=self.shape)).astype(self.dtype)
                            dis=self.shape[0]-len(s)
                            for i in range(self.shape[0]):
                                if i>=dis and s[i-dis]!=1:
                                    self.numpy_tensor[i]=s[i-dis]

            elif api_config.api_name in ["paddle.expand_as"]:
                if self.dtype=='float16':
                    self.dtype='float32'
            # f
            elif api_config.api_name in ["paddle.full"]:
                if self.check_arg(api_config, 1, "fill_value"):
                    if "int" in dtype:
                        self.numpy_tensor = (numpy.random.randint(1, 65535, size=self.shape)).astype(self.dtype)
                    else:
                        dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                        self.numpy_tensor = (numpy.random.random(self.shape) + 0.5).astype(dtype)
                else:
                    self.numpy_tensor = (numpy.random.randint(0, 2048, size=self.shape)).astype(self.dtype)

            # g
            elif api_config.api_name in ["paddle.gammainc", "paddle.gammaincc"]:
                if "int" in self.dtype:
                    self.numpy_tensor = numpy.random.randint(0, 65535, size=self.shape).astype(self.dtype)
                else:
                    dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                    self.numpy_tensor = numpy.abs(numpy.random.random(self.shape)).astype(dtype)
            elif api_config.api_name.startswith("paddle.geometric.segment_"):
                if self.check_arg(api_config, 1, "segment_ids"):
                    batch_size = self.get_arg(api_config, 0, "x").shape[0]
                    max_segments = numpy.random.randint(1, batch_size + 1)
                    self.numpy_tensor = numpy.sort(
                        numpy.random.randint(0, max_segments, size=self.shape).astype(self.dtype)
                    )
            elif api_config.api_name.startswith("paddle.geometric.send_u"):
                if self.check_arg(api_config, 1, "src_index") or self.check_arg(api_config, 2, "dst_index"):
                    num_nodes = self.get_arg(api_config, 0, "x").shape[0]
                    self.numpy_tensor = numpy.random.randint(0, num_nodes, size=self.shape).astype(self.dtype)
            # h
            # i
            elif api_config.api_name in ["paddle.index_add", "paddle.index_fill"]:
                if self.check_arg(api_config, 1, "index"):
                    self.numpy_tensor = self.generate_random_index(api_config)
            elif api_config.api_name in ["paddle.index_sample"]:
                if self.check_arg(api_config, 1, "index"):
                    x_dim = self.get_arg(api_config, 0, "x").shape[1]
                    self.numpy_tensor = numpy.random.randint(0, x_dim, size=self.shape)
            elif api_config.api_name in ["paddle.index_select"]:
                if self.check_arg(api_config, 1, "index"):
                    self.numpy_tensor = self.generate_random_index(api_config, allow_none=True)
            elif api_config.api_name.startswith("paddle.incubate.segment_"):
                if self.check_arg(api_config, 1, "segment_ids"):
                    batch_size = self.get_arg(api_config, 0, "x").shape[0]
                    max_segments = numpy.random.randint(1, batch_size + 1)
                    self.numpy_tensor = numpy.sort(
                        numpy.random.randint(0, max_segments, size=self.shape).astype(self.dtype)
                    )
            # j
            # k
            # l
            elif api_config.api_name in ["paddle.logspace"]:
                if self.check_arg(api_config, 2, "num"):
                    self.numpy_tensor = numpy.random.randint(1, 65535, size=self.shape)
            elif api_config.api_name.startswith("paddle.linalg."):
                if api_config.api_name.endswith("cholesky"):
                    if self.check_arg(api_config, 0, "x"):
                        if len(self.shape) < 2 or self.shape[-1] != self.shape[-2]:
                            raise ValueError("Shape must have at least 2 dimensions and last two dimensions must be equal")
                        batch_dims = self.shape[:-2]
                        matrix_dim = self.shape[-1]
                        A = numpy.random.random(batch_dims + [matrix_dim, matrix_dim]).astype(self.dtype)
                        if len(batch_dims) > 0:
                            tensor = numpy.einsum('...ij,...kj->...ik', A, A)
                        else:
                            tensor = numpy.dot(A, A.T)
                        tensor += numpy.eye(matrix_dim, dtype=self.dtype) * 1e-6
                        print("cholesky tensor", tensor)
                        self.numpy_tensor = tensor
                elif api_config.api_name.endswith("cov"):
                    if self.check_arg(api_config, 0, "x"):
                        if len(self.shape) < 1 or len(self.shape) > 2:
                            raise ValueError("Shape must have 1 or 2 dimensions for covariance input")
                        tensor = numpy.random.random(self.shape).astype(self.dtype)
                        tensor += numpy.random.random(self.shape).astype(self.dtype) * 1e-6
                        self.numpy_tensor = tensor
                    elif self.check_arg(api_config, 3, "fweights"):
                        x_shape = self.get_arg(api_config, 0, "x").shape
                        rowvar = self.get_arg(api_config, 1, "rowvar")
                        if rowvar is None:
                            rowvar = True
                        n_observations = (x_shape[1] if rowvar else x_shape[0]) if len(x_shape) > 1 else x_shape[0]
                        self.numpy_tensor = numpy.random.randint(1, 11, size=(n_observations,)).astype(self.dtype)
                    elif self.check_arg(api_config, 4, "aweights"):
                        x_shape = self.get_arg(api_config, 0, "x").shape
                        rowvar = self.get_arg(api_config, 1, "rowvar")
                        if rowvar is None:
                            rowvar = True
                        n_observations = (x_shape[1] if rowvar else x_shape[0]) if len(x_shape) > 1 else x_shape[0]
                        if self.dtype in ["float32", "float64"]:
                            self.numpy_tensor = numpy.random.uniform(0.1, 1.0, size=(n_observations,)).astype(self.dtype)
                        else:
                            self.numpy_tensor = numpy.random.randint(1, 11, size=(n_observations,)).astype(self.dtype)
                elif api_config.api_name.endswith("eigh"):
                    if self.check_arg(api_config, 0, "x"):
                        if len(self.shape) < 2 or self.shape[-1] != self.shape[-2]:
                            raise ValueError("Shape must have at least 2 dimensions and last two dimensions must be equal")
                        batch_dims = self.shape[:-2]
                        matrix_dim = self.shape[-1]
                        A = numpy.random.random(batch_dims + [matrix_dim, matrix_dim]).astype(self.dtype)
                        if self.dtype in ['complex64', 'complex128']:
                            A = A + 1j * numpy.random.random(batch_dims + [matrix_dim, matrix_dim]).astype(self.dtype)
                            tensor = A + A.swapaxes(-1, -2).conj()  # A + A^H
                        else:
                            if len(batch_dims) > 0:
                                tensor = numpy.einsum('...ij,...kj->...ik', A, A)
                            else:
                                tensor = numpy.dot(A, A.T)
                        tensor += numpy.eye(matrix_dim, dtype=self.dtype) * 1e-6
                        self.numpy_tensor = tensor
                elif api_config.api_name.endswith("lstsq"):
                    if self.check_arg(api_config, 0, "x") or self.check_arg(api_config, 1, "y"):
                        if len(self.shape) < 2:
                            raise ValueError("Shape must have at least 2 dimensions for lstsq x")
                        batch_dims = self.shape[:-2]
                        M, N = self.shape[-2], self.shape[-1]
                        self.numpy_tensor = numpy.random.random(batch_dims + [M, N]).astype(self.dtype)
                elif api_config.api_name.endswith("lu_unpack"):
                    if self.check_arg(api_config, 0, "x"):
                        if len(self.shape) < 2:
                            raise ValueError("Shape must have at least 2 dimensions for LU matrix")
                        batch_dims = self.shape[:-2]
                        LU_tensor = numpy.random.random(self.shape).astype(self.dtype)
                        K = min(self.shape[-2], self.shape[-1])
                        LU_tensor[..., range(K), range(K)] += 1e-6
                        self.numpy_tensor = LU_tensor
                    if self.check_arg(api_config, 1, "pivot"):
                        M = self.get_arg(api_config, 0, "x").shape[-2]
                        self.numpy_tensor = numpy.random.randint(1, M + 1, size=self.shape).astype(self.dtype)
                elif api_config.api_name.endswith("pca_lowrank"):
                    self.numpy_tensor = numpy.random.randn(*self.shape).astype(self.dtype)
            # m
            elif api_config.api_name in ["paddle.matrix_transpose"]:
                if self.check_arg(api_config, 0, "x"):
                    if len(self.shape) < 2:
                        matrix_shape = [2, 2]
                        if "int" in self.dtype:
                            self.numpy_tensor = numpy.random.randint(-65535, 65535, size=matrix_shape).astype(self.dtype)
                        else:
                            dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                            self.numpy_tensor = (numpy.random.random(matrix_shape) - 0.5).astype(dtype)
                    else:
                        if "int" in self.dtype:
                            self.numpy_tensor = (numpy.random.randint(-65535, 65535, size=self.shape)).astype(self.dtype)
                        else:
                            dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                            self.numpy_tensor = (numpy.random.random(self.shape) - 0.5).astype(dtype)
            elif api_config.api_name in ["paddle.mean", "paddle.max", "paddle.min"]:
                if self.check_arg(api_config, 1, "axis"):
                    self.numpy_tensor = self.generate_random_axes(api_config)

            elif api_config.api_name in ["paddle.multinomial"]:
                if self.check_arg(api_config, 0, 'x'):
                    self.numpy_tensor = numpy.abs(numpy.random.random(self.shape)).astype(self.dtype)
                
                if key == "num_samples" or index == 1:
                    if 'replacement' in api_config.kwargs and self.get_arg(api_config,2,'replacement')==True:
                        max_allow=1024
                    else:
                        inputs=self.get_arg(api_config,0,'x')
                        inputs=inputs.numpy_tensor
                        max_allow=(inputs > 0).sum().item()
                    self.numpy_tensor=numpy.random.randint(1,max_allow+1, size=self.shape).astype(self.dtype)
                    
            elif api_config.api_name in ["paddle.multiplex"]:
                s = self.get_arg(api_config, 0, 'inputs')
                if key == "index" or index == 1:
                    self.numpy_tensor = (numpy.random.randint(0,len(s), size=self.shape)).astype(self.dtype)

            elif api_config.api_name in ["paddle.multiply"]:
                if self.dtype=='bfloat16':
                    self.dtype='float32'    
                self.numpy_tensor = numpy.random.random(self.shape).astype(self.dtype)

            elif api_config.api_name in ["paddle.nn.functional.max_unpool1d", "paddle.nn.functional.max_unpool2d", "paddle.nn.functional.max_unpool3d"] and self.check_arg(api_config, 0, "x"):
                # use max_pool to generate legal max_unpool input
                kernel_size = self.get_initialized_value(api_config, 2, "kernel_size")
                stride = self.get_initialized_value(api_config, 3, "stride")
                padding = self.get_initialized_value(api_config, 4, "padding")
                padding = 0 if padding is None else padding
                stride = kernel_size if stride is None else stride
                unpool_output_size = self.get_initialized_value(api_config, 5, "output_size")
                pool_input_size = unpool_output_size
                
                ndim = 1
                if "max_unpool2d" in api_config.api_name:
                    ndim = 2
                elif "max_unpool3d" in api_config.api_name:
                    ndim = 3
                if isinstance(kernel_size, int): 
                    kernel_size = [kernel_size] * ndim
                if isinstance(stride, int):
                    stride = [stride] * ndim
                if isinstance(padding, int):
                    padding = [padding] * ndim
                    
                # if max_unpool output_size (max_pool input_size) is not set, calculate manually
                unpool_input_size = self.get_arg(api_config, 0, "x").shape
                pool_output_size = unpool_input_size
                if pool_input_size is None:
                    if ndim == 1:
                        w_in = pool_output_size[-1]
                        w_out = (w_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
                        pool_input_size = [*pool_output_size[:-1], w_out]
                    elif ndim == 2:
                        h_in, w_in = pool_output_size[-2], pool_output_size[-1]
                        h_out = (h_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
                        w_out = (w_in - 1) * stride[1] - 2 * padding[1] + kernel_size[1]
                        pool_input_size = [*pool_output_size[:-2], h_out, w_out]
                    else:
                        d_in, h_in, w_in = pool_output_size[-3], pool_output_size[-2], pool_output_size[-1]
                        d_out = (d_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
                        h_out = (h_in - 1) * stride[1] - 2 * padding[1] + kernel_size[1]
                        w_out = (w_in - 1) * stride[2] - 2 * padding[2] + kernel_size[2]
                        pool_input_size = [*pool_output_size[:-3], d_out, h_out, w_out]
                elif len(pool_input_size) == ndim:
                    # fill the lost dimensions since unpool_output_size has only last ndim dims
                    pool_input_size = [*pool_output_size[:-ndim], *pool_input_size[-ndim:]]
                elif len(pool_input_size) != len(pool_output_size):
                    raise ValueError(f"invalid argument output_size {pool_input_size} for {api_config.api_name}, len(output_size) should be {ndim} or {len(pool_output_size)} or output_size == None, got len(output_size)={len(pool_input_size)} and output_size={unpool_output_size}")
                    
                x = paddle.to_tensor(self.get_random_numpy_tensor(shape=pool_input_size, data_type=self.dtype))
                max_poolxd_func = eval(api_config.api_name.replace("max_unpool", "max_pool"))
                x, indices = max_poolxd_func(x, kernel_size, stride, padding, return_mask=True)
                self.numpy_tensor = x.numpy()
                self.set_tensor_arg_value(api_config, 1, "indices", indices)
                return self.numpy_tensor
                
            # n

            elif api_config.api_name in ["paddle.nn.functional.adaptive_avg_pool2d",'paddle.nn.functional.adaptive_avg_pool3d']:
                if key == "output_size" or index == 1:
                    s = self.get_arg(api_config, 0, "x")
                    s=s.shape
                    self.numpy_tensor = numpy.random.randint(1,2*numpy.max(s), size=self.shape).astype(self.dtype)
            elif api_config.api_name in ["paddle.nn.functional.adaptive_log_softmax_with_loss"]:
                if self.check_arg(api_config, 1, "label"):
                    cutoffs = self.get_arg(api_config, 4, "cutoffs")
                    if isinstance(cutoffs, list) and cutoffs:
                        n_classes = cutoffs[-1]
                    else:
                        n_classes = numpy.random.randint(5, 20)
                    if "int" not in self.dtype:
                        self.dtype = "int64"  
                    if len(self.shape) == 0:
                        self.shape = [1]
                    self.numpy_tensor = numpy.random.randint(0, n_classes, size=self.shape).astype(self.dtype)
            elif api_config.api_name in ['paddle.nn.functional.affine_grid']:
                if key == "out_shape" or index == 1:
                    s = self.get_arg(api_config, 0, "theta")
                    s = s.shape
                    self.numpy_tensor = numpy.random.randint(1,128, size=self.shape).astype(self.dtype)
                    self.numpy_tensor[0]=s[0]
            elif api_config.api_name in ['paddle.nn.functional.alpha_dropout']:
                if key == "x" or index == 0:
                    if self.dtype=='bfloat16':
                        self.dtype='float32'
                    self.numpy_tensor = numpy.random.random(self.shape).astype(self.dtype)

            elif api_config.api_name in ['paddle.nn.functional.interpolate']:
                if key == "size" or index == 1 or key == "scale_factor" or index == 2:
                    self.numpy_tensor = numpy.random.randint(1,128, size=self.shape).astype(self.dtype)
            
            elif api_config.api_name in ["paddle.nn.functional.gather_tree"]:
                if self.check_arg(api_config, 1, "parents"):
                    sequences = self.get_arg(api_config, 0, "sequences")
                    if hasattr(sequences, 'shape') and len(sequences.shape) >= 3:
                        beam_size = sequences.shape[2]
                    else:
                        beam_size = self.shape[2] if len(self.shape) >= 3 else 4
                    beam_size = 1 if beam_size < 1 else beam_size 
                    parents = numpy.zeros(self.shape, dtype=self.dtype)
                    for t in range(self.shape[0]):  
                        for b in range(self.shape[1]):  
                            for i in range(self.shape[2]): 
                                parents[t, b, i] = numpy.random.randint(0, beam_size)
                    self.numpy_tensor = parents
            
            elif api_config.api_name in ["paddle.nn.functional.gaussian_nll_loss"]:
                if self.check_arg(api_config, 2, "var"):
                    dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                    self.numpy_tensor = (numpy.random.random(self.shape) + 1.0).astype(dtype)

            elif api_config.api_name in ['paddle.nn.functional.grid_sample']:
                if self.dtype=='float16':
                    self.dtype='float32'
                    self.numpy_tensor = numpy.random.random(self.shape).astype(self.dtype)

            elif api_config.api_name in ['paddle.nn.functional.hsigmoid_loss']:
                nclass = self.get_arg(api_config, 2, "num_classes")
                weight = self.get_arg(api_config, 3, "weight")
                if key == "label" or index == 1:
                    self.numpy_tensor = numpy.random.randint(0,nclass, size=self.shape).astype(self.dtype)
                elif key == "path_table" or index == 5:
                    self.numpy_tensor = numpy.random.randint(0,weight.shape[0], size=self.shape).astype(self.dtype)
                elif key == "path_code" or index == 6:
                    self.numpy_tensor = numpy.random.randint(0,2, size=self.shape).astype(self.dtype)

            elif api_config.api_name in ['paddle.nn.functional.upsample']:
                if self.get_arg(api_config, 1, 'size'):
                    self.numpy_tensor = numpy.random.randint(1,128, size=self.shape).astype(self.dtype)
                if self.get_arg(api_config, 2, 'scale_factor'):
                    self.numpy_tensor = numpy.ones(self.shape).astype(self.dtype)+numpy.abs(numpy.random.random(self.shape)).astype(self.dtype)
            
            elif api_config.api_name in ['paddle.nn.functional.binary_cross_entropy']:
                if index==0 or key=='input':
                    self.numpy_tensor = numpy.random.rand(*self.shape).astype(self.dtype)
                elif index==1 or key=='label':
                    self.numpy_tensor = numpy.random.randint(0,2,size=self.shape).astype(self.dtype)

            elif api_config.api_name in ['paddle.nn.functional.margin_cross_entropy']:
                if index==1 or key=='label':
                    s=self.get_arg(api_config,0,'logits')
                    self.numpy_tensor = numpy.random.randint(0,s.shape[1], size=self.shape).astype(self.dtype)

            elif api_config.api_name in ['paddle.nn.functional.multi_margin_loss']:
                if index==1 or key=='label':
                    s=self.get_arg(api_config,0,'input')
                    self.numpy_tensor = numpy.random.randint(0,s.shape[1], size=self.shape).astype(self.dtype)
            # o
            elif api_config.api_name in ["paddle.ones"]:
                if len(self.shape) == 0:
                    self.numpy_tensor = numpy.array(random.randint(1, 2048), dtype=self.dtype)
                else:
                    self.numpy_tensor = numpy.random.randint(1, 65535, size=self.shape).astype(self.dtype)
            # p
            elif api_config.api_name in ["paddle.prod"]:
                if self.check_arg(api_config, 1, "axis"):
                    self.numpy_tensor = self.generate_random_axes(api_config)
            elif api_config.api_name in ["paddle.put_along_axis", "paddle.Tensor.put_along_axis"]:
                if self.check_arg(api_config, 1, "indices"):
                    x_tensor = self.get_arg(api_config, 0, "x")
                    x_dims = len(x_tensor.shape) if x_tensor.shape else 0
                    if len(self.shape) != x_dims:
                        new_shape = []
                        for i, dim in enumerate(x_tensor.shape):
                            if i < len(self.shape):
                                new_shape.append(self.shape[i])
                            else:
                                new_shape.append(1) 
                        indices = numpy.zeros(new_shape, dtype="int64")
                        for axis in range(x_dims):
                            if axis < len(self.shape):
                                dim_size = x_tensor.shape[axis]
                                if dim_size > 0:
                                    axis_indices = numpy.random.randint(0, dim_size, size=new_shape[axis])
                                    if new_shape[axis] > 1:
                                        axis_indices[0] = 0
                                        axis_indices[-1] = dim_size - 1
                                    idx_tuple = tuple([slice(None)] * axis + [slice(None, new_shape[axis])] + [slice(None)] * (x_dims - axis - 1))
                                    indices[idx_tuple] = axis_indices.reshape([-1] + [1] * (x_dims - axis - 1))
                        self.numpy_tensor = indices
                        self.shape = new_shape  
                    else:
                        axis = self.get_arg(api_config, 3, "axis")
                        axis = axis if isinstance(axis, int) else 0
                        axis = axis if axis >= 0 else axis + x_dims
                        if 0 <= axis < x_dims:
                            dim_size = x_tensor.shape[axis]
                            indices = numpy.random.randint(0, dim_size, size=self.shape).astype("int64")
                            if numpy.prod(self.shape) > 1:
                                flat_indices = indices.flatten()
                                flat_indices[0] = 0
                                flat_indices[-1] = dim_size - 1
                                indices = flat_indices.reshape(self.shape)
                            self.numpy_tensor = indices
                    self.dtype = "int64"
                elif self.check_arg(api_config, 2, "values"):
                    x_tensor = self.get_arg(api_config, 0, "x")
                    indices = self.get_arg(api_config, 1, "indices")
                    if hasattr(indices, 'shape'):
                        if indices.shape != self.shape:
                            if numpy.prod(self.shape) == 1:
                                self.numpy_tensor = numpy.full(indices.shape, self.get_random_numpy_tensor(shape=[], data_type=self.dtype)[()], dtype=self.dtype)
                            else:
                                random_values = self.get_random_numpy_tensor(shape=numpy.prod(indices.shape), data_type=self.dtype)
                                self.numpy_tensor = random_values.reshape(indices.shape)
                        else:
                            self.numpy_tensor = self.get_random_numpy_tensor(shape=self.shape, data_type=self.dtype)
            # q
            elif api_config.api_name in ["paddle.quantile"]:
                if not (key == "x" or index == 0):
                    self.numpy_tensor = numpy.random.rand(1).astype(self.dtype)

            # r                
            elif api_config.api_name in ["paddle.Tensor.reshape","paddle.reshape"]:
                if index == 0 or key == "x":
                    if not hasattr(api_config, "shape"):
                        api_config.shape = self.shape
                    if not hasattr(api_config, "maxvalue"):
                        api_config.maxvalue = self.numel()
                    if not hasattr(api_config, "tensornum"):
                        api_config.tensornum = 0
                    for arg in api_config.args:
                        if isinstance(arg, list) or isinstance(arg, tuple):
                            i = 0
                            for item in arg:
                                if "int" in str(type(item)):
                                    if item == 0:
                                        api_config.maxvalue = api_config.maxvalue // self.shape[i]
                                    elif not item == -1:
                                        api_config.maxvalue = api_config.maxvalue // item
                                if "Tensor" in str(type(item)):
                                    api_config.tensornum += 1
                                i += 1
                    for thekey, thevalue in api_config.kwargs.items():
                        if isinstance(thevalue, list) or isinstance(thevalue, tuple):
                            i = 0
                            for item in thevalue:
                                if "int" in str(type(item)):
                                    if item == 0:
                                        api_config.maxvalue = api_config.maxvalue // self.shape[i]
                                    elif not item == -1:
                                        api_config.maxvalue = api_config.maxvalue // item
                                if "Tensor" in str(type(item)):
                                    api_config.tensornum += 1
                                i += 1
                else:
                    self.dtype = "int32"
                    if self.shape != [] and self.shape != [1]:
                        self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                        for i in range(self.shape[0]):
                            if i < self.shape[0]-1:
                                self.numpy_tensor[i] = numpy.random.randint(1, api_config.maxvalue+1)
                                while api_config.maxvalue % self.numpy_tensor[i]:
                                    self.numpy_tensor[i] = numpy.random.randint(1, api_config.maxvalue+1)
                                api_config.maxvalue = api_config.maxvalue // self.numpy_tensor[i] 
                            else:
                                self.numpy_tensor[i] = api_config.maxvalue
                    else:
                        if api_config.tensornum == 1:
                            self.numpy_tensor = numpy.random.randint(api_config.maxvalue, api_config.maxvalue+1, size=self.shape).astype(self.dtype)
                        else:
                            api_config.tensornum -= 1
                            self.numpy_tensor = numpy.random.randint(1, api_config.maxvalue+1, size=self.shape).astype(self.dtype)
                            while api_config.maxvalue % self.numpy_tensor:
                                self.numpy_tensor = numpy.random.randint(1, api_config.maxvalue+1, size=self.shape).astype(self.dtype)
                            api_config.maxvalue = api_config.maxvalue // self.numpy_tensor
            
            # s
            elif api_config.api_name in ["paddle.slice"]:
                # if not hasattr(api_config, "element1"):
                #     if "axes" in api_config.kwargs:
                #         lens = len(api_config.kwargs["axes"])
                #     else:
                #         lens = len(api_config.args[1])
                #     api_config.element1 = lens + 1
                # if not hasattr(api_config, "element2"):
                #     if "starts" in api_config.kwargs:
                #         item = api_config.kwargs["starts"]
                #     else:
                #         item = api_config.args[2]
                #     if isinstance(item, list):
                #         api_config.element2 = api_config.element1 + len(item)
                #     else:
                #         api_config.element2 = api_config.element1 + 1
                if len(api_config.args) > 1:
                    axis = api_config.args[1]
                else:
                    axis = api_config.kwargs["axes"]
                if (index is not None and index == 0) or key == "input":
                    if not hasattr(api_config, "shape"):
                        api_config.shape = self.shape
                elif (index is not None and index == 2) or key == "starts":
                    num = []
                    for i in axis:
                        num.append(api_config.shape[i])
                    if not hasattr(api_config,"indice"):
                        api_config.indice = 0
                    if not hasattr(api_config,"start"):
                        api_config.start = []
                    if self.shape == []:
                        x = numpy.random.randint(0, 2)
                        if x == 0:
                            self.numpy_tensor = numpy.random.randint(0, num[api_config.indice]-1, self.shape)
                        else:
                            self.numpy_tensor = numpy.random.randint(-65535, -1, self.shape)
                        api_config.start.append(self.numpy_tensor)
                        api_config.indice += 1
                    else:
                        self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                        for i in range(self.numel()):
                            x = numpy.random.randint(0, 2)
                            if x == 0:
                                self.numpy_tensor[i] = numpy.random.randint(0, num[api_config.indice]-1)
                            else:
                                self.numpy_tensor[i] = numpy.random.randint(-65535,-1)
                            api_config.start.append(self.numpy_tensor[i])
                            api_config.indice += 1
                else:
                    if not hasattr(api_config,"start"):
                        if len(api_config.args) > 2:
                            api_config.start = api_config.args[2]
                        else:
                            api_config.start = api_config.kwargs["starts"]
                    num = []
                    for i in axis:
                        num.append(api_config.shape[i])
                    start = api_config.start
                    for i in range(len(start)):
                        if start[i] < 0:
                            start[i] = start[i] if start[i] > -1*num[i] else -1*num[i] 
                            start[i] += num[i]
                    if not hasattr(api_config,"index"):
                        api_config.index = 0                    
                    if self.shape == []:
                        x = numpy.random.randint(0, 2)
                        if x == 0:
                            self.numpy_tensor = numpy.random.randint(start[api_config.index]+1, 65535, self.shape)
                        else:
                            if start[api_config.index]-num[i] == 0:
                                start[api_config.inex] -= 1
                            self.numpy_tensor = numpy.random.randint(start[api_config.index]-num[i]+1, 0, self.shape)
                        api_config.index += 1                       
                    else:
                        self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                        for i in range(self.numel()):
                            x = numpy.random.randint(0, 2)
                            if x == 0:
                                self.numpy_tensor[i] = numpy.random.randint(start[api_config.index]+1, 65535)
                            else:
                                if start[api_config.index]-num[i] == 0:
                                    start[api_config.inex] -= 1
                                self.numpy_tensor[i] = numpy.random.randint(start[api_config.index]-num[api_config.index]+1, 0)
                            api_config.index += 1

            elif api_config.api_name in ["paddle.scatter"]:
                if key == "index" or index == 1:
                    d=self.get_arg(api_config, 0, "x")
                    s=d.shape[0]
                    self.numpy_tensor = numpy.random.randint(0, s, size=self.shape).astype(self.dtype)

            elif api_config.api_name in ["paddle.scatter_nd"]:
                future_data=self.get_arg(api_config, 2, "shape")     
                if (key == "index" or index == 0) and future_data and len(future_data):
                    self.numpy_tensor=numpy.zeros(self.shape)
                    s=self.shape
                    for ii in range(len(future_data)):  
                        if ii>=s[-1]:
                            break
                        self.numpy_tensor[...,ii] = numpy.random.randint(-future_data[ii], future_data[ii], size=self.numpy_tensor[...,ii].shape).astype(self.dtype)

            elif api_config.api_name in ["paddle.scatter_nd_add"]:
                if key == "index" or index == 1:
                    org=self.get_arg(api_config, 0, "x")
                    org=org.shape
                    self.numpy_tensor=numpy.zeros(self.shape)
                    ind=self.get_arg(api_config, 1, "index")
                    s=ind.shape
                    for ii in range(s[-1]):  
                        self.numpy_tensor[...,ii] = numpy.random.randint(-org[ii], org[ii], size=self.numpy_tensor[...,ii].shape).astype(self.dtype)
            elif api_config.api_name in ["paddle.shard_index"]:
                if self.check_arg(api_config, 0, "input"):
                    index_num = self.get_arg(api_config, 1, "index_num")
                    if index_num is None:
                        index_num = numpy.random.randint(1, 1000)
                    self.numpy_tensor = numpy.random.randint(0, index_num, size=self.shape).astype(self.dtype)
            elif api_config.api_name in ["paddle.sum", "paddle.squeeze"]:
                if self.check_arg(api_config, 1, "axis"):
                    self.numpy_tensor = self.generate_random_axes(api_config)
            elif api_config.api_name in ["paddle.split"]:
                if self.check_arg(api_config, 2, "axis"):
                    x_shape = self.get_arg(api_config, 0, "x").shape
                    num_or_sections = self.get_arg(api_config, 1, "num_or_sections")
                    if isinstance(num_or_sections, (list, tuple)):
                        neg_one_count = sum(1 for x in num_or_sections if x == -1)
                        if neg_one_count > 1:
                            raise ValueError(
                                f"num_or_sections can contain at most one -1, but got {num_or_sections}"
                            )
                        num_splits = len(num_or_sections)
                        known_size = sum(num_or_sections) + neg_one_count
                    elif isinstance(num_or_sections, int):
                        num_splits = num_or_sections
                        known_size = None
                    else:
                        raise ValueError(
                            f"num_or_sections must be an int, list, or tuple, but got {type(num_or_sections)}"
                        )

                    target_dim = None
                    max_dim = len(x_shape)
                    if max_dim == 0:
                        target_dim = numpy.random.randint(-1, 0)
                    else:
                        for dim in range(max_dim):
                            dim_size = x_shape[dim]
                            if isinstance(num_or_sections, int) and dim_size % num_splits == 0:
                                target_dim = dim
                            elif isinstance(num_or_sections, (list, tuple)):
                                if neg_one_count == 0 and dim_size == known_size:
                                    target_dim = dim
                                elif neg_one_count == 1 and dim_size > known_size:
                                    target_dim = dim
                    if target_dim is None:
                        raise ValueError(
                            f"No valid axis found for paddle.split with x.shape={x_shape} and num_or_sections={num_or_sections}"
                        )

                    shape_len = len(self.shape)
                    if shape_len == 0:
                        self.numpy_tensor = numpy.array(target_dim, dtype=self.dtype)
                    elif shape_len == 1 and self.shape[0] == 1:
                        self.numpy_tensor = numpy.array([target_dim], dtype=self.dtype)
                    else:
                        raise ValueError(
                            f"Invalid shape for 'axis' Tensor in paddle.split. "
                            f"Expected a 0-D or 1-D Tensor, but got shape {self.shape}."
                        )

            elif api_config.api_name in ["paddle.nn.functional.softmax"]:
                # for TensorConfig axis
                x_tensor_config = self.get_arg(api_config, 0, "x")
                axis_config = self.get_arg(api_config, 1, "axis")
                
                if self.check_arg(api_config, 0, "x"):
                    self.numpy_tensor = self.get_random_numpy_tensor(self.shape, self.dtype)
                elif self.check_arg(api_config, 1, "axis"):
                    len_shape_x = len(x_tensor_config.shape)
                    # specify if axis is a scalar tensor, else is a int according to doc
                    if isinstance(axis_config, TensorConfig):
                        axis = self.get_random_numpy_tensor(axis_config.shape, axis_config.dtype, min=-len_shape_x, max=len_shape_x)
                        self.numpy_tensor = axis
                
                return self.numpy_tensor
            
            elif api_config.api_name in ["paddle.standard_normal"]:
                if index==0 or key=='shape': 
                    self.numpy_tensor =numpy.random.randint(1, 128, size=self.shape).astype(self.dtype)

            elif api_config.api_name in ["paddle.strided_slice"]:
                s=self.get_arg(api_config,0,'x')
                if self.check_arg(api_config,1,'axes'):
                    self.numpy_tensor =numpy.random.randint(0,len(s.shape), size=self.shape).astype(self.dtype)
                elif index:
                    axes=self.get_arg(api_config,1,'axes')
                    for i in range(len(axes)):
                        if isinstance(axes[i],TensorConfig):
                            axes[i]=int(axes[i].numpy_tensor)
                    if self.check_arg(api_config,2,'starts'):
                        axes=self.get_arg(api_config,1,'axes')
                        if not isinstance(axes,list):
                            axes=axes.numpy_tensor
                        ind=kwargs['list_index'][0]
                        self.numpy_tensor =numpy.random.randint(0,s.shape[axes[ind]]-1, size=self.shape).astype(self.dtype)
                    elif self.check_arg(api_config,3,'ends'):
                        ind=kwargs['list_index'][0]
                        pre=self.get_arg(api_config,2,'starts')
                        self.numpy_tensor =numpy.random.randint(pre[ind].numpy_tensor+1,s.shape[axes[ind]], size=self.shape).astype(self.dtype)
                    elif self.check_arg(api_config,4,'strides'):
                        ind=kwargs['list_index'][0]
                        self.numpy_tensor =numpy.random.randint(1,s.shape[axes[ind]], size=self.shape).astype(self.dtype)

            # t
            elif api_config.api_name in ["paddle.Tensor.take_along_axis", "paddle.take_along_axis"]:
                if self.check_arg(api_config, 1, "indices"):
                    arr = self.get_arg(api_config, 0, "arr")
                    min_dim = min(arr.shape)
                    indices = (numpy.random.randint(0, min_dim-1, size=self.numel())).astype("int64")
                    self.numpy_tensor = indices.reshape(self.shape)
                    self.dtype = "int64"
                    
            elif api_config.api_name in ["paddle.Tensor.clip"]:
                if index>0 and key!='x':
                    self.numpy_tensor=numpy.random.random()-0.5
                if key == "max" or index == 2:
                    pre=self.get_arg(api_config, 1, "min")
                    self.numpy_tensor=numpy.clip(self.numpy_tensor,pre.numpy_tensor,None)
            
            elif api_config.api_name in ['paddle.Tensor.gather',"paddle.gather"]:
                if key == "index" or index == 1:
                    s=self.get_arg(api_config, 0, "x")

                    if 'axis' in api_config.kwargs:
                        tmp=self.get_arg(api_config,arg_name='axis')
                        if isinstance(tmp,TensorConfig):
                            tmp=tmp.shape
                            tmp=tmp[0]
                    else:
                        tmp=0
                    self.numpy_tensor = (numpy.random.randint(0,s.shape[tmp], size=self.shape)).astype(self.dtype)
                elif key == "axis" or index == 2:
                    self.numpy_tensor = (numpy.random.randint(0,2, size=self.shape)).astype(self.dtype)

            elif api_config.api_name in ["paddle.Tensor.gather_nd","paddle.gather_nd"]:
                if key == "index" or index == 1:
                    if 'x' in api_config.kwargs:
                        org=self.get_arg(api_config,arg_name='x')
                    else:
                        org=self.get_arg(api_config,0)
                    org=org.shape
                    if 'index' in api_config.kwargs:
                        s=self.get_arg(api_config,arg_name='index')
                    else:
                        s=self.get_arg(api_config,1)
                    s=s.shape
                    self.numpy_tensor=numpy.zeros(s)
                    for i in range(s[-1]):
                        self.numpy_tensor[...,i]=(numpy.random.randint(0,org[i], size=self.numpy_tensor[...,i].shape)).astype(self.dtype)

            elif api_config.api_name in ["paddle.Tensor.index_select"]:
                if self.check_arg(api_config,1,'index'):
                    axis=self.get_arg(api_config, 2, 'axis')
                    if axis is None:
                        axis=0
                    inputs=self.get_arg(api_config, 0, "x")
                    self.numpy_tensor = numpy.random.randint(0,inputs.shape[axis], size=self.shape).astype(self.dtype)

            elif api_config.api_name in ["paddle.Tensor.tile"]:
                if index==1 or key=='repeat_times':
                    self.numpy_tensor = numpy.random.randint(1,128, size=self.shape).astype(self.dtype)
            # u
            elif api_config.api_name in ["paddle.Tensor.unflatten","paddle.unflatten"]:
                if key == "x" or index == 0:
                    if not hasattr(api_config,"first_shape"):
                        api_config.first_shape = self.shape
                elif key == "axis" or index == 1:
                    self.numpy_tensor = numpy.random.randint(0, len(api_config.first_shape), size=self.shape).astype(self.dtype)
                    if not hasattr(api_config, "axis"):
                        api_config.axis = self.numpy_tensor
                elif key == "shape" or index >= 2:
                    if not hasattr(api_config, "axis"):
                        if len(api_config.args) > 1:
                            api_config.axis = api_config.args[1]
                        else:
                            api_config.axis = api_config.kwargs["axis"]
                    if not hasattr(api_config, "maxvalue"):
                        maxvalue = api_config.first_shape[api_config.axis]
                        api_config.maxvalue = maxvalue
                    if len(api_config.args) > 2:
                        arg = api_config.args[2]
                    else:
                        arg = api_config.kwargs["shape"]
                    if isinstance(arg,TensorConfig):
                        self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                        for i in range(self.numel()-1):
                            self.numpy_tensor[i] = numpy.random.randint(1, maxvalue+1)
                            while maxvalue % self.numpy_tensor.any():
                                self.numpy_tensor[i] = numpy.random.randint(1, maxvalue+1)
                            maxvalue = maxvalue // self.numpy_tensor[i]
                        self.numpy_tensor[self.numel()-1] = maxvalue
                    elif isinstance(arg, list) or isinstance(arg, tuple):
                        if not hasattr(api_config,"tensornum"):    
                            api_config.tensornum = 0
                            for item in arg:
                                if "int" in str(type(item)) and not item == -1:
                                    api_config.maxvalue = api_config.maxvalue // item
                                if "Tensor" in str(type(item)):
                                    api_config.tensornum += 1
                        if api_config.tensornum > 1:
                            self.numpy_tensor = numpy.random.randint(1, api_config.maxvalue+1, size=self.shape).astype(self.dtype)
                            while maxvalue % self.numpy_tensor:
                                self.numpy_tensor[i] = numpy.random.randint(1, api_config.maxvalue+1, size=self.shape).astype(self.dtype)
                            maxvalue = maxvalue // self.numpy_tensor[i]  
                            api_config.tensor_num -= 1
                        else:
                            self.numpy_tensor = numpy.random.randint(api_config.maxvalue, api_config.maxvalue+1, size=self.shape).astype(self.dtype)
            
            elif api_config.api_name in ["paddle.unsqueeze"]:
                if self.check_arg(api_config, 1, "axis"):
                    x_shape = self.get_arg(api_config, 0, "x").shape
                    max_dim = len(x_shape) + 1
                    if len(self.shape) == 0:
                        dim = numpy.random.randint(0, max_dim)
                        if numpy.random.rand() > 0.5:
                            dim -= max_dim
                        self.numpy_tensor = numpy.array(dim, dtype=self.dtype)
                    elif len(self.shape) == 1:
                        dims = numpy.random.choice(max_dim, size=self.shape[0], replace=False)
                        mask = numpy.random.rand(self.shape[0]) > 0.5
                        dims = numpy.where(mask, dims - max_dim, dims)
                        self.numpy_tensor = numpy.array(dims, dtype=self.dtype)
                    else:
                        raise ValueError(
                            f"Invalid shape for 'axis' Tensor in paddle.unsqueeze. "
                            f"Expected a 0-D or 1-D Tensor, but got shape {self.shape}."
                        )
            # v
            # w
            # x
            # y
            # z
            elif api_config.api_name in ["paddle.zeros"]:
                self.numpy_tensor = numpy.random.randint(0, 2048, size = self.shape)
                
            elif api_config.api_name in ["paddle.nn.functional.zeropad2d"]:
                if self.check_arg(api_config, 0, "x"):
                    self.numpy_tensor = self.get_random_numpy_tensor(self.shape, self.dtype)
                elif self.check_arg(api_config, 1, "padding"):
                    # padding value should not be too large 
                    self.numpy_tensor = self.get_random_numpy_tensor(self.shape, self.dtype, min=0, max=10)
                    
            # _
            elif api_config.api_name in ["paddle.Tensor.__getitem__","paddle.Tensor.__setitem__"] and (len(api_config.args) > 1 and str(api_config.args[1]) == str(self) or str(api_config.args[0]) != str(self)):
                arr = self.get_arg(api_config, 0, "arr")
                min_dim = min(arr.shape)
                indices = (numpy.random.randint(0, min_dim, size=self.numel())).astype("int64")
                self.numpy_tensor = indices.reshape(self.shape)
            if self.numpy_tensor is None:
                if USE_CACHED_NUMPY:
                    dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                    self.numpy_tensor = self.get_cached_numpy(dtype, self.shape)
                else:
                    if "int" in self.dtype:
                        self.numpy_tensor = (numpy.random.randint(-65535, 65535, size=self.shape)).astype(self.dtype)
                    else:
                        dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                        self.numpy_tensor = (numpy.random.random(self.shape) - 0.5).astype(dtype)
        return self.numpy_tensor

    def get_paddle_tensor(self, api_config, index=None, key=None, **kwargs):
        if index is not None:
            self.index = index
        if key is not None:
            self.key = key

        if self.dtype in ["float8_e5m2", "float8_e4m3fn"]:
            print("Warning ", self.dtype, "not supported")
            return

        if self.paddle_tensor is None:
            self.paddle_tensor = paddle.to_tensor(
                self.get_numpy_tensor(api_config, index, key, **kwargs),
                dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            )
            self.paddle_tensor.stop_gradient = True
            if self.dtype in ['float32', 'float64', 'float16', 'complex64', 'complex128', 'bfloat16']:
                if self.dtype == "bfloat16":
                    self.paddle_tensor = paddle.cast(self.paddle_tensor, dtype="uint16")
                self.paddle_tensor.stop_gradient = False
        return self.paddle_tensor

    def get_torch_tensor(self, api_config):
        if self.dtype in ["float8_e5m2", "float8_e4m3fn"]:
            print("Warning ", self.dtype, "not supported")
            return

        device = torch.device("cuda:0")
        torch.set_default_device(device)
        if self.torch_tensor is None:
            self.torch_tensor = torch.tensor(
                self.get_numpy_tensor(api_config),
                dtype=self.convert_dtype_to_torch_type(self.dtype)
                if self.dtype != 'bfloat16'
                else torch.float32,
                requires_grad=True if self.dtype in ['float32', 'float64', 'float16', 'complex64', 'complex128', 'bfloat16'] else False,
            )
            if self.dtype == "bfloat16":
                self.torch_tensor = self.torch_tensor.to(dtype=torch.bfloat16)
        return self.torch_tensor

    def clear_tensor(self):
        self.torch_tensor = None
        self.paddle_tensor = None
        self.numpy_tensor = None
        torch.cuda.empty_cache()
        paddle.device.cuda.empty_cache()

    def clear_paddle_tensor(self):
        del self.paddle_tensor
        self.paddle_tensor = None
        paddle.device.cuda.empty_cache()

    def clear_numpy_tensors(self):
        del self.numpy_tensor
        self.numpy_tensor = None

    def clear_torch_tensor(self):
        del self.torch_tensor
        self.torch_tensor = None
        torch.cuda.empty_cache()

    def fill_numpy_tensor(self, full_value):
        self.numpy_tensor = numpy.full(shape=self.shape, fill_value=full_value, dtype=self.dtype)

    def check_arg(self, api_config, arg_pos=None, arg_name=None):
        """Checks if the argument in api_config matches this instance"""
        return (
            (arg_pos is not None and hasattr(self, "index") and self.index == arg_pos)
            or (arg_name is not None and hasattr(self, "key") and self.key == arg_name)
        )

    def get_arg(self, api_config, arg_pos=None, arg_name=None):
        """Get the argument value from the api_config"""
        if arg_pos is not None and 0 <= arg_pos < len(api_config.args):
            return api_config.args[arg_pos]
        if arg_name and arg_name in api_config.kwargs:
            return api_config.kwargs[arg_name]
        return None

    def get_initialized_value(self, api_config, arg_pos=None, arg_name=None):
        """Get the initialized numpy_tensor value from the api_config instead of the TensorConfig"""    
        # for uninitialized numpy_tensor, return None implicitly as numpy_tensor is None 
        if arg_pos is not None and 0 <= arg_pos < len(api_config.args):
            if isinstance(api_config.args[arg_pos], TensorConfig):
                return api_config.args[arg_pos].numpy_tensor
            else:
                return api_config.args[arg_pos]
        if arg_name and arg_name in api_config.kwargs:
            if isinstance(api_config.kwargs[arg_name], TensorConfig):
                return api_config.kwargs[arg_name].numpy_tensor
            else:
                return api_config.kwargs[arg_name]
        # for args that does not appear in api_config
        if arg_pos >= len(api_config.args) or arg_name not in api_config.kwargs:
            return None
        # error case 
        if arg_pos is None and arg_name is None:
            raise ValueError("either arg_pos or arg_name must be provided.")
        elif arg_pos:
            if arg_pos < 0:
                raise IndexError(f"argument position {arg_pos} is out of range for api_config with {len(api_config.args)} arguments.")
            else: 
                # case type(api_config.args[arg_pos]) != TensorConfig:
                raise TypeError(f"argument at position {arg_pos} is not of type TensorConfig.")
        else:
            # case type(api_config.kwargs[arg_name]) != TensorConfig:
            raise TypeError(f"argument '{arg_name}' is not of type TensorConfig.")
    
    def set_tensor_arg_value(self, api_config, arg_pos=None, arg_name=None, value=None):
        if arg_pos is not None and 0 <= arg_pos < len(api_config.args) and isinstance(api_config.args[arg_pos], TensorConfig):
            api_config.args[arg_pos].numpy_tensor = value
        elif arg_name and arg_name in api_config.kwargs and isinstance(api_config.kwargs[arg_name], TensorConfig):
            api_config.kwargs[arg_name].numpy_tensor = value
        else:
            raise ValueError(f"argument at position {arg_pos} or name '{arg_name}' is not of type TensorConfig.")
        
    def get_random_numpy_tensor(self, shape=None, data_type=None, min=None, max=None):
        # extract default init logic 
        if USE_CACHED_NUMPY:
            dtype = "float32" if data_type == "bfloat16" else data_type
            numpy_tensor = self.get_cached_numpy(dtype, shape)
        else:
            if "int" in data_type:
                min = min if min is not None else -65535
                max = max if max is not None else 65535
                numpy_tensor = (numpy.random.randint(min, max, size=shape)).astype(data_type)
            else:
                # TO DO: check boundary and cached numpy
                dtype = "float32" if data_type == "bfloat16" else data_type
                min = min if min is not None else numpy.finfo(dtype).min / 2
                max = max if max is not None else numpy.finfo(dtype).max / 2
                numpy_tensor = (numpy.random.uniform(min, max, size=shape)).astype(dtype)
        return numpy_tensor

class APIConfig:
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.args = copy.deepcopy(self.args)
        result.kwargs = copy.deepcopy(self.kwargs)
        result.api_name = self.api_name
        return result

    def __init__(self, config):
        config = config.replace("\n", "")
        self.config = config
        self.args = []
        self.kwargs = collections.OrderedDict()
        config = config.replace("Tensor(", "TensorConfig(")

        self.api_name, offset = self.get_api(config)

        if self.api_name == "paddle.einsum":
            tmp = config[config.index("\"") + 1:]
            value = tmp[:tmp.index("\"")]
            offset = config.index("\"") + 1 + tmp.index("\"")
            if "equation" in config:
                self.append_kwargs("equation", value)
            else:
                self.append_args(value)

        while(True):
            tocken, offset = self.get_tocken(config, offset)
            if offset is None:
                return

            is_kwarg = config[offset] == '='
            if is_kwarg:
                key = tocken
                tocken, offset = self.get_tocken(config, offset+1)

            value, offset = self.get_one_arg(tocken, config, offset)
            
            if offset is None:
                return

            if is_kwarg:
                self.append_kwargs(key, value)
            else:
                self.append_args(value)

    def append_args(self, arg):
        self.args.append(arg)
        
    def append_kwargs(self, name, arg):
        self.kwargs[name] = arg

    def dump_item_str(self, item):
        type_mapping = {
            numpy.int16: int,
            numpy.int32: int,
            numpy.int64: int,
            numpy.float16: float,
            numpy.float32: float,
            numpy.float64: float,
            numpy.integer: int,
            numpy.floating: float,
            numpy.bool_: bool,
            numpy.complexfloating: complex,
            numpy.str_: str,
            numpy.bytes_: bytes,
            # numpy.unicode_: str,
        }
        for numpy_type, builtin_type in type_mapping.items():
            if isinstance(item, numpy_type):
                item = builtin_type(item)
                break

        if isinstance(item, TensorConfig):
            return str(item)
        elif isinstance(item, paddle.base.core.DataType):
            return "Dtype(" + str(item)[7:] + ")"
        elif isinstance(item, paddle.base.core.VarDesc.VarType):
            return "VarType(" + str(item)[7:] + ")"
        elif isinstance(item, list):
            result = "list["
            for sub_item in item:
                tmp = self.dump_item_str(sub_item)
                if tmp == "":
                    return ""
                result = result + tmp + ","
            result = result + "]"
            return result
        elif isinstance(item, tuple):
            result = "tuple("
            for sub_item in item:
                tmp = self.dump_item_str(sub_item)
                if tmp == "":
                    return ""
                result = result + tmp + ","
            result = result + ")"
            return result
        elif isinstance(item, slice):
            return (
                "slice("
                + str(item.start)
                + ","
                + str(item.stop)
                + ","
                + str(item.step)
                + ")"
            )
        elif isinstance(item, complex):
            return (
                "complex("
                + self.dump_item_str(item.real)
                + ","
                + self.dump_item_str(item.imag)
                + ")"
            )
        elif item is None:
            return "None"
        elif isinstance(
            item, (paddle.base.Variable, paddle.base.libpaddle.pir.Value)
        ):
            return ""
        elif item == math.inf:
            return "math.inf"
        elif item == -math.inf:
            return "-math.inf"
        elif item == math.nan:
            return "math.nan"
        elif item == -math.nan:
            return "-math.nan"
        elif isinstance(item, (bool, int, float)):
            return str(item)
        elif isinstance(item, str):
            return '"' + item + '"'
        elif isinstance(item, type):
            return (
                "type("
                + str(item)[str(item).index("'") + 1 : str(item).rindex("'")]
                + ")"
            )
        else:
            return str(item)


    def __str__(self):
        result = ""
        result = result + self.api_name + "("
        for arg in self.args:
            result = result + self.dump_item_str(arg) + ", "
        
        for key, value in self.kwargs.items():
            result = result + key + "=" + self.dump_item_str(value) + ", "

        result = result + ")"
        return result

    # def get_tocken(self, config, offset):
    #     def is_int(tocken):
    #         try:
    #             int(tocken)
    #             return True
    #         except Exception as err:
    #             return False
    #     pattern = r'\b[A-Za-z0-9._+-]+\b|-[A-Za-z0-9._+-]+\b'
    #     match = re.search(pattern, config[offset:])
    #     if match:
    #         if is_int(match.group()) and config[offset + match.start() + len(match.group())] == ".":
    #             return match.group()+".", offset + match.start() + len(match.group()) + 1
    #         return match.group(), offset + match.start() + len(match.group())
    #     return None, None

    def get_tocken(self,config, offset):
        def is_int(token):
            try:
                int(token)
                return True
            except Exception as err:
                return False

        # Modified pattern to handle decimal numbers starting with dot
        pattern = r'\b[A-Za-z0-9._+-]+\b|-[A-Za-z0-9._+-]+\b|\.[0-9]+'
        match = re.search(pattern, config[offset:])
        if match:
            token = match.group()
            # Handle the case where token starts with dot followed by digits
            if token.startswith('.') and token[1:].isdigit():
                return token, offset + match.start() + len(token)

            if is_int(token) and offset + match.start() + len(token) < len(config) and config[
                offset + match.start() + len(token)] == ".":
                return token + ".", offset + match.start() + len(token) + 1
            return token, offset + match.start() + len(token)
        return None, None


    def get_api(self, config):
        return config[0:config.index("(")], len(config[0:config.index("(")])

    def get_tensor(self, config, offset):
        config = config[offset:]
        tensor_str = config[config.index("TensorConfig"):config.index(")")+1]
        return eval(tensor_str), offset + len(tensor_str)

    def get_dtype(self, config, offset):
        tocken, offset = self.get_tocken(config, offset)
        return paddle.pir.core.convert_np_dtype_to_dtype_(tocken), offset

    def get_vartype(self, config, offset):
        tocken, offset = self.get_tocken(config, offset)
        return paddle.base.framework.convert_np_dtype_to_proto_type(tocken), offset

    def get_list(self, config, offset):
        result = []
        tmp = 0
        last_index = offset
        for i in range(offset, len(config)):
            if config[i] == "[":
                tmp = tmp + 1
            if config[i] == "]":
                tmp = tmp - 1
            if tmp == 0:
                last_index = i
                break
        
        list_str = config[offset: last_index+1]
        if "TensorConfig" not in list_str:
            list_str = list_str.replace(",", " ")

        offset = 1
        while(True):
            tocken, offset = self.get_tocken(list_str, offset)
            if offset is None:
                break

            value, offset = self.get_one_arg(tocken, list_str, offset)

            if offset is None:
                break

            result.append(value)

        return result, last_index+1

    def get_tuple(self, config, offset):
        result = []
        tmp = 0
        last_index = offset
        for i in range(offset, len(config)):
            if config[i] == "(":
                tmp = tmp + 1
            if config[i] == ")":
                tmp = tmp - 1
            if tmp == 0:
                last_index = i
                break
        
        tuple_str = config[offset: last_index+1]

        tuple_str = tuple_str.replace(",", " , ")

        offset = 1
        while(True):
            tocken, offset = self.get_tocken(tuple_str, offset)
            if offset is None:
                break

            value, offset = self.get_one_arg(tocken, tuple_str, offset)

            if offset is None:
                break

            result.append(value)

        return tuple(result), last_index+1

    def get_slice(self, config, offset):
        config = config[offset:]
        slice_str = config[config.index("("):config.index(")")+1]
        return eval("slice"+slice_str), offset+len(slice_str)

    def get_complex(self, config, offset):
        config = config[offset:]
        complex_str = config[config.index("("):config.index(")")+1]
        if "nan" in complex_str and complex_str[complex_str.index('nan')-1] != ".":
            complex_str = complex_str.replace("nan", "float('nan')")
        return eval("complex"+complex_str), offset+len(complex_str)

    def get_numpy_type(self, config, offset):
        config = config[offset:]
        numpy_type_str = config[config.index("(")+1:config.index(")")]
        if numpy_type_str == "numpy.bool":
            return numpy.bool_, offset+len(numpy_type_str)+2
        return eval(numpy_type_str), offset+len(numpy_type_str)+2

    def get_one_arg(self, tocken, config, offset):
        if tocken == "TensorConfig":
            value, offset = self.get_tensor(config, offset-len(tocken))
        elif tocken == "Dtype":
            value, offset = self.get_dtype(config, offset)
        elif tocken == "VarType":
            value, offset = self.get_vartype(config, offset)
        elif tocken == "list":
            value, offset = self.get_list(config, offset)
        elif tocken == "tuple":
            value, offset = self.get_tuple(config, offset)
        elif tocken == "slice":
            value, offset = self.get_slice(config, offset)
        elif tocken == "complex":
            value, offset = self.get_complex(config, offset)
        elif tocken == "type":
            value, offset = self.get_numpy_type(config, offset)
        elif tocken == "nan":
            value = float('nan')
        elif tocken is not None and config[offset - len(tocken) - 1] == "\"":
            # fix tocken is not correct in str with spaces
            next_quote_idx  = config.index("\"", offset)
            value = config[offset - len(tocken):next_quote_idx]
            offset = next_quote_idx
        elif tocken is None:
            return None, None
        else:
            if tocken[0]=='.':
                tocken='0'+tocken
            value = eval(tocken)
        return value, offset


def analyse_configs(config_path):
    with open(config_path, "r") as f:
        configs = f.readlines()
        f.close()

    api_configs = []
    for config in configs:
        # print(config)
        api_config = APIConfig(config)
        api_configs.append(api_config)
    return api_configs

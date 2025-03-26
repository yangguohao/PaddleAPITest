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
        return "Tensor("+str(self.shape)+",\""+str(self.dtype)+"\")"
    def __repr__(self):
        return "Tensor("+str(self.shape)+",\""+str(self.dtype)+"\")"

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

    def get_numpy_tensor(self, api_config):
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
            # b
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
            elif api_config.api_name in ["paddle.Tensor.take_along_axis", "paddle.take_along_axis"]:
                if (len(api_config.args) > 1 and str(api_config.args[1]) == str(self)) or "indices" in api_config.kwargs:
                    if len(api_config.args) > 0:
                        arr = api_config.args[0]
                    elif "arr" in api_config.kwargs:
                        arr = api_config.kwargs["arr"]
                min_dim = min(arr.shape)
                indices = (numpy.random.randint(0, min_dim-1, size=self.numel())).astype("int64")
                self.numpy_tensor = indices.reshape(self.shape)
                self.dtype = "int64"

            elif api_config.api_name in ["paddle.zeros","paddle.eye"]:
                self.numpy_tensor = numpy.random.randint(0, 2048, size = self.shape)

            elif api_config.api_name in ["paddle.full"]:
                if (len(api_config.args) > 1 and str(api_config.args[1]) == str(self)) or ("fill_value" in api_config.kwargs and api_config.kwargs["fill_value"] == str(self)):
                    if "int" in self.dtype:
                        self.numpy_tensor = (numpy.random.randint(1, 65535, size=self.shape)).astype(self.dtype)
                    else:
                        dtype = "float32" if self.dtype == "bfloat16" else self.dtype
                        self.numpy_tensor = (numpy.random.random(self.shape) + 0.5).astype(dtype)
                else:
                    self.numpy_tensor = (numpy.random.randint(0, 2048, size=self.shape)).astype("int64")
                    self.dtype = "int64"
            
            elif api_config.api_name in ["paddle.add_n","paddle.matmul"]:
                if self.dtype in [numpy.float16, numpy.float32, "float16", "float32", "bfloat16"]:
                    self.dtype = numpy.float64
                    self.numpy_tensor = (numpy.random.random(self.shape) + 0.5).astype(self.dtype)
                elif self.dtype in [numpy.int8, numpy.int16, numpy.int32, numpy.uint8, numpy.uint16, \
                                    "int8", "int16", "int32", "uint8", "uint16", ]:
                    self.dtype = numpy.int64
                    self.numpy_tensor = (numpy.random.randint(0, 65535, size=self.shape)).astype(self.dtype)
                elif self.dtype in [numpy.complex64, "complex64"]:
                    self.numpy_tensor = (numpy.random.random(self.shape) + 0.5).astype("complex128") + ((numpy.random.random(self.shape) + 0.5) * j).astype("complex128")
                    self.dtype = numpy.complex128

            # u
            # v
            # w
            # x
            # y
            # z
            # _
            elif api_config.api_name in ["paddle.Tensor.__getitem__","paddle.Tensor.__setitem__"] and (len(api_config.args) > 1 and str(api_config.args[1]) == str(self) or str(api_config.args[0]) != str(self)):
                arr = None
                if len(api_config.args) > 0:
                    arr = api_config.args[0]
                elif "arr" in api_config.kwargs:
                    arr = api_config.kwargs["arr"]
                min_dim = min(arr.shape)
                indices = (numpy.random.randint(0, min_dim, size=self.numel())).astype("int64")
                self.numpy_tensor = indices.reshape(self.shape)
            else:
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

    def get_paddle_tensor(self, api_config):
        if self.dtype in ["float8_e5m2", "float8_e4m3fn"]:
            print("Warning ", self.dtype, "not supported")
            return

        if self.paddle_tensor is None:
            self.paddle_tensor = paddle.to_tensor(
                self.get_numpy_tensor(api_config),
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
    
    def convert_dtype_to_numpy_type(self, config):

        if("bfloat16" in config):
            config = config.replace("bfloat16", "numpy.uint16")
        else:
            config = config.replace("float32", "numpy.float32")
            config = config.replace("float16", "numpy.float16")
            config = config.replace("float64", "numpy.float64")
            config = config.replace("int16", "numpy.int16")
            config = config.replace("int8", "numpy.int8")
            config = config.replace("bool", "numpy.bool_")
            config = config.replace("uint8", "numpy.uint8")
            config = config.replace("uint16", "numpy.uint16")
            config = config.replace("int32", "numpy.int32")
            config = config.replace("int64", "numpy.int64")
            config = config.replace("complex64", "numpy.complex64")
            config = config.replace("complex128", "numpy.complex128")
        
        return config

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

    def get_tocken(self, config, offset):
        def is_int(tocken):
            try:
                int(tocken)
                return True
            except Exception as err:
                return False
        pattern = r'\b[A-Za-z0-9+-._]+\b|-[A-Za-z0-9+-._]+\b'
        match = re.search(pattern, config[offset:])
        if match:
            if is_int(match.group()) and config[offset + match.start() + len(match.group())] == ".":
                return match.group()+".", offset + match.start() + len(match.group()) + 1
            return match.group(), offset + match.start() + len(match.group())
        return None, None

    def get_api(self, config):
        return config[0:config.index("(")], len(config[0:config.index("(")])

    def get_tensor(self, config, offset):
        config = config[offset:]
        tensor_str = config[config.index("TensorConfig"):config.index(")")+1]
        
        try:
            tensor  = eval(tensor_str)
        except Exception as err:
            tensor_str = self.convert_dtype_to_numpy_type(tensor_str)
            tensor = eval(tensor_str)
            offset-=5

        return tensor, offset + len(tensor_str)

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
        elif config[offset] == '\"':
            value = tocken
        elif tocken is None:
            return None, None
        else:
            try:
                value = eval(tocken)
            except Exception as err:
                value = tocken
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

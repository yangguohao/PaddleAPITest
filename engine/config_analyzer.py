import re
import collections
import paddle
import numpy
import math
import json
import torch

class TensorConfig:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
    def __str__(self):
        return "TensorConfig("+str(self.shape)+","+self.dtype+")"

def get_tocken(config, offset):
    pattern = r'\b[A-Za-z0-9+-._]+\b|-[A-Za-z0-9+-._]+\b'
    match = re.search(pattern, config[offset:])
    if match:
        return match.group(), offset + match.start() + len(match.group())
    return None, None

def get_api(config):
    return config[0:config.index("(")], len(config[0:config.index("(")])

def get_tensor(config, offset):
    config = config[offset:]
    tensor_str = config[config.index("TensorConfig"):config.index(")")+1]
    return eval(tensor_str), offset + len(tensor_str)

def get_dtype(config, offset):
    tocken, offset = get_tocken(config, offset)
    return paddle.pir.core.convert_np_dtype_to_dtype_(tocken), offset

def get_vartype(config, offset):
    tocken, offset = get_tocken(config, offset)
    return paddle.base.framework.convert_np_dtype_to_proto_type(tocken), offset

def get_list(config, offset):
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
        tocken, offset = get_tocken(list_str, offset)
        if offset is None:
            break

        value, offset = get_one_arg(tocken, list_str, offset)

        if offset is None:
            break

        result.append(value)

    return result, last_index+1

def get_tuple(config, offset):
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
    if "TensorConfig" not in tuple_str:
        tuple_str = tuple_str.replace(",", " ")

    offset = 1
    while(True):
        tocken, offset = get_tocken(tuple_str, offset)
        if offset is None:
            break

        value, offset = get_one_arg(tocken, tuple_str, offset)

        if offset is None:
            break

        result.append(value)

    return tuple(result), last_index+1

def get_slice(config, offset):
    config = config[offset:]
    slice_str = config[config.index("("):config.index(")")+1]
    return eval("slice"+slice_str), offset+len(slice_str)

def get_complex(config, offset):
    config = config[offset:]
    complex_str = config[config.index("("):config.index(")")+1]
    if "nan" in complex_str and complex_str[complex_str.index('nan')-1] != ".":
        complex_str = complex_str.replace("nan", "float('nan')")
    return eval("complex"+complex_str), offset+len(complex_str)

def get_numpy_type(config, offset):
    config = config[offset:]
    numpy_type_str = config[config.index("(")+1:config.index(")")]
    return eval(numpy_type_str), offset+len(numpy_type_str)+2

def get_one_arg(tocken, config, offset):
    if tocken == "TensorConfig":
        value, offset = get_tensor(config, offset-len(tocken))
    elif tocken == "Dtype":
        value, offset = get_dtype(config, offset)
    elif tocken == "VarType":
        value, offset = get_vartype(config, offset)
    elif tocken == "list":
        value, offset = get_list(config, offset)
    elif tocken == "tuple":
        value, offset = get_tuple(config, offset)
    elif tocken == "slice":
        value, offset = get_slice(config, offset)
    elif tocken == "complex":
        value, offset = get_complex(config, offset)
    elif tocken == "type":
        value, offset = get_numpy_type(config, offset)
    elif tocken == "nan":
        value = float('nan')
    elif config[offset] == '\"':
        value = tocken
    elif tocken is None:
        return None, None
    else:
        value = eval(tocken)
    return value, offset

class APIConfig:
    def __init__(self, config):
        config = config.replace("\n", "")
        self.config = config
        self.args = []
        self.kwargs = collections.OrderedDict()
        config = config.replace("Tensor", "TensorConfig")

        self.api_name, offset = get_api(config)

        while(True):
            tocken, offset = get_tocken(config, offset)
            if offset is None:
                return

            is_kwarg = config[offset] == '='
            if is_kwarg:
                key = tocken
                tocken, offset = get_tocken(config, offset+1)

            value, offset = get_one_arg(tocken, config, offset)
            
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
        
    def __str__(self):
        result = "APIConfig:"
        result = result + self.api_name + "("
        for arg in self.args:
            result = result + str(arg) + ","
        
        for key, value in self.kwargs.items():
            result = result + key + "=" + str(value) + ","
        
        result = result + ")"
        return result

    def test(self):
        pass

class APITestBase:
    def __init__(self, api_config):
        self.api_config = api_config
    def rand_tensor(self, tensor_config):
        if tensor_config.dtype in ["float32", "float64"]:
            return paddle.rand(tensor_config.shape, tensor_config.dtype)
        if tensor_config.dtype in ["complex64", "complex128"]:
            return paddle.randn(tensor_config.shape, tensor_config.dtype)
        elif tensor_config.dtype in ["float16", "bfloat16", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"]:
            return paddle.rand(tensor_config.shape, "float64").cast(tensor_config.dtype)
        else:
            raise ValueError("not support")
        

    def test(self):
        pass
with open("../paddle_to_torch/paddle2torch_regular_dict.json", "r") as f:
    paddle_to_torch = json.load(f, object_pairs_hook=collections.OrderedDict)

class APITestAccuracy(APITestBase):
    def __init__(self, api_config):
        self.api_config = api_config
    def test(self):
        api = eval(self.api_config.api_name)
        paddle_to_torch_args_map = paddle_to_torch[self.api_config.api_name]["paddle_torch_args_map"]
        paddle_args_list = list(paddle_to_torch_args_map)
        torch_api = eval(paddle_to_torch[self.api_config.api_name]["torch_api"])
        args = []
        kwargs = {}
        merged_kwargs = {}
        index = 0
        for arg_config in self.api_config.args:
            if isinstance(arg_config, TensorConfig):
                value = self.rand_tensor(arg_config)
            else:
                value = arg_config
                if isinstance(value, list):
                    for i in range(len(value)):
                        if isinstance(value[i], TensorConfig):
                            value[i] = self.rand_tensor(value[i])
                if isinstance(value, tuple):
                    tmp = list(value)
                    for i in range(len(tmp)):
                        if isinstance(tmp[i], TensorConfig):
                            tmp[i] = self.rand_tensor(tmp[i])
                    value = tuple(tmp)
            args.append(value)
            merged_kwargs[paddle_args_list[index]] = value
            index = index + 1
        
        for key, arg_config in self.api_config.kwargs.items():
            if isinstance(arg_config, TensorConfig):
                value = self.rand_tensor(arg_config)
            else:
                value = arg_config
                if isinstance(value, list):
                    for i in range(len(value)):
                        if isinstance(value[i], TensorConfig):
                            value[i] = self.rand_tensor(value[i])
                if isinstance(value, tuple):
                    tmp = list(value)
                    for i in range(len(tmp)):
                        if isinstance(tmp[i], TensorConfig):
                            tmp[i] = self.rand_tensor(tmp[i])
                    value = tuple(tmp)
            kwargs[key] = value
            merged_kwargs[key] = value
        api(*tuple(args), **kwargs)
           
  

def analyse_configs(config_path):
    with open(config_path, "r") as f:
        configs = f.readlines()
        f.close()

    api_configs = []
    for config in configs:
        print(config.replace("\n", ""), "begin", flush=True)
        api_config = APIConfig(config)
        api_configs.append(api_config)
        print(api_config.config, " -> ", api_config, flush=True)
        case = APITestAccuracy(api_config)
        case.test()
        print("finish", flush=True)
    return api_configs

if __name__ == '__main__':
    api_configs = analyse_configs("../api_config/api_config.txt")
    for api_config in api_configs:
        print(api_config.config, " -> ", api_config, flush=True)
        case = APITestAccuracy(api_config)
        case.test()
        print("finish", flush=True)

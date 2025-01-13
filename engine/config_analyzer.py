import re
import collections
import paddle
import numpy
import math

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
    elif config[offset] == '\"':
        value = tocken
    elif tocken is None:
        return None, None
    else:
        value = eval(tocken)
    return value, offset

class APIConfig:
    def __init__(self, config):
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

    
def analyse_configs(config_path):
    with open(config_path, "r") as f:
        configs = f.readlines()
        f.close()

    api_configs = []
    for config in configs:
        print(config)
        api_config = APIConfig(config)
        print(api_config)
        api_configs.append(api_config)
        
    return api_configs

if __name__ == '__main__':
    api_configs = analyse_configs("../api_config/api_config.txt")
    for config in api_configs:
        config.test()

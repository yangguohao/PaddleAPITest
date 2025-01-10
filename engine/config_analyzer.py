import re
import collections

class APIConfig:
    def __init__(self, name):
        self.name = name
        self.args = []
        self.kwargs = collections.OrderedDict()
    
    def append_args(self, arg):
        self.args.append(arg)
        
    def append_kwargs(self, name, arg)
        self.kwargs[name] = arg


class TensorConfig:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
    def __str__(self):
        return "TensorConfig("+str(self.shape)+","+self.dtype+")"

def get_tocken(config, offset):
    pattern = r'\b[A-Za-z0-9]+\b'
    match = re.search(pattern, config[offset:])
    if match:
        return match.group(), match.start()
    return None, None

def get_list(config, offset):
    config = config[offset:]
    return config[config.index("["):config.index("]")+1], offset+config.index("[")

def get_api(config):
    return config[0:config.index("(")]

def get_tensor(config, offset):
    config = config[offset:]
    tensor_str = config[config.index("TensorConfig"):config.index(")")+1]
    return eval(tensor_str)
    
def analyse_config(config):
    config = config.replace("Tensor", "TensorConfig")

    api_name = get_api(config)
    offset = len(api_name)
    api = APIConfig(api_name)

    while(True):
        tocken, offset = get_tocken(config, offset)
        if tocken == "TensorConfig":
            t = get_tensor(config, offset)
            api.append_args(t)
        elif tocken is None:
            return api

        offset = offset + len(tocken)

    return api
        # shape, offset = get_list(config, offset)
        # print(tocken)
        # print(shape)
    
    
def analyse_configs(config_path):
    with open(config_path, "r") as f:
        configs = f.readlines()
        f.close()

    api_configs = []
    for config in configs:
        api_configs.append(analyse_config(config))
        return
        
    return api_configs

if __name__ == '__main__':
    analyse_configs("../api_config/api_config.txt")

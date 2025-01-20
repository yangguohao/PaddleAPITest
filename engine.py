from paddle_to_torch.paddle_to_torch import paddle_to_torch
from api_config import TensorConfig, APIConfig, analyse_configs

import re
import collections
import paddle
import numpy
import math
import json
import torch
import paddle
import inspect
import copy

class APITestBase:
    def __init__(self, api_config):
        self.api_config = api_config

    def ana_api_info(self):
        torch_api, paddle_to_torch_args_map = paddle_to_torch(api_config.api_name)
        if paddle_to_torch_args_map is None:
            print("[paddle_to_torch2]", self.api_config.config, "\napi need manual fix")
            return False
        self.paddle_api = eval(self.api_config.api_name)
        self.torch_api = eval(torch_api)

        api_sig = inspect.signature(self.paddle_api)
        api_args_list = list(api_sig.parameters.keys())

        self.paddle_args_config = self.api_config.args
        self.paddle_kwargs_config = self.api_config.kwargs
        self.paddle_merged_kwargs_config = {}
        index = 0
        for arg_config in self.api_config.args:
            self.paddle_merged_kwargs_config[api_args_list[index]] = arg_config
            index = index + 1

        for key, arg_config in self.api_config.kwargs.items():
            self.paddle_merged_kwargs_config[key] = arg_config

        self.torch_args_config = []
        self.torch_kwargs_config = {}

        for key, value in self.paddle_merged_kwargs_config.items():
            if key == "name":
                continue
            if key not in paddle_to_torch_args_map:
                print("[paddle_to_torch]", self.api_config.config, "\n ", key, "not in paddle_to_torch_args_map, can not call torch")
                return False

            self.torch_kwargs_config[paddle_to_torch_args_map[key]] = value

        self.paddle_args = []
        self.paddle_kwargs = {}
        self.paddle_merged_kwargs = {}

        for i in range(len(self.paddle_args_config)):
            if isinstance(self.paddle_args_config[i], TensorConfig):
                self.paddle_args.append(self.paddle_args_config[i].get_paddle_tensor())
            elif isinstance(self.paddle_args_config[i], list):
                tmp = []
                for j in range(len(self.paddle_args_config[i])):
                    if isinstance(self.paddle_args_config[i][j], TensorConfig):
                        tmp.append(self.paddle_args_config[i][j].get_paddle_tensor())
                    else:
                        tmp.append(self.paddle_args_config[i][j])
                self.paddle_args.append(tmp)
            elif isinstance(self.paddle_args_config[i], tuple):
                tmp = []
                for j in range(len(self.paddle_args_config[i])):
                    if isinstance(self.paddle_args_config[i][j], TensorConfig):
                        tmp.append(self.paddle_args_config[i][j].get_paddle_tensor())
                    else:
                        tmp.append(self.paddle_args_config[i][j])
                self.paddle_args.append(tuple(tmp))
            else:
                self.paddle_args.append(self.paddle_args_config[i])

        for key, arg_config in self.paddle_kwargs_config.items():
            if isinstance(arg_config, TensorConfig):
                self.paddle_kwargs[key] = arg_config.get_paddle_tensor()
            elif isinstance(arg_config, list):
                value = []
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        value.append(arg_config[i].get_paddle_tensor())
                    else:
                        value.append(arg_config[i])
                self.paddle_kwargs[key] = value
            elif isinstance(arg_config, tuple):
                tmp = []
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        tmp.append(arg_config[i].get_paddle_tensor())
                    else:
                        tmp.append(arg_config[i])
                self.paddle_kwargs[key] = tuple(tmp)
            else:
                self.paddle_kwargs[key] = arg_config

        for key, arg_config in self.paddle_merged_kwargs_config.items():
            if isinstance(arg_config, TensorConfig):
                self.paddle_merged_kwargs[key] = arg_config.get_paddle_tensor()
            elif isinstance(arg_config, list):
                value = []
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        value.append(arg_config[i].get_paddle_tensor())
                    else:
                        value.append(arg_config[i])
                self.paddle_merged_kwargs[key] = value
            elif isinstance(arg_config, tuple):
                tmp = []
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        tmp.append(arg_config[i].get_paddle_tensor())
                    else:
                        tmp.append(arg_config[i])
                self.paddle_merged_kwargs[key] = tuple(tmp)
            else:
                self.paddle_merged_kwargs[key] = arg_config

        self.torch_args = []
        self.torch_kwargs = {}

        for key, arg_config in self.torch_kwargs_config.items():
            if isinstance(arg_config, TensorConfig):
                self.torch_kwargs[key] = arg_config.get_torch_tensor()
            elif isinstance(arg_config, list):
                value = []
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        value.append(arg_config[i].get_torch_tensor())
                    else:
                        value.append(arg_config[i])
                self.torch_kwargs[key] = value
            elif isinstance(arg_config, tuple):
                tmp = []
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        tmp.append(arg_config[i].get_torch_tensor())
                    else:
                        tmp.append(arg_config[i])
                self.torch_kwargs[key] = tuple(tmp)
            else:
                self.torch_kwargs[key] = arg_config
        # print(self.torch_kwargs_config)
        # print(self.torch_kwargs)
        return True

    def np_assert_accuracy(
        self,
        np_paddle,
        np_torch,
        atol,
        rtol,
        api,
    ):
        if np_paddle.dtype == numpy.bool:
            numpy.testing.assert_equal(np_paddle, np_torch)
            return
        # max_atol_idx = numpy.argmax(numpy.abs(np_paddle - np_torch))
        # np_paddle_flatten = np_paddle.flatten()
        # np_torch_flatten = np_torch.flatten()
        # sub_res = np_paddle_flatten - np_torch_flatten
        # nonzero_idx = numpy.nonzero(np_torch_flatten)
        # sub_res = sub_res.take(nonzero_idx)
        # np_torch_flatten_nonzero = np_torch_flatten.take(nonzero_idx).flatten()
        # np_paddle_flatten_nonzero = np_paddle_flatten.take(nonzero_idx).flatten()
        # if sub_res.size ==0:
        #     max_rtol_idx = 0
        # else:
        #     max_rtol_idx = numpy.argmax(numpy.abs(sub_res / np_torch_flatten_nonzero))
        numpy.testing.assert_allclose(
            np_paddle,
            np_torch,
            atol,
            rtol,
            # err_msg=(
            #     '{api}: compare failed,\n'.format(
            #         api=api,
            #     )
            #     + 'max_atol value, paddle_value: {value_paddle}, torch_value: {value_torch},\n'.format(
            #         value_paddle=str(np_paddle_flatten[max_atol_idx].item()),
            #         value_torch=str(np_torch_flatten[max_atol_idx].item()),
            #     )
            #     + 'max_rtol value , torch_value: {value_paddle}, paddle_value: {value_torch},\n'.format(
            #         value_paddle=str(np_paddle_flatten_nonzero[max_rtol_idx].item()) if max_rtol_idx < len(np_paddle_flatten_nonzero) else '',
            #         value_torch=str(np_torch_flatten_nonzero[max_rtol_idx].item()) if max_rtol_idx < len(np_torch_flatten_nonzero) else '',
            #     )
            # ),
        )

    def test(self):
        pass

class APITestAccuracy(APITestBase):
    def __init__(self, api_config):
        self.api_config = api_config
    def test(self):
        if not self.ana_api_info():
            return

        device = torch.device("cuda:0")
        torch.set_default_device(device)

        try:
            torch_output = self.torch_api(*tuple(self.torch_args), **self.torch_kwargs)
        except Exception as err:
            print("[torch error]", self.api_config.config, "\n", str(err))
            return

        try:
            paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
        except Exception as err:
            print("[paddle error]", self.api_config.config, "\n", str(err))
            return

        if isinstance(paddle_output, paddle.Tensor):
            if isinstance(torch_output, torch.Tensor):
                try:
                    self.np_assert_accuracy(paddle_output.numpy(), torch_output.cpu().numpy(), 1e-2, 1e-2, self.api_config)
                except Exception as err:
                    print("[accuracy error]", self.api_config.config, "\n", str(err))
                    return
            else:
                print("[output type diff error]", self.api_config.config)
        elif isinstance(paddle_output, (list, tuple)):
            if isinstance(paddle_output, tuple):
                paddle_output = list(paddle_output)
            if not isinstance(torch_output, (list, tuple)):
                print("[output type diff error]", self.api_config.config)
                return
            if isinstance(torch_output, tuple):
                torch_output = list(torch_output)
            if len(paddle_output) != len(torch_output):
                print("[output type diff error]", self.api_config.config)
                return
            for i in range(len(paddle_output)):
                try:
                    self.np_assert_accuracy(paddle_output[i].numpy(), torch_output[i].cpu().numpy(), 1e-2, 1e-2, self.api_config)
                except Exception as err:
                    print("[accuracy error]", self.api_config.config, "\n", str(err))
                    return

        print("[Pass]", self.api_config.config)
  

if __name__ == '__main__':
    api_configs = analyse_configs("api_config/api_config.txt")
    for api_config in api_configs:
        case = APITestAccuracy(api_config)
        case.test()

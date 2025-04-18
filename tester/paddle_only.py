import os

import filelock
import paddle
from filelock import FileLock
from func_timeout import func_set_timeout

from .base import APITestBase

DIR_PATH = os.path.dirname(os.path.realpath(__file__))[0:os.path.dirname(os.path.realpath(__file__)).index("PaddleAPITest")+13]
TEST_LOG_PATH = os.path.join(DIR_PATH, "tester/api_config/test_log")
LOG_FILES = {
    "paddle_error": os.path.join(TEST_LOG_PATH, "api_config_paddle_error.txt"),
    "pass": os.path.join(TEST_LOG_PATH, "api_config_pass.txt"),
}

def write_to_log(log_type, message):
    if log_type not in LOG_FILES:
        print(f"Invalid log type: {log_type}")
        return
    log_file = LOG_FILES[log_type]
    lock_file = log_file + ".lock"
    try:
        with FileLock(lock_file, timeout=10):
            with open(log_file, "a") as f:
                f.write(message + "\n")
                f.flush()
    except filelock.Timeout:
        print(f"Timeout waiting for lock on {log_file}")
    except Exception as e:
        print(f"Error writing to {log_file}: {str(e)}")

class APITestPaddleOnly(APITestBase):
    def __init__(self, api_config, test_amp):
        super().__init__(api_config)
        self.test_amp = test_amp
    
    @func_set_timeout(600)
    def test(self):
        if self.need_skip():
            print("[Skip]")
            return

        if not self.ana_paddle_api_info():
            print("ana_paddle_api_info failed")
            return

        try:
            if not self.gen_paddle_input():
                print("gen_paddle_input failed")
                return  
            if self.test_amp:
                with paddle.amp.auto_cast():
                    paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)      
            else:
                paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
            if self.need_check_grad():
                inputs_list = self.get_paddle_input_list()
                result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(paddle_output)
                if len(inputs_list) != 0 and len(result_outputs) != 0 and len(result_outputs_grads) != 0:
                    out_grads = paddle.grad(result_outputs, inputs_list, grad_outputs=result_outputs_grads,allow_unused=True)
        except Exception as err:
            paddle_output = None
            result_outputs = None
            result_outputs_grads = None
            out_grads = None
            if "gradient_accumulator.cc" in str(err) or "Out of memory" in str(err):
                return
            print("[paddle error]", self.api_config.config, "\n", str(err))
            write_to_log("paddle_error", self.api_config.config)
            if "CUDA error" in str(err) or "memory corruption" in str(err):
                raise Exception(err)
            return

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            print("[cuda error]", self.api_config.config, "\n", str(err))
            paddle_output = None
            result_outputs = None
            result_outputs_grads = None
            out_grads = None
            write_to_log("paddle_error", self.api_config.config)
            return

        paddle_output = None
        result_outputs = None
        result_outputs_grads = None
        out_grads = None
        print("[Pass]", self.api_config.config)
        write_to_log("pass", self.api_config.config)
  

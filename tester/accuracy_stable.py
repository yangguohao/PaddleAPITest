import traceback

import paddle
import torch
import numpy

from .api_config.log_writer import log_accuracy_stable, write_to_log
from .base import APITestBase
from .paddle_to_torch import get_converter


cuda_errors = frozenset(
    [
        "CUDA error",
        "memory corruption",
        "CUDA out of memory",
        "Out of memory error",
    ]
)


class APITestAccuracyStable(APITestBase):
    def __init__(self, api_config, **kwargs):
        super().__init__(api_config)
        self.test_amp = kwargs.get("test_amp", False)
        self.converter = get_converter()
        torch.set_printoptions(
            profile="short", edgeitems=2, threshold=100, linewidth=120
        )
        torch.set_default_device("cuda")

    def test(self):
        if self.need_skip():
            print("[Skip]", flush=True)
            return

        if not self.ana_api_info():
            print("ana_api_info failed", flush=True)
            return

        try:
            convert_result = self.converter.convert(self.api_config.api_name)
        except Exception as e:
            print(
                f"[paddle_to_torch] Convertion failed for {self.api_config.config}: {str(e)}",
                flush=True,
            )
            write_to_log("paddle_to_torch_failed", self.api_config.config)
            return
        if not convert_result.is_supported:
            print(
                f"[paddle_to_torch] Unsupported API {self.api_config.api_name}: {convert_result.error_message}",
                flush=True,
            )
            write_to_log("paddle_to_torch_failed", self.api_config.config)
            return
        if not convert_result.code or not convert_result.code.is_valid():
            print(
                f"[paddle_to_torch] No code generated for {self.api_config.api_name}",
                flush=True,
            )
            write_to_log("paddle_to_torch_failed", self.api_config.config)
            return

        try:
            if not self.gen_numpy_input():
                print("gen_numpy_input failed")
                return
        except Exception as err:
            print("[numpy error]", self.api_config.config, "\n", str(err))
            traceback.print_exc()
            write_to_log("numpy_error", self.api_config.config)
            return

        torch_output_pair = []
        torch_grad_pair = []
        paddle_output_pair = []
        paddle_grad_pair = []

        # iter twice
        for i in range(2):
            # ======== torch ========
            torch_output, torch_out_grads, torch_grad_success = self.get_torch_output(
                convert_result
            )
            if torch_output is None:
                return
            torch.cuda.empty_cache()

            # ======== paddle ========
            paddle_output, paddle_out_grads = self.get_paddle_output(torch_grad_success)
            if paddle_output is None:
                return
            paddle.device.cuda.empty_cache()

            # ======== format ========
            torch_output, torch_out_grads, paddle_output, paddle_out_grads = (
                self.format_output(
                    torch_output, torch_out_grads, paddle_output, paddle_out_grads
                )
            )

            # ======== add to pair ========
            # if torch_grad_success = False, out_grads = [] and compare return
            torch_output_pair.append(torch_output)
            torch_grad_pair.append(torch_out_grads)
            paddle_output_pair.append(paddle_output)
            paddle_grad_pair.append(paddle_out_grads)

        # ======== summary ========
        self.compare(torch_output_pair[0], paddle_output_pair[0], "T1P1")
        self.compare(torch_grad_pair[0], paddle_grad_pair[0], "T1P1B")
        self.compare(torch_output_pair[1], paddle_output_pair[1], "T2P2")
        self.compare(torch_grad_pair[1], paddle_grad_pair[1], "T2P2B")
        self.compare(torch_output_pair[0], paddle_output_pair[1], "T1P2")
        self.compare(torch_grad_pair[0], paddle_grad_pair[1], "T1P2B")
        self.compare(torch_output_pair[1], paddle_output_pair[0], "T2P1")
        self.compare(torch_grad_pair[1], paddle_grad_pair[0], "T2P1B")
        self.compare(torch_output_pair[0], torch_output_pair[1], "T1T2")
        self.compare(torch_grad_pair[0], torch_grad_pair[1], "T1T2B")
        self.compare(paddle_output_pair[0], paddle_output_pair[1], "P1P2")
        self.compare(paddle_grad_pair[0], paddle_grad_pair[1], "P1P2B")

        print(f"[Pass] {self.api_config.config}", flush=True)
        write_to_log("pass", self.api_config.config)

    def get_torch_output(self, convert_result):
        # ======== run torch forward ========:
        torch_output = None
        try:
            if not self.gen_torch_input():
                print("gen_torch_input failed", flush=True)
                return None, None, None

            exec_globals = {"torch": torch}
            exec_locals = {
                "args": self.torch_args,
                "kwargs": self.torch_kwargs,
                "result": None,
                **self.torch_kwargs,
            }
            if self.api_config.api_name == "paddle.nn.functional.rnnt_loss":
                if paddle.device.get_device() == "cpu":
                    exec_locals["fused_log_softmax"] = False

            code = convert_result.code
            if code.preprocess_compiled:
                exec(code.preprocess_compiled, exec_globals, exec_locals)
            if code.core_compiled:
                if self.test_amp:
                    with torch.autocast(device_type="cuda"):
                        exec(code.core_compiled, exec_globals, exec_locals)
                else:
                    exec(code.core_compiled, exec_globals, exec_locals)
            if code.postprocess_compiled:
                exec(code.postprocess_compiled, exec_globals, exec_locals)

            output_var = convert_result.output_var or "result"
            torch_output = exec_locals[output_var]
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            err_str = str(err)
            print(f"[torch error] {self.api_config.config}\n{err_str}", flush=True)
            traceback.print_exc()
            write_to_log("torch_error", self.api_config.config)
            if any(cuda_err in err_str for cuda_err in cuda_errors):
                raise
            return None, None, None

        # ======== run torch backward ========
        torch_grad_success = False
        torch_out_grads = []
        if self.need_check_grad():
            try:
                inputs_list = self.get_torch_input_list()
                result_outputs, result_outputs_grads = (
                    self.gen_torch_output_and_output_grad(torch_output)
                )
                if inputs_list and result_outputs and result_outputs_grads:
                    torch_out_grads = torch.autograd.grad(
                        outputs=result_outputs,
                        inputs=inputs_list,
                        grad_outputs=result_outputs_grads,
                    )
                    torch_grad_success = True
            except Exception as err:
                err_str = str(err)
                if err_str.startswith("Too large tensor to get cached numpy: "):
                    print(
                        f"[numpy error] {self.api_config.config}\n{err_str}",
                        flush=True,
                    )
                    write_to_log("numpy_error", self.api_config.config)
                    return None, None, None
                print(err_str, flush=True)
                if any(cuda_err in err_str for cuda_err in cuda_errors):
                    raise

            try:
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                err_str = str(err)
                print(
                    f"[torch error] backward {self.api_config.config}\n{err_str}",
                    flush=True,
                )
                traceback.print_exc()
                write_to_log("torch_error", self.api_config.config)
                raise

        def process_torch_outputs(obj):
            if isinstance(obj, (torch.return_types.max, torch.return_types.min)):
                obj = obj.values
            if isinstance(obj, (list, tuple)):
                obj = list(obj)
            return obj

        torch_output = process_torch_outputs(torch_output)
        torch_out_grads = process_torch_outputs(torch_out_grads)
        return torch_output, torch_out_grads, torch_grad_success

    def get_paddle_output(self, torch_grad_success):
        # ======== run paddle forward ========
        paddle_output = None
        try:
            if not self.gen_paddle_input():
                print("gen_paddle_input failed")
                return None, None

            # determine the dtype
            self.api_config.dtype = None
            for arg in self.paddle_args:
                if isinstance(arg, paddle.Tensor):
                    self.api_config.dtype = arg.dtype
                    break
            if self.api_config.dtype is None:
                for arg in self.paddle_kwargs.values():
                    if isinstance(arg, paddle.Tensor):
                        self.api_config.dtype = arg.dtype
                        break
            # if there is no tensor in args and kwargs, use float32 as default
            if self.api_config.dtype is None:
                self.api_config.dtype = paddle.float32

            # find the first arg
            first_arg = (
                self.paddle_args[0]
                if len(self.paddle_args) > 0
                else next(iter(self.paddle_kwargs.values()))
            )
            if "paddle.Tensor." in self.api_config.api_name:
                api_name = self.api_config.api_name.split(".")[-1]
                api = getattr(self.paddle_args[0], api_name)
                if self.test_amp:
                    with paddle.amp.auto_cast():
                        paddle_output = api(*self.paddle_args[1:], **self.paddle_kwargs)
                else:
                    paddle_output = api(*self.paddle_args[1:], **self.paddle_kwargs)
            else:
                if self.test_amp:
                    with paddle.amp.auto_cast():
                        paddle_output = self.paddle_api(
                            *self.paddle_args, **self.paddle_kwargs
                        )
                else:
                    paddle_output = self.paddle_api(
                        *self.paddle_args, **self.paddle_kwargs
                    )
            if (
                self.api_config.api_name[-1] == "_"
                and self.api_config.api_name[-2:] != "__"
            ) or self.api_config.api_name == "paddle.Tensor.__setitem__":
                paddle_output = first_arg
        except Exception as err:
            err_str = str(err)
            if self.should_ignore_paddle_error(err_str):
                print(f"[Pass] {self.api_config.config}", flush=True)
                write_to_log("pass", self.api_config.config)
                return None, None
            print(f"[paddle error] {self.api_config.config}\n{err_str}", flush=True)
            traceback.print_exc()
            write_to_log("paddle_error", self.api_config.config)
            if any(cuda_err in err_str for cuda_err in cuda_errors):
                raise
            return None, None

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            print(f"[cuda error] {self.api_config.config}\n{str(err)}", flush=True)
            write_to_log("paddle_error", self.api_config.config)
            raise

        # ======== run paddle backward ========
        paddle_out_grads = []
        if torch_grad_success:
            try:
                inputs_list = self.get_paddle_input_list()
                result_outputs, result_outputs_grads = (
                    self.gen_paddle_output_and_output_grad(paddle_output)
                )
                if inputs_list and result_outputs and result_outputs_grads:
                    paddle_out_grads = paddle.grad(
                        result_outputs,
                        inputs_list,
                        grad_outputs=result_outputs_grads,
                        allow_unused=True,
                    )
            except Exception as err:
                err_str = str(err)
                if err_str.startswith("Too large tensor to get cached numpy: "):
                    print(
                        f"[numpy error] {self.api_config.config}\n{err_str}",
                        flush=True,
                    )
                    write_to_log("numpy_error", self.api_config.config)
                    return None, None
                if self.should_ignore_paddle_error(err_str):
                    print(f"[Pass] {self.api_config.config}", flush=True)
                    write_to_log("pass", self.api_config.config)
                    return None, None
                print(
                    f"[paddle error] backward {self.api_config.config}\n{err_str}",
                    flush=True,
                )
                traceback.print_exc()
                write_to_log("paddle_error", self.api_config.config)
                if any(cuda_err in err_str for cuda_err in cuda_errors):
                    raise
                return None, None

            try:
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                print(
                    f"[cuda error] backward {self.api_config.config}\n{str(err)}",
                    flush=True,
                )
                write_to_log("paddle_error", self.api_config.config)
                raise

        def process_paddle_outputs(obj):
            if isinstance(obj, (list, tuple)):
                obj = list(obj)
            return obj

        paddle_output = process_paddle_outputs(paddle_output)
        paddle_out_grads = process_paddle_outputs(paddle_out_grads)
        return paddle_output, paddle_out_grads

    def format_output(
        self, paddle_output, torch_output, paddle_out_grads, torch_out_grads
    ):
        # ======== format output ========
        if self.api_config.api_name == "paddle.incubate.nn.functional.fused_rms_norm":
            paddle_output = paddle_output[0]
        elif self.api_config.api_name == "paddle.unique":
            if "return_index=True" in self.api_config.config:
                paddle_output = list(paddle_output)
                paddle_output.pop(1)
            paddle_output = tuple(paddle_output)
        elif self.api_config.api_name in {
            "paddle.mode",
            "paddle.Tensor.mode",
            "paddle.incubate.nn.functional.fused_layer_norm",
            "paddle.kthvalue",
        }:
            paddle_output = paddle_output[0]
            torch_output = torch_output[0]
        elif self.api_config.api_name in {
            "paddle.strided_slice",
            "paddle.vander",
        } and any(s < 0 for s in paddle_output.strides):
            # torch's from_dlpack now don't support negative strides
            paddle_output = paddle_output.contiguous()
        elif self.api_config.api_name == "paddle.linalg.eigh":
            # The output of eigen vectors are not unique, because multiplying an eigen vector by -1 in the real case
            # or by e^(i*\theta) in the complex case produces another set of valid eigen vectors of the matrix.
            # So we test whether the elements of each coef_vector (i.e. paddle_output / torch_output for each eigen vector)
            # are all the same and whether the |coef| == 1 for simplicity.
            paddle_output, torch_output = list(paddle_output), list(torch_output)
            eigvector_len = paddle_output[1].shape[-2]
            paddle_eigvectors = (
                paddle_output.pop(1).matrix_transpose().reshape([-1, eigvector_len])
            )
            torch_eigvectors = (
                torch_output.pop(1).transpose(-1, -2).reshape((-1, eigvector_len))
            )
            paddle_output, torch_output = [], []
            for i in range(paddle_eigvectors.shape[0]):
                coef_vector = paddle.to_tensor(
                    paddle_eigvectors[i].numpy() / torch_eigvectors[i].numpy(),
                    dtype=paddle_eigvectors[i].dtype,
                )
                coef_vector = coef_vector.round(2)
                coef_0 = (
                    paddle_eigvectors[i].numpy()[0] / torch_eigvectors[i].numpy()[0]
                )
                coef_vector_approx = torch.tensor([coef_0] * eigvector_len)
                abs_coef = coef_vector.abs().astype("float64")[0]
                one = torch.tensor(1.0, dtype=torch.float64)
                paddle_output.append([coef_vector, abs_coef])
                torch_output.append([coef_vector_approx, one])

        # ======== format gradient ========
        if self.api_config.api_name == "paddle.Tensor.__setitem__":
            torch_out_grads = torch_out_grads[0]
            paddle_out_grads = paddle_out_grads[0]
        # All configs that not compared with torch should be copied
        # to tester/api_config/5_accuracy/accuracy_gpu_error_grads_diff.txt
        if self.api_config.api_name in {
            "paddle.nn.functional.scaled_dot_product_attention",
        }:
            paddle_out_grads = paddle_out_grads[:3]
            torch_out_grads = torch_out_grads[:3]
        elif self.api_config.api_name in {
            "paddle.lerp",
            "paddle.tensordot",
        }:
            paddle_out_grads = paddle_out_grads[:2]
            torch_out_grads = torch_out_grads[:2]
        elif self.api_config.api_name in {
            "paddle.Tensor.fill_diagonal_tensor",
            "paddle.diagonal_scatter",
            "paddle.incubate.softmax_mask_fuse",
            "paddle.nn.functional.binary_cross_entropy",
            "paddle.nn.functional.binary_cross_entropy_with_logits",
            "paddle.nn.functional.cross_entropy",
            "paddle.nn.functional.sigmoid_focal_loss",
            "paddle.nn.functional.gaussian_nll_loss",
            "paddle.nn.functional.kl_div",
            "paddle.scale",
        }:
            paddle_out_grads = paddle_out_grads[:1]
            torch_out_grads = torch_out_grads[:1]
        elif self.api_config.api_name in {
            "paddle.combinations",
            "paddle.nn.utils.parameters_to_vector",
            "paddle.cdist",
        }:
            paddle_out_grads = []
            torch_out_grads = []

        return paddle_output, torch_output, paddle_out_grads, torch_out_grads

    def compare(self, input1, input2, comp):
        if isinstance(input1, (paddle.Tensor, torch.Tensor)):
            if isinstance(input2, (paddle.Tensor, torch.Tensor)):
                try:
                    self.assert_accuracy(input1, input2, comp)
                except Exception as err:
                    print(
                        f"[{comp}] [accuracy error] {self.api_config.config}\n{str(err)}",
                        flush=True,
                    )
                    write_to_log("accuracy_error", self.api_config.config)
                    return
            else:
                print(
                    f"[{comp}] [accuracy error] {self.api_config.config}\n[not compare],",
                    f"{type(input1)} / {type(input2)}",
                    flush=True,
                )
                write_to_log("accuracy_error", self.api_config.config)
                return
        elif isinstance(input1, (list, tuple)):
            if not isinstance(input2, (list, tuple)):
                print(
                    f"[{comp}] [accuracy error] {self.api_config.config}\n[not compare],",
                    f"{type(input1)} / {type(input2)}",
                    flush=True,
                )
                write_to_log("accuracy_error", self.api_config.config)
                return
            if len(input1) != len(input2):
                print(
                    f"[{comp}] [accuracy error] {self.api_config.config}\n[not compare],",
                    f"{type(input1)} : {len(input1)} /",
                    f"{type(input2)} : {len(input2)}",
                    flush=True,
                )
                write_to_log("accuracy_error", self.api_config.config)
                return
            for item1, item2 in zip(input1, input2):
                if isinstance(item1, (paddle.Tensor, torch.Tensor)) and isinstance(
                    item2, (paddle.Tensor, torch.Tensor)
                ):
                    try:
                        self.assert_accuracy(item1, item2, comp)
                    except Exception as err:
                        print(
                            f"[{comp}] [accuracy error] {self.api_config.config}\n{str(err)}",
                            flush=True,
                        )
                        write_to_log("accuracy_error", self.api_config.config)
                        return
                elif not isinstance(
                    item1, (paddle.Tensor, torch.Tensor)
                ) and not isinstance(item2, (paddle.Tensor, torch.Tensor)):
                    try:
                        self.assert_accuracy(
                            torch.tensor(item1), torch.tensor(item2), comp
                        )
                    except Exception as err:
                        print(
                            f"[{comp}] [accuracy error] {self.api_config.config}\n{str(err)}",
                            flush=True,
                        )
                        write_to_log("accuracy_error", self.api_config.config)
                        return
                else:
                    print(
                        f"[{comp}] [accuracy error] {self.api_config.config}\n[not compare]",
                        f"{type(item1)} / {type(item2)}",
                        flush=True,
                    )
                    write_to_log("accuracy_error", self.api_config.config)
                    return
        else:
            try:
                self.assert_accuracy(torch.tensor(input1), torch.tensor(input2), comp)
            except Exception as err:
                print(
                    f"[{comp}] [accuracy error] {self.api_config.config}\n{str(err)}",
                    flush=True,
                )
                write_to_log("accuracy_error", self.api_config.config)
                return

    def assert_accuracy(self, tensor1, tensor2, comp):
        if not tensor1.is_contiguous():
            tensor1 = tensor1.contiguous()
        if not tensor2.is_contiguous():
            tensor2 = tensor2.contiguous()

        api_name = self.api_config.api_name
        config = self.api_config.config[:120000]
        dtype = self.api_config.dtype
        check_dtype = self.should_check_dtype()

        first = "Paddle" if comp[0] == "P" else "Torch"
        second = "Paddle" if comp[2] == "P" else "Torch"

        if isinstance(tensor1, paddle.Tensor):
            dlpack = paddle.utils.dlpack.to_dlpack(tensor1)  # type: ignore
            tensor1 = torch.utils.dlpack.from_dlpack(dlpack)  # type: ignore
        if isinstance(tensor2, paddle.Tensor):
            dlpack = paddle.utils.dlpack.to_dlpack(tensor2)  # type: ignore
            tensor2 = torch.utils.dlpack.from_dlpack(dlpack)  # type: ignore

        def error_msg(msg):
            return (
                f"Not equal to tolerance rtol=0.0, atol=0.0\n"
                f"{msg}\n"
                f"{first}: (shape={tensor1.shape}, dtype={tensor1.dtype})\n"
                f"{tensor1}\n"
                f"{second}: (shape={tensor2.shape}, dtype={tensor2.dtype})\n"
                f"{tensor2}"
            )

        try:
            torch.testing.assert_close(
                tensor1,
                tensor2,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
                check_device=False,
                check_dtype=check_dtype,
                msg=error_msg,
            )
            log_accuracy_stable(
                "Identical",
                api_name,
                config,
                dtype,
                comp,
            )
        except Exception as err:
            err_str = str(err)
            is_acc_err = False
            if err_str.startswith("Comparing"):
                print(f"torch_assert failed, try np_assert", flush=True)
                try:
                    numpy.testing.assert_allclose(
                        tensor1.cpu().numpy(),
                        tensor2.cpu().numpy(),
                        rtol=0.0,
                        atol=0.0,
                        equal_nan=True,
                        strict=True,
                    )
                    log_accuracy_stable(
                        "Identical",
                        api_name,
                        config,
                        dtype,
                        comp,
                    )
                except Exception as err_np:
                    err_str = str(err_np)
                    err_list = err_str.split("\n", maxsplit=3)
                    if len(err_list) > 3 and err_list[3].startswith(
                        "Mismatched elements"
                    ):
                        is_acc_err = True
            else:
                err_list = err_str.split("\n", maxsplit=1)
                if len(err_list) > 1 and (
                    err_list[1].startswith("Tensor-likes")
                    or err_list[1].startswith("Scalars")
                ):
                    is_acc_err = True
            if is_acc_err:
                log_accuracy_stable(
                    err_str,
                    api_name,
                    config,
                    dtype,
                    comp,
                )
            else:
                raise

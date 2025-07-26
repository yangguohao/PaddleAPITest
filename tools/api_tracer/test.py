# demo.py
import torch
import os

from api_tracer import APITracer

# def setup_custom_op():
#     """模拟加载一个C++自定义算子库"""


def run_pytorch_code():
    print("--- [Demo] Running PyTorch code to be traced ---")
    # 1. 常规API调用
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cpu")
    c = torch.matmul(a, b)
    d = torch.relu(c)
    e = torch.add(d, 10)
    f = torch.nn.functional.softmax(e, dim=1)
    g = torch.cat((a, b), dim=0)

    # # 2. 自定义算子调用
    # print("\n--- [Demo] Calling custom operator ---")
    # h =
    print("--- [Demo] PyTorch code finished ---\n")


def main():
    output_dir = "trace_output"
    os.makedirs(output_dir, exist_ok=True)
    trace_file = os.path.join(output_dir, "pytorch_api_trace.yaml")

    # setup_custom_op()

    # 步骤 1: 初始化工具链
    tracer = APITracer(dialect="torch", output_path=trace_file)

    # 步骤 2: 启动抓取
    try:
        tracer.start()

        # 步骤 3: 运行你的目标PyTorch代码
        run_pytorch_code()

    finally:
        # 步骤 4: 停止抓取
        tracer.stop()


if __name__ == "__main__":
    main()

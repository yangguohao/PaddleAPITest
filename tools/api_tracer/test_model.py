import json
import os
import sys

import yaml

from tools.api_tracer import APITracer

os.environ["HF_HOME"] = (
    "/root/paddlejob/workspace/env_run/lihaoyang/PaddleAPITest/tools/api_tracer/.huggingface"
)

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_CONFIGS = {
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "params": {"max_length": 100, "temperature": 0.7, "do_sample": True},
    },
    # "qwen3": {
    #     "name": "Qwen/Qwen3-0.6B",
    #     "params": {"max_length": 100, "temperature": 0.7, "do_sample": True},
    # },
    # "deepseek-r1": {
    #     "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    #     "params": {"max_length": 100, "temperature": 0.7, "do_sample": True},
    # },
    # "ernie-4.5": {
    #     "name": "baidu/ERNIE-4.5-0.3B-PT",
    #     "params": {"max_length": 100, "temperature": 0.7, "do_sample": True},
    # },
}


def main():
    prompt = "你好！请告诉我如何学习 PyTorch?"

    for model_key, config in MODEL_CONFIGS.items():
        print(f"Running {model_key}...")
        tokenizer = AutoTokenizer.from_pretrained(config["name"])
        model = AutoModelForCausalLM.from_pretrained(config["name"])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        params = config["params"]
        with APITracer("torch", f"tools/api_tracer/trace_output/{model_key}") as tracer:
            outputs = model.generate(
                inputs["input_ids"],
                max_length=params["max_length"],
                num_return_sequences=1,
                temperature=params["temperature"],
                do_sample=params["do_sample"],
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated Response:", response)

        del model, tokenizer
        torch.cuda.empty_cache()


# api_calls = []

# def trace_function(frame, event, arg):
#     if event == "call":
#         code = frame.f_code
#         if "torch" in code.co_filename:
#             func_name = code.co_name
#             args = frame.f_locals
#             api_calls.append({"function": func_name, "module": "torch", "args": args})
#     return trace_function

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     with record_function("model_inference"):

# sys.setprofile(trace_function)

# <code>

# sys.setprofile(None)

# print(prof.key_averages().table(sort_by="cpu_time_total"))
# prof.export_chrome_trace("tools/api_tracer/trace_output/trace.json")

# with open("tools/api_tracer/trace_output/api_calls.json", "w", encoding="utf-8") as f:
#     json.dump(api_calls, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()

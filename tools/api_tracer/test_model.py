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

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = "你好！请告诉我如何学习 PyTorch?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

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

with APITracer("torch", "tools/api_tracer/trace_output") as tracer:
    outputs = model.generate(
        inputs["input_ids"],
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
    )

# sys.setprofile(None)

# print(prof.key_averages().table(sort_by="cpu_time_total"))
# prof.export_chrome_trace("tools/api_tracer/trace_output/trace.json")

# with open("tools/api_tracer/trace_output/api_calls.json", "w", encoding="utf-8") as f:
#     json.dump(api_calls, f, indent=2, ensure_ascii=False)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Response:", response)

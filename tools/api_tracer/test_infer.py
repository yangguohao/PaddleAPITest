import json
import os
import sys
import traceback

import yaml

from tools.api_tracer import APITracer

os.environ["HF_HOME"] = "tools/api_tracer/.huggingface"

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    # "Qwen/Qwen2-0.5B",
    # "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-30B-A3B",
    # "deepseek-ai/DeepSeek-V2-Lite",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "baidu/ERNIE-4.5-0.3B-PT",
    # "baidu/ERNIE-4.5-21B-A3B-PT",
]


def run_inference_test(model_name: str):
    print(f"🚀 Running inference test for: {model_name}")
    true_model_name = "/".join(model_name.rsplit("/", 2)[-2:])
    output_path = f"tools/api_tracer/trace_output_test_infer/{true_model_name}"
    tracer = APITracer(
        "torch", output_path=output_path, levels=[0, 1], merge_output=True
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Model Class: {model.__class__}")
        print(f"Tokenizer Class: {tokenizer.__class__}")

        with open(os.path.join(output_path, "model_info.txt"), "w") as f:
            f.write(f"Model: {model.__class__}\n")
            f.write(f"Tokenizer: {tokenizer.__class__}\n")

        prompt = "Hello! Can you tell me how to learn PyTorch?"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad() and tracer:
            outputs = model.generate(
                inputs["input_ids"],
                num_return_sequences=1,
                max_length=100,
                temperature=0.7,
                do_sample=True,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n--- Generated Response ---")
        print(response)
        print("--------------------------\n")
        print(f"✅ Test for {true_model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during inference for {true_model_name}: {e}")


def main():
    for model_name in MODELS:
        run_inference_test(model_name)


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

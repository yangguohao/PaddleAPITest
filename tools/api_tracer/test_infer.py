import os

os.environ["HF_HOME"] = "tools/api_tracer/.huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import traceback

import numpy as np
import torch
from api_tracer import APITracer
from decord import VideoReader, cpu
from diffusers.pipelines.auto_pipeline import (AutoPipelineForImage2Image,
                                               AutoPipelineForText2Image)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from PIL import Image
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForImageTextToText, AutoProcessor,
                          AutoTokenizer)

TextGenerationMODELS = [
    # "Qwen/Qwen2-0.5B",
    # "Qwen/Qwen2-57B-A14B",
    # "Qwen/Qwen2.5-0.5B",
    # "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-30B-A3B",
    # "meta-llama/Llama-2-7b-hf",
    # "meta-llama/Llama-3.1-8B"
    # "deepseek-ai/DeepSeek-V2-Lite",
    # "deepseek-ai/DeepSeek-V3",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "baidu/ERNIE-4.5-0.3B-PT",
    # "baidu/ERNIE-4.5-21B-A3B-PT",
    # "moonshotai/Kimi-K2-Instruct",
    # "zai-org/GLM-4.5",
    # "mistralai/Magistral-Small-2507",
    # "MiniMaxAI/MiniMax-M1-40k",
    # "state-spaces/mamba2-2.7b",
    # "RWKV/RWKV7-Goose-World3-2.9B-HF",  #  maybe fail, change to fla-hub/rwkv7-2.9B-world
]

ImageTexttoTextModels = [
    # "Qwen/Qwen2.5-VL-3B-Instruct",
    # "deepseek-ai/deepseek-vl2-tiny",  # need to clone deepseek_vl2 project
    # "llava-hf/llava-1.5-7b-hf",
    # "meta-llama/Llama-4-Maverick-17B-128E",
    # "baidu/ERNIE-4.5-VL-28B-A3B-PT",
    # "zai-org/GLM-4.1V-9B-Thinking",
    # "ByteDance/Dolphin",
    # "Salesforce/blip2-opt-2.7b",
    # "OpenGVLab/InternVL3-1B",
    # "moonshotai/Kimi-VL-A3B-Instruct",
    # "XiaomiMiMo/MiMo-VL-7B-SFT",
    # "echo840/MonkeyOCR",
]

VideoTexttoTextModels = [
    # "Kwai-Keye/Keye-VL-8B-Preview",
]

TexttoImageModels = [
    # "stabilityai/stable-diffusion-3-medium-diffusers",
    # "black-forest-labs/FLUX.1-dev",
    # "jieliu/SD3.5M-FlowGRPO-GenEval",
]

TexttoVideoModels = [
    # "Wan-AI/Wan2.1-T2V-14B",
]

Imageto3DModels = [
    # "tencent/HunyuanWorld-1",
]

AnytoAnyModels = [
    # "deepseek-ai/Janus-Pro-1B",
    # "ByteDance-Seed/BAGEL-7B-MoT",
]


def run_inference_test_tg(model_name: str, apply_template: bool = False):
    print(f"🚀 Running Text Generation Inference Test for: {model_name}")
    true_model_name = "/".join(model_name.rsplit("/", 2)[-2:])
    output_path = f"tools/api_tracer/trace_output_test_infer/{true_model_name}"
    tracer = APITracer(
        "torch",
        output_path=output_path,
        levels=[0, 1],
        merge_output=True,
        record_stack=True,
        stack_format="full",
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        print(f"Model Class: {model.__class__}")
        print(f"Tokenizer Class: {tokenizer.__class__}")

        with open(os.path.join(output_path, "model_info.txt"), "w") as f:
            f.write(f"Model: {model.__class__}\n")
            f.write(f"Tokenizer: {tokenizer.__class__}\n")

        prompt = "Hello! Can you tell me how to learn PyTorch?"
        if apply_template:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
        else:
            inputs = tokenizer(
                prompt, return_tensors="pt", padding=True, return_attention_mask=True
            ).to(model.device)

        with torch.no_grad() and tracer:
            outputs = model.generate(
                inputs["input_ids"],
                num_return_sequences=1,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                attention_mask=inputs["attention_mask"],
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n--- Generated Response ---")
        print(response)
        print("--------------------------\n")
        print(f"✅ Test for {true_model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during inference for {true_model_name}: {e}")


def run_inference_test_i2t(model_name: str):
    print(f"🚀 Running Image2Text Inference Test for: {model_name}")
    true_model_name = "/".join(model_name.rsplit("/", 2)[-2:])
    output_path = f"tools/api_tracer/trace_output_test_infer/{true_model_name}"
    os.makedirs(output_path, exist_ok=True)
    tracer = APITracer(
        "torch", output_path=output_path, levels=[0, 1], merge_output=True
    )

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        print(f"Model Class: {model.__class__}")
        print(f"Processor Class: {processor.__class__}")

        with open(os.path.join(output_path, "model_info.txt"), "w") as f:
            f.write(f"Model: {model.__class__}\n")
            f.write(f"Processor: {processor.__class__}\n")

        question = "What is in this image?"
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            }
        ]
        if hasattr(processor, "chat_template") and processor.chat_template is not None:
            prompt = processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
        else:
            prompt = f"<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"

        image = Image.open("tools/api_tracer/baidu.jpg")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(
            model.device, torch.float16
        )

        with torch.no_grad() and tracer:
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )

        response = processor.decode(outputs[0], skip_special_tokens=True)

        print("\n--- Generated Response ---")
        print(response)
        print("--------------------------\n")
        print(f"✅ Test for {true_model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during inference for {true_model_name}: {e}")


def run_inference_test_v2t(model_name: str):
    print(f"🚀 Running Video2Text Inference Test for: {model_name}")
    true_model_name = "/".join(model_name.rsplit("/", 2)[-2:])
    output_path = f"tools/api_tracer/trace_output_test_infer/{true_model_name}"
    os.makedirs(output_path, exist_ok=True)
    tracer = APITracer(
        "torch", output_path=output_path, levels=[0, 1], merge_output=True
    )

    video_path = "tools/api_tracer/sample_video.mp4"
    try:
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        print(f"Model Class: {model.__class__}")
        print(f"Tokenizer Class: {tokenizer.__class__}")

        vr = VideoReader(video_path, ctx=cpu(0))
        frame_indices = np.linspace(0, len(vr) - 1, 8, dtype=int)
        video_frames = vr.get_batch(frame_indices).asnumpy()

        prompt = "describe the video in detail"
        messages = [
            {"role": "user", "content": f"Received a video. <vi_soi>{prompt}<vi_eoi>"}
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        with torch.no_grad() and tracer:
            outputs = model.generate(
                text=text,
                video=torch.tensor(video_frames, dtype=torch.bfloat16).to("cuda"),
                do_sample=False,
                max_new_tokens=100,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n--- Generated Response ---")
        print(response)
        print("--------------------------\n")
        print(f"✅ Test for {true_model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during inference for {true_model_name}: {e}")


def run_inference_test_t2i(model_name: str):
    print(f"🚀 Running Text2Image Inference Test for: {model_name}")
    true_model_name = "/".join(model_name.rsplit("/", 2)[-2:])
    output_path = f"tools/api_tracer/trace_output_test_infer/{true_model_name}"
    os.makedirs(output_path, exist_ok=True)
    tracer = APITracer(
        "torch", output_path=output_path, levels=[0, 1], merge_output=True
    )

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp16",
            trust_remote_code=True,
        ).to("cuda")

        print(f"Pipeline Class: {pipe.__class__}")

        prompt = "A majestic lion jumping from a big rock, high quality, cinematic"

        with torch.no_grad() and tracer:
            image = pipe(prompt=prompt, num_inference_steps=25).images[0]

        print(f"✅ Test for {true_model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during inference for {true_model_name}: {e}")


def run_inference_test_t2v(model_name: str):
    print(f"🚀 Running Text2Video Inference Test for: {model_name}")
    true_model_name = "/".join(model_name.rsplit("/", 2)[-2:])
    output_path = f"tools/api_tracer/trace_output_test_infer/{true_model_name}"
    os.makedirs(output_path, exist_ok=True)
    tracer = APITracer(
        "torch", output_path=output_path, levels=[0, 1], merge_output=True
    )

    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to("cuda")

        print(f"Pipeline Class: {pipe.__class__}")

        prompt = "A panda eating bamboo on a rock."

        with torch.no_grad() and tracer:
            video_frames = pipe(prompt, num_inference_steps=25, num_frames=20).frames

        print(f"✅ Test for {true_model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during inference for {true_model_name}: {e}")


def run_inference_test_i2d(model_name: str):
    print(f"🚀 Running Image23D Inference Test for: {model_name}")
    true_model_name = "/".join(model_name.rsplit("/", 2)[-2:])
    output_path = f"tools/api_tracer/trace_output_test_infer/{true_model_name}"
    os.makedirs(output_path, exist_ok=True)
    tracer = APITracer(
        "torch", output_path=output_path, levels=[0, 1], merge_output=True
    )

    image_path = "tools/api_tracer/baidu.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Image file not found at {image_path}.")
        return

    try:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
        ).to("cuda")

        print(f"Pipeline Class: {pipe.__class__}")

        input_image = Image.open(image_path).convert("RGB").resize((256, 256))

        with torch.no_grad() and tracer:
            images = pipe(
                input_image, num_inference_steps=64, frame_size=256, output_type="pil"
            ).images

        print(f"✅ Test for {true_model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during inference for {true_model_name}: {e}")


def run_inference_test_a2a(model_name: str):
    print(f"🚀 Running Any2Any Inference Test for: {model_name}")
    true_model_name = "/".join(model_name.rsplit("/", 2)[-2:])
    output_path = f"tools/api_tracer/trace_output_test_infer/{true_model_name}"
    os.makedirs(output_path, exist_ok=True)
    tracer = APITracer(
        "torch", output_path=output_path, levels=[0, 1], merge_output=True
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        print(f"Model Class: {model.__class__}")
        print(f"Processor Class: {processor.__class__}")

        with open(os.path.join(output_path, "model_info.txt"), "w") as f:
            f.write(f"Model: {model.__class__}\n")
            f.write(f"Processor: {processor.__class__}\n")

        prompt = "Describe the object in the image."
        image = Image.open("tools/api_tracer/baidu.jpg")

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(
            model.device, dtype=torch.bfloat16
        )

        with torch.no_grad() and tracer:
            outputs = model.generate(**inputs, max_new_tokens=100)

        response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        print("\n--- Generated Response ---")
        print(response)
        print("--------------------------\n")
        print(f"✅ Test for {true_model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during inference for {true_model_name}: {e}")


def main():
    for model_name in TextGenerationMODELS:
        if "RWKV" in model_name:
            run_inference_test_tg(model_name, apply_template=True)
        else:
            run_inference_test_tg(model_name)
    for model_name in ImageTexttoTextModels:
        run_inference_test_i2t(model_name)

    for model_name in ImageTexttoTextModels:
        run_inference_test_i2t(model_name)

    for model_name in VideoTexttoTextModels:
        run_inference_test_v2t(model_name)

    for model_name in TexttoImageModels:
        run_inference_test_t2i(model_name)

    for model_name in TexttoVideoModels:
        run_inference_test_t2v(model_name)

    for model_name in Imageto3DModels:
        run_inference_test_i2d(model_name)

    for model_name in AnytoAnyModels:
        run_inference_test_a2a(model_name)


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

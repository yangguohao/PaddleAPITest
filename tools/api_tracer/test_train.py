import os

os.environ["HF_HOME"] = "tools/api_tracer/.huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import traceback
from pathlib import Path

import numpy as np
import torch
from api_tracer import APITracer
from datasets import load_dataset
from decord import VideoReader, cpu
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from PIL import Image
from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                          AutoProcessor, AutoTokenizer)
from transformers.data.data_collator import (DataCollatorForLanguageModeling,
                                             DataCollatorForSeq2Seq)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

MODELS_DIR = Path("/root/paddlejob/workspace/env_run/models")
# MODELS_DIR = Path("/root/paddlejob/workspace/env_run/bos/huggingface")

TextGenerationMODELS = [
    # "Qwen/Qwen2-0.5B",
    # "Qwen/Qwen2-57B-A14B",
    # "Qwen/Qwen2.5-0.5B",
    # "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-30B-A3B",
    # "meta-llama/Llama-2-7b-hf",
    # "meta-llama/Llama-3.1-8B"
    # "deepseek-ai/DeepSeek-V2-Lite",  # need transformers<4.49
    # "deepseek-ai/DeepSeek-V3",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "baidu/ERNIE-4.5-0.3B-PT",
    # "baidu/ERNIE-4.5-21B-A3B-PT",
    # "moonshotai/Kimi-K2-Instruct",
    # "zai-org/GLM-4.5",
    # "mistralai/Magistral-Small-2507",
    # "MiniMaxAI/MiniMax-M1-40k",
    # "state-spaces/mamba2-2.7b",
    # "RWKV/RWKV7-Goose-World3-2.9B-HF",  #  maybe fail, change to fla-hub/rwkv7-2.9B-world, need transformers<4.50
]

ImageTexttoTextModels = [
    # "Qwen/Qwen2.5-VL-3B-Instruct",
    # "deepseek-ai/deepseek-vl2-tiny",  # need to clone deepseek_vl2 project
    # "llava-hf/llava-1.5-7b-hf",
    # "meta-llama/Llama-4-Maverick-17B-128E",
    # "baidu/ERNIE-4.5-VL-28B-A3B-PT",  # need transformers<4.54
    # "zai-org/GLM-4.1V-9B-Thinking",
    # "ByteDance/Dolphin",
    # "Salesforce/blip2-opt-2.7b",  # need transformers<4.50
    # "OpenGVLab/InternVL3-1B",
    # "moonshotai/Kimi-VL-A3B-Instruct",  # need transformers<4.50
    # "XiaomiMiMo/MiMo-VL-7B-SFT",
    # "echo840/MonkeyOCR",  # need to clone MonkeyOCR project
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
    # "Wan-AI/Wan2.1-T2V-14B-Diffusers",
]

Imageto3DModels = [
    # "tencent/HunyuanWorld-1",
]

AnytoAnyModels = [
    # "deepseek-ai/Janus-Pro-1B",
    # "ByteDance-Seed/BAGEL-7B-MoT",
]


def run_training_test_tg(model_name: str):
    print(f"🚀 Running Text Generation Training Test for: {model_name})")
    model_path = MODELS_DIR / model_name
    output_path = f"tools/api_tracer/trace_output_test_train/{model_name}"
    tracer = APITracer(
        "torch", output_path=output_path, levels=[0, 1], merge_output=True
    )
    tracer.start()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if "Llama" in model_name:
            llama_chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "{{'<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}"
                "{% elif message['role'] == 'user' %}"
                "{{'<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}"
                "{% elif message['role'] == 'assistant' %}"
                "{{'<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{'<|start_header_id|>assistant<|end_header_id|>\n\n'}}"
                "{% endif %}"
            )
            tokenizer.chat_template = llama_chat_template

        print(f"Model Class: {model.__class__}")
        print(f"Tokenizer Class: {tokenizer.__class__}")

        with open(os.path.join(output_path, "model_info.txt"), "w") as f:
            f.write(f"Model: {model.__class__}\n")
            f.write(f"Tokenizer: {tokenizer.__class__}\n")

        dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train[:500]")

        def preprocess_function(examples):
            all_texts = []
            for conv_a, conv_b in zip(
                examples["conversation_a"], examples["conversation_b"]
            ):
                text_a = tokenizer.apply_chat_template(
                    conv_a, tokenize=False, add_generation_prompt=False
                )
                all_texts.append(text_a)
                text_b = tokenizer.apply_chat_template(
                    conv_b, tokenize=False, add_generation_prompt=False
                )
                all_texts.append(text_b)
            return tokenizer(all_texts, truncation=True, max_length=512)

        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=100,
            remove_columns=next(iter(dataset)).keys(),
        )

        gradient_checkpointing = False if "RWKV" in model_name else True

        output_dir = output_path + "/train_output"
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=2e-5,
            logging_steps=20,
            save_strategy="no",
            bf16=True,
            report_to="none",
            max_steps=1,
            gradient_checkpointing=gradient_checkpointing,  # disable this when Unexpected keyword arguments: num_items_in_batch
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        trainer.train()

        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        print(f"✅ Test for {model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during training for {model_name}: {e}")
    finally:
        tracer.stop()


class DolphinTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch):
        inputs.pop("input_ids", None)
        inputs.pop("attention_mask", None)
        return super().training_step(model, inputs, num_items_in_batch)


def run_training_test_i2t(model_name: str):
    print(f"🚀 Running Image2Text Training Test for: {model_name})")
    model_path = MODELS_DIR / model_name
    output_path = f"tools/api_tracer/trace_output_test_train/{model_name}"
    tracer = APITracer(
        "torch", output_path=output_path, levels=[0, 1], merge_output=True
    )
    tracer.start()

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            # use_fast=False,  # this will cause error in GLM-4.1V
        )
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        model.config.pad_token_id = processor.tokenizer.pad_token_id

        if "Dolphin" in model_name:
            model.config.decoder.pad_token_id = model.config.pad_token_id
            if processor.tokenizer.bos_token_id is not None:
                model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
            else:
                model.config.decoder_start_token_id = processor.tokenizer.eos_token_id

        print(f"Model Class: {model.__class__}")
        print(f"Processor Class: {processor.__class__}")

        with open(os.path.join(output_path, "model_info.txt"), "w") as f:
            f.write(f"Model: {model.__class__}\n")
            f.write(f"Processor: {processor.__class__}\n")

        dataset = load_dataset("HongchengGao/TuringEyeTest", split="full")
        dataset_sample = dataset.select(range(100))

        def preprocess_function(examples):
            model_inputs = {
                "input_ids": [],
                "labels": [],
                "pixel_values": [],
                "attention_mask": [],
                "image_grid_thw": [],
            }

            for i in range(len(examples["Question"])):
                question = examples["Question"][i]
                groundtruth = examples["Groundtruth"][i]
                image = examples["Image"][i].convert("RGB")

                user_prompt_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image"},
                        ],
                    },
                ]
                full_conversation_messages = user_prompt_messages + [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": groundtruth}],
                    },
                ]

                if processor.tokenizer.chat_template is not None:
                    prompt_only_text = processor.tokenizer.apply_chat_template(
                        user_prompt_messages, tokenize=False, add_generation_prompt=True
                    )
                    prompt_ids_len = len(
                        processor.tokenizer(prompt_only_text).input_ids
                    )

                    inputs = processor(
                        text=full_conversation_messages,
                        images=image,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                        max_length=1024,
                        add_generation_prompt=False,
                    )
                else:
                    manual_chat_template = (
                        "{% for message in messages %}"
                        "{% if message.role == 'user' %}"
                        "{{ message.content[0].text }} <image>"
                        "{% elif message.role == 'assistant' %}"
                        "{{ message.content[0].text }}"
                        "{% endif %}"
                        "{% endfor %}"
                        "{% if add_generation_prompt %} {% endif %}"
                    )

                    prompt_for_len_calc = processor.tokenizer.apply_chat_template(
                        user_prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        chat_template=manual_chat_template,
                    )
                    prompt_ids_len = len(
                        processor.tokenizer(prompt_for_len_calc).input_ids
                    )

                    full_prompt_text = processor.tokenizer.apply_chat_template(
                        full_conversation_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        chat_template=manual_chat_template,
                    )

                    inputs = processor(
                        text=[full_prompt_text],
                        images=image,
                        return_tensors="pt",
                        padding=False,
                        truncation=True,
                        max_length=1024,
                    )

                input_ids = inputs["input_ids"][0]
                labels = input_ids.clone()
                labels[:prompt_ids_len] = -100

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)
                model_inputs["pixel_values"].append(inputs["pixel_values"][0])

                if "attention_mask" in inputs:
                    model_inputs["attention_mask"].append(inputs["attention_mask"][0])

                if "image_grid_thw" in inputs:
                    model_inputs["image_grid_thw"].append(inputs["image_grid_thw"][0])

            if not model_inputs["attention_mask"]:
                del model_inputs["attention_mask"]

            if not model_inputs["image_grid_thw"]:
                del model_inputs["image_grid_thw"]

            return model_inputs

        tokenized_dataset = dataset_sample.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset_sample.column_names,
            batch_size=4,
        )

        output_dir = output_path + "/train_output"
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            save_strategy="no",
            bf16=True,
            report_to="none",
            max_steps=1,
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=processor.tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

        TrainerClass = DolphinTrainer if "Dolphin" in model_name else Trainer

        trainer = TrainerClass(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        trainer.train()

        trainer.save_model()
        processor.save_pretrained(output_dir)

        print(f"✅ Test for {model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during training for {model_name}: {e}")
    finally:
        tracer.stop()


def sample_frames_from_video(video_path, num_frames=8):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(frame) for frame in frames]


def run_training_test_v2t(model_name: str):
    print(f"🚀 Running Video-Text-to-Text Training Test for: {model_name})")
    model_path = MODELS_DIR / model_name
    output_path = f"tools/api_tracer/trace_output_test_train/{model_name}"
    tracer = APITracer(
        "torch", output_path=output_path, levels=[0, 1], merge_output=True
    )
    tracer.start()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model.config.pad_token_id = processor.tokenizer.pad_token_id

        print(f"Model Class: {model.__class__}")
        print(f"Processor Class: {processor.__class__}")

        with open(os.path.join(output_path, "model_info.txt"), "w") as f:
            f.write(f"Model: {model.__class__}\n")
            f.write(f"Processor: {processor.__class__}\n")

        dataset = load_dataset("microsoft/msr_vtt", split="train")
        dataset_sample = dataset.select(range(50))

        def preprocess_function(examples):
            num_frames_for_model = 8
            model_inputs = {"input_ids": [], "labels": [], "pixel_values": []}

            for i in range(len(examples["text"])):
                video_path = examples["video"][i]["path"]
                caption = examples["text"][i]
                frames = sample_frames_from_video(
                    video_path, num_frames=num_frames_for_model
                )
                if frames is None:
                    continue

                prompt = f"User: Describe the following video.\nAssistant: {caption}"

                inputs = processor(
                    text=prompt,
                    images=frames,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=1024,
                )
                input_ids = inputs["input_ids"][0]
                labels = input_ids.clone()

                prompt_without_answer = (
                    "User: Describe the following video.\nAssistant:"
                )
                prompt_only_inputs = processor(
                    text=prompt_without_answer, images=frames, return_tensors="pt"
                )
                prompt_ids_len = len(prompt_only_inputs["input_ids"][0])
                labels[:prompt_ids_len] = -100

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)
                model_inputs["pixel_values"].append(inputs["pixel_values"][0])

            return model_inputs

        tokenized_dataset = dataset_sample.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset_sample.column_names,
            batch_size=2,
        )

        output_dir = output_path + "/train_output"
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            save_strategy="no",
            bf16=True,
            report_to="none",
            max_steps=1,
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=processor.tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        trainer.train()

        trainer.save_model()
        processor.save_pretrained(output_dir)

        print(f"✅ Test for {model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during training for {model_name}: {e}")
    finally:
        tracer.stop()


def run_training_test_t2i(model_name: str):
    print(f"🚀 Running Text-to-Image Training Test for: {model_name})")
    model_path = MODELS_DIR / model_name
    output_path = f"tools/api_tracer/trace_output_test_train/{model_name}"
    tracer = APITracer(
        "torch", output_path=output_path, levels=[0, 1], merge_output=True
    )
    tracer.start()

    try:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline.to(device)

        unet = pipeline.unet
        text_encoder = pipeline.text_encoder
        vae = pipeline.vae
        tokenizer = pipeline.tokenizer
        image_processor = pipeline.image_processor
        noise_scheduler = pipeline.scheduler

        print(f"Unet Class: {unet.__class__}")
        print(f"Text Encoder Class: {text_encoder.__class__}")

        with open(os.path.join(output_path, "model_info.txt"), "w") as f:
            f.write(f"Pipeline: {pipeline.__class__}\n")
            f.write(f"Unet: {unet.__class__}\n")
            f.write(f"Text Encoder: {text_encoder.__class__}\n")

        dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train[:100]")

        optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)

        max_train_steps = 1
        gradient_accumulation_steps = 4
        global_step = 0

        unet.train()
        vae.eval()
        text_encoder.eval()

        for step, batch in enumerate(dataset):
            if global_step >= max_train_steps:
                break

            text_input = tokenizer(
                batch["text"],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            pixel_values = image_processor.preprocess(
                batch["image"].convert("RGB"), return_tensors="pt"
            )["pixel_values"]
            pixel_values = pixel_values.to(device, dtype=torch.bfloat16)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            with torch.no_grad():
                prompt_embeds = pipeline.encode_prompt(
                    prompt=batch["text"],
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (1,), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            model_pred = unet(noisy_latents, timesteps, prompt_embeds).sample

            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                print(
                    f"Step: {global_step}, Loss: {loss.item() * gradient_accumulation_steps}"
                )

        pipeline.save_pretrained(output_dir)

        print(f"✅ Test for {model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during training for {model_name}: {e}")
    finally:
        tracer.stop()


def run_training_test_t2v(model_name: str):
    print(f"🚀 Running Text-to-Video Training Test for: {model_name})")
    model_path = MODELS_DIR / model_name
    output_path = f"tools/api_tracer/trace_output_test_train/{model_name}"
    tracer = APITracer(
        "torch", output_path=output_path, levels=[0, 1], merge_output=True
    )
    tracer.start()

    try:
        pipeline = AnimateDiffPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline.to(device)

        unet = pipeline.unet
        text_encoder = pipeline.text_encoder
        vae = pipeline.vae
        tokenizer = pipeline.tokenizer
        noise_scheduler = pipeline.scheduler

        print(f"Unet Class: {unet.__class__}")
        print(f"Text Encoder Class: {text_encoder.__class__}")
        with open(os.path.join(output_path, "model_info.txt"), "w") as f:
            f.write(f"Pipeline: {pipeline.__class__}\n")
            f.write(f"Unet: {unet.__class__}\n")

        dataset = load_dataset("microsoft/msr_vtt", split="train[:50]")

        optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)
        max_train_steps = 1
        global_step = 0

        unet.train()
        vae.eval()
        text_encoder.eval()

        for batch in dataset:
            if global_step >= max_train_steps:
                break

            frames = sample_frames_from_video(batch["video"]["path"], num_frames=16)
            if frames is None:
                continue

            # Preprocess inputs
            prompt = batch["text"]
            prompt_ids = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids.to(device)

            video_tensor = torch.stack(
                [
                    torch.from_numpy(np.array(frame)).permute(2, 0, 1) / 127.5 - 1.0
                    for frame in frames
                ]
            ).to(device, dtype=torch.bfloat16)

            with torch.no_grad():
                prompt_embeds = text_encoder(prompt_ids)[0]
                video_latents = (
                    vae.encode(video_tensor).latent_dist.sample()
                    * vae.config.scaling_factor
                )

            noise = torch.randn_like(video_latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (1,), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(video_latents, noise, timesteps)

            model_pred = unet(
                noisy_latents, timesteps, encoder_hidden_states=prompt_embeds
            ).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            print(f"Step: {global_step}, Loss: {loss.item()}")

        pipeline.save_pretrained(output_dir)

        print(f"✅ Test for {model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during training for {model_name}: {e}")
    finally:
        tracer.stop()


def run_training_test_i2d3(model_name: str):
    print(f"🔶 Skipping Image-to-3D Training Test for: {model_name}")
    print(
        "Reason: Image-to-3D models like TripoSR/HunyuanWorld-1 are typically released as inference-only pipelines "
        "and do not have a straightforward training/fine-tuning setup without access to the original training scripts."
    )


def run_training_test_a2a(model_name: str):
    print(f"🚀 Running Any-to-Any Training Test for: {model_name})")
    model_path = MODELS_DIR / model_name
    output_path = f"tools/api_tracer/trace_output_test_train/{model_name}"
    tracer = APITracer(
        "torch", output_path=output_path, levels=[0, 1], merge_output=True
    )
    tracer.start()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model.config.pad_token_id = processor.tokenizer.pad_token_id

        print(f"Model Class: {model.__class__}")
        print(f"Processor Class: {processor.__class__}")

        with open(os.path.join(output_path, "model_info.txt"), "w") as f:
            f.write(f"Model: {model.__class__}\n")
            f.write(f"Processor: {processor.__class__}\n")

        dataset = load_dataset("HongchengGao/TuringEyeTest", split="full")
        dataset_sample = dataset.select(range(100))

        def preprocess_function(examples):
            model_inputs = {"input_ids": [], "labels": [], "pixel_values": []}

            for i in range(len(examples["Question"])):
                question = examples["Question"][i]
                groundtruth = examples["Groundtruth"][i]
                image = examples["Image"][i].convert("RGB")

                prompt = [
                    {"type": "image", "content": image},
                    {"type": "text", "content": f"\nQuestion: {question}\nAnswer: "},
                ]
                full_conversation = prompt + [{"type": "text", "content": groundtruth}]

                inputs = processor(
                    full_conversation,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                )

                prompt_len_inputs = processor(prompt, return_tensors="pt")
                prompt_ids_len = len(prompt_len_inputs["input_ids"][0])

                input_ids = inputs["input_ids"][0]
                labels = input_ids.clone()
                labels[:prompt_ids_len] = -100

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)
                model_inputs["pixel_values"].append(inputs["pixel_values"][0])

            return model_inputs

        tokenized_dataset = dataset_sample.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset_sample.column_names,
            batch_size=4,
        )

        output_dir = output_path + "/train_output"
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            save_strategy="no",
            bf16=True,
            report_to="none",
            max_steps=1,
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=processor.tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        trainer.train()

        trainer.save_model()
        processor.save_pretrained(output_dir)

        print(f"✅ Test for {model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during training for {model_name}: {e}")
    finally:
        tracer.stop()


def main():
    for model_name in TextGenerationMODELS:
        run_training_test_tg(model_name)

    for model_name in ImageTexttoTextModels:
        run_training_test_i2t(model_name)

    for model_name in VideoTexttoTextModels:
        run_training_test_v2t(model_name)

    for model_name in TexttoImageModels:
        run_training_test_t2i(model_name)

    for model_name in TexttoVideoModels:
        run_training_test_t2v(model_name)

    for model_name in Imageto3DModels:
        run_training_test_i2d3(model_name)

    for model_name in AnytoAnyModels:
        run_training_test_a2a(model_name)


if __name__ == "__main__":
    main()

import os
import traceback

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "tools/api_tracer/.huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.data.data_collator import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from tools.api_tracer import APITracer

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
    # "RWKV/RWKV7-Goose-World3-2.9B-HF",
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
]

VideoTexttoTextModels = [
    # "Kwai-Keye/Keye-VL-8B-Preview",
]

TexttoImageModels = [
    # "stabilityai/stable-diffusion-3-medium",
    # "black-forest-labs/FLUX.1-dev",
    # "echo840/MonkeyOCR",
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


def run_training_test_tg(model_name: str):
    print(f"🚀 Running Text Generation Training Test for: {model_name})")
    true_model_name = "/".join(model_name.rsplit("/", 2)[-2:])
    output_path = f"tools/api_tracer/trace_output_test_train/{true_model_name}"
    tracer = APITracer(
        "torch", output_path=output_path, levels=[0, 1], merge_output=True
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if "Llama" in true_model_name:
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
            gradient_checkpointing=True,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        with tracer:
            trainer.train()

        print(f"✅ Test for {true_model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during training for {true_model_name}: {e}")


class DolphinTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch):
        inputs.pop("input_ids", None)
        inputs.pop("attention_mask", None)
        return super().training_step(model, inputs, num_items_in_batch)


def run_training_test_i2t(model_name: str):
    print(f"🚀 Running Image2Text Training Test for: {model_name})")
    true_model_name = "/".join(model_name.rsplit("/", 2)[-2:])
    output_path = f"tools/api_tracer/trace_output_test_train/{true_model_name}"
    tracer = APITracer(
        "torch", output_path=output_path, levels=[0, 1], merge_output=True
    )

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            # use_fast=False,  # this will cause error in GLM-4.1V
        )
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        model.config.pad_token_id = processor.tokenizer.pad_token_id

        if "Dolphin" in true_model_name:
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

        TrainerClass = DolphinTrainer if "Dolphin" in true_model_name else Trainer

        trainer = TrainerClass(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # with tracer:
        trainer.train()

        print(f"✅ Test for {true_model_name} finished.")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ An error occurred during training for {true_model_name}: {e}")


def main():
    for model_name in TextGenerationMODELS:
        run_training_test_tg(model_name)

    for model_name in ImageTexttoTextModels:
        run_training_test_i2t(model_name)


if __name__ == "__main__":
    main()

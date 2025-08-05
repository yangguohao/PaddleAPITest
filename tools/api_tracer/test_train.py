import os
import traceback

os.environ["HF_HOME"] = "tools/api_tracer/.huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                          AutoProcessor, AutoTokenizer)
from transformers.data.data_collator import (DataCollatorForLanguageModeling,
                                             DataCollatorForSeq2Seq)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from tools.api_tracer import APITracer

MODELS = [
    # "Qwen/Qwen2-0.5B",
    # "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-30B-A3B",
    # "Qwen/Qwen2.5-VL-3B-Instruct",
    # "deepseek-ai/DeepSeek-V2-Lite",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "baidu/ERNIE-4.5-0.3B-PT",
]


def run_training_test(model_name: str):
    print(f"🚀 Running training test for: {model_name})")
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
        tokenizer = AutoTokenizer.from_pretrained(model_name)
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

        training_args = TrainingArguments(
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


def run_training_test_vision(model_name: str):
    print(f"🚀 Running training test for: {model_name})")
    output_path = f"tools/api_tracer/trace_output_test_train/{model_name}"
    tracer = APITracer("torch", output_path=output_path)

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
            use_fast=False,
        )

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
                "attention_mask": [],
                "pixel_values": [],
                "labels": [],
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
                    }
                ]
                full_conversation_messages = user_prompt_messages + [
                    {"role": "assistant", "content": groundtruth}
                ]

                prompt_text = processor.tokenizer.apply_chat_template(
                    user_prompt_messages, tokenize=False, add_generation_prompt=True
                )
                prompt_ids_len = len(processor.tokenizer(prompt_text).input_ids)

                inputs = processor(
                    text=full_conversation_messages,
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
                model_inputs["attention_mask"].append(inputs["attention_mask"][0])
                model_inputs["pixel_values"].append(inputs["pixel_values"][0])
                model_inputs["labels"].append(labels)

            model_inputs["pixel_values"] = torch.cat(
                model_inputs["pixel_values"], dim=0
            )
            return model_inputs

        tokenized_dataset = dataset_sample.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset_sample.column_names,
            batch_size=4,
        )

        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            logging_steps=5,
            save_strategy="no",
            bf16=True,
            report_to="none",
            max_steps=20,
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=processor.tokenizer, model=model
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        with tracer:
            trainer.train()

        print(f"✅ Test for {model_name} finished.")
    except Exception as e:
        print(f"❌ An error occurred during training for {model_name}: {e}")


def main():
    for model_name in MODELS:
        if "VL" in model_name:
            run_training_test_vision(model_name)
        else:
            run_training_test(model_name)


if __name__ == "__main__":
    main()

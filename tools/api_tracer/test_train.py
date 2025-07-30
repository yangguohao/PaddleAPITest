import os

os.environ["HF_HOME"] = "tools/api_tracer/.huggingface"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from tools.api_tracer import APITracer

MODELS = [
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-30B-A3B",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "baidu/ERNIE-4.5-0.3B-PT",
]


def run_training_test(model_name: str):
    print(f"🚀 Running training test for: {model_name})")
    output_path = f"tools/api_tracer/trace_output_test_train/{model_name}"
    tracer = APITracer("torch", output_path=output_path)

    try:
        tracer.start()

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Model Class: {model.__class__}")
        print(f"Tokenizer Class: {tokenizer.__class__}")

        with open(os.path.join(output_path, "model_info.txt"), "w") as f:
            f.write(f"Model: {model.__class__}\n")
            f.write(f"Tokenizer: {tokenizer.__class__}\n")

        dataset = load_dataset(
            "lmsys/chatbot_arena_conversations", split="train", streaming=True
        )
        dataset_sample = dataset.take(1000)

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

        tokenized_dataset = dataset_sample.map(
            preprocess_function,
            batched=True,
            remove_columns=next(iter(dataset_sample)).keys(),
        )

        save_model_path = f"{output_path}/finetuned-arena"
        training_args = TrainingArguments(
            output_dir=save_model_path,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            logging_steps=20,
            save_steps=50,
            bf16=True,
            report_to="none",
            max_steps=100,
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        trainer.train()

        final_model_path = f"{output_path}/finetuned-final"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
    except Exception as e:
        print(f"An error occurred during training for {model_name}: {e}")
    finally:
        tracer.stop()
        print(f"✅ Test for {model_name} finished.")


def main():
    for model_name in MODELS:
        run_training_test(model_name)


if __name__ == "__main__":
    main()

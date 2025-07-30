import os

os.environ["HF_HOME"] = "tools/api_tracer/.huggingface"

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from tools.api_tracer import APITracer

tracer = APITracer("torch", output_path="tools/api_tracer/test_train_trace_output")
tracer.start()

# model_name = "Qwen/Qwen3-0.6B"
# model_name = "Qwen/Qwen3-30B-A3B"
model_name = "baidu/ERNIE-4.5-0.3B-PT"

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
print("model:", model.__class__)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("tokenizer:", tokenizer.__class__)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(
    "lmsys/chatbot_arena_conversations", split="train", streaming=True
)

dataset_sample = dataset.take(10000)


def preprocess_function(examples):
    all_texts = []
    for conv_a, conv_b in zip(examples["conversation_a"], examples["conversation_b"]):
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

training_args = TrainingArguments(
    output_dir="tools/api_tracer/qwen3-0.6b-finetuned-arena",
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

print("🚀 开始训练...")
trainer.train()
print("✅ 训练完成！")

final_model_path = "tools/api_tracer/qwen3-0.6b-finetuned-final"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"模型已保存至: {final_model_path}")

tracer.stop()

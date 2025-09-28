import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from datasets import Dataset

from peft import get_peft_model, LoraConfig, TaskType
import os
import pickle

# === Load model & tokenizer ===
model_path = "/NBL/llama_model_ds2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

# === Load linear_replacement weights ===
for i, layer in enumerate(model.model.layers):
    if getattr(layer, "linear_replacement", None):
        with open(f"/NBL/llm_variables_deepseek/xlayer_{i}_weights.pkl", "rb") as f:
            w, b = pickle.load(f)
        layer.initialize_linear_from_pickle(w.to(torch.bfloat16), b.to(torch.bfloat16))

# === Find target LoRA modules ===
def find_linear_replacement_layers(model):
    return [
        name for name, module in model.named_modules()
        if "linear_replacement" in name and isinstance(module, torch.nn.Linear)
    ]

target_modules = find_linear_replacement_layers(model)
print(f"LoRA will be applied to {len(target_modules)} linear_replacement layers:")
print(target_modules)

if not target_modules:
    raise ValueError("No linear_replacement layers were found.")

# === Configure LoRA ===
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === Load validation set in streaming mode ===
"""
streamed_dataset = load_dataset("c4", "en", split="validation", streaming=True)
streamed_dataset = iter(streamed_dataset)
"""
streamed_dataset = load_dataset("cerebras/SlimPajama-627B", split="validation", streaming=True)
streamed_dataset = iter(streamed_dataset)

MAX_SAMPLES = 5000  # Reduce or increase based on memory
raw_examples = [next(streamed_dataset) for _ in range(MAX_SAMPLES)]

# Convert to HuggingFace Dataset object
dataset = Dataset.from_list(raw_examples)


def preprocess(example):
    # Truncate to max length and shift tokens for causal LM
    inputs = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=1024
    )
    inputs["labels"] = inputs["input_ids"].copy()  # CLM target = same as input
    return inputs

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# === Data collator ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Training arguments ===
training_args = TrainingArguments(
    output_dir="./lora-finetuned-slim",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,  # Change to 3+ for better results
    logging_dir="./logs",
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    fp16=False,
    save_total_limit=2,
    report_to="none"
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Train ===
trainer.train()

# === Save LoRA adapter ===
model.save_pretrained("./lora-linear-replacement-slim")

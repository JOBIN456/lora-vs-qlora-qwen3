import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# --------------------------------------------------
# 1. Model
# --------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,     # was torch_dtype
    device_map="auto",
)

# --------------------------------------------------
# 2. Dataset
# --------------------------------------------------

dataset = load_dataset("tatsu-lab/alpaca", split="train")
dataset = dataset.select(range(1000))


# --------------------------------------------------
# 3. Convert Alpaca → Chat format
# --------------------------------------------------

def format_example(example):
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]

    if input_text:
        user_message = f"{instruction}\n\nInput:\n{input_text}"
    else:
        user_message = instruction

    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": output},
    ]

    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    }


dataset = dataset.map(format_example, remove_columns=dataset.column_names)


# --------------------------------------------------
# 4. LoRA configuration
# --------------------------------------------------

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


# --------------------------------------------------
# 5. Training configuration
# --------------------------------------------------

training_args = SFTConfig(
    output_dir="./qwen3-lora",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    report_to="none",
    dataset_text_field="text",
    max_length=512,          # was max_seq_length
)
# --------------------------------------------------
# 6. Trainer
# --------------------------------------------------

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    processing_class=tokenizer,
)
# --------------------------------------------------
# 7. Train
# --------------------------------------------------

trainer.train()
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
print(torch.cuda.memory_reserved() / 1e9, "GB reserved")

# --------------------------------------------------
# 8. Save LoRA adapter
# --------------------------------------------------

trainer.save_model("./qwen3-lora")

print("LoRA training completed!")
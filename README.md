# LoRA vs QLoRA — Qwen3-1.7B

Side-by-side **LoRA vs QLoRA** fine-tuning of **Qwen3-1.7B** on Alpaca.
Two self-contained scripts using 🤗 Transformers, PEFT & TRL — compare bf16 LoRA
against 4-bit NF4 QLoRA with identical hyperparameters, LoRA config, and dataset.

---

## Why this repo?

Parameter-efficient fine-tuning (PEFT) has two popular flavors:

- **LoRA** — trains small adapter matrices on top of a full-precision (bf16) base model.
- **QLoRA** — first quantizes the base model to 4-bit NF4, *then* trains LoRA adapters on top.

QLoRA trades a little accuracy for a large memory saving, letting you fine-tune bigger
models on smaller GPUs. This repo lets you **see that trade-off directly** — both scripts
share the same model, data, LoRA config, and trainer setup. The **only difference is
quantization**.

---

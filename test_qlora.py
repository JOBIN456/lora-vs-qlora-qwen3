import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-1.7B"
ADAPTER_PATH = "./qwen3-qlora"

# --------------------------------------------------
# Load base model in 4-bit (same config as training)
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# --------------------------------------------------
# Load QLoRA adapter on top
# --------------------------------------------------
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# --------------------------------------------------
# Test prompts — same as LoRA test, for fair comparison
# --------------------------------------------------
test_prompts = [
    "Give three tips for staying focused while studying.",
    "Explain photosynthesis in simple terms.",
    "Write a short poem about the ocean.",
    "What is the capital of France?",
]

def generate_response(prompt, max_new_tokens=200):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    response = full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    return response.strip()


if __name__ == "__main__":
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"[{i}] PROMPT: {prompt}")
        print(f"{'='*60}")
        response = generate_response(prompt)
        print(f"RESPONSE:\n{response}")
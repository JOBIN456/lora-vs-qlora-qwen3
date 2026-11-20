import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-1.7B"
ADAPTER_PATH = "./qwen3-lora"   # or "./qwen3-lora/checkpoint-125"

# --------------------------------------------------
# Load base model + tokenizer
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)

# --------------------------------------------------
# Load LoRA adapter on top
# --------------------------------------------------
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# --------------------------------------------------
# Test prompts
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
    # strip the prompt portion, keep only the generated reply
    response = full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    return response.strip()


if __name__ == "__main__":
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"[{i}] PROMPT: {prompt}")
        print(f"{'='*60}")
        response = generate_response(prompt)
        print(f"RESPONSE:\n{response}")
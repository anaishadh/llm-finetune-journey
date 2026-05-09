# ============================================================
# Stage 4: Chat with your fine-tuned NovaBrew bot
# ============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FINETUNED_MODEL = "outputs/novabrew-model"

# ============================================================
# STEP 1: LOAD BASE MODEL + YOUR LORA ADAPTERS ON TOP
# ============================================================

print("Loading your fine-tuned NovaBrew bot...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Load YOUR LoRA adapters on top of base model
model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
model.eval()  # inference mode, no training

print("NovaBrew bot ready!\n")

# ============================================================
# STEP 2: CHAT FUNCTION
# ============================================================

def chat(user_message):
    system_prompt = (
        "You are a friendly and helpful customer support agent for NovaBrew, "
        "a smart coffee machine company. Answer questions accurately and politely "
        "based on NovaBrew product knowledge. If you don't know the answer, "
        "direct the customer to support@novabrew.com."
    )

    # Format exactly like training data
    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (not the prompt)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()

# ============================================================
# STEP 3: INTERACTIVE CHAT LOOP
# ============================================================

print("=" * 50)
print("NovaBrew Customer Support Bot")
print("Type 'quit' to exit")
print("=" * 50)

# First test with something FROM training data
print("\nTest 1 - Question the model saw during training:")
response = chat("How do I reset NovaBrew to factory settings?")
print(f"Bot: {response}")

print("\nTest 2 - Question the model NEVER saw:")
response = chat("My NovaBrew is making a strange noise.")
print(f"Bot: {response}")

print("\nTest 3 - Completely unrelated question:")
response = chat("What is the capital of France?")
print(f"Bot: {response}")

# Interactive mode
print("\n" + "=" * 50)
print("Now ask it anything!")
print("=" * 50)

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() == 'quit':
        break
    if user_input:
        response = chat(user_input)
        print(f"Bot: {response}")
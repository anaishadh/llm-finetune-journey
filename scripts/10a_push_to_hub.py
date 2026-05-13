# ============================================================
# Stage 10a: Push fine-tuned model to HuggingFace Hub
#
# This is also our "model registry" step (from 9d):
# - Model gets a permanent version on HuggingFace
# - Anyone can download and reproduce your exact model
# - Links back to your training config
# - This IS the industry model registry for open source
# ============================================================

from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# STEP 1: LOGIN TO HUGGINGFACE
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "anaishadh"
REPO_NAME = "novabrew-support-bot"
FULL_REPO = f"{HF_USERNAME}/{REPO_NAME}"

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FINETUNED_MODEL = "outputs/novabrew-r16"

print("=" * 50)
print("Pushing NovaBrew model to HuggingFace Hub")
print(f"Destination: {FULL_REPO}")
print("=" * 50)

login(token=HF_TOKEN)
api = HfApi()

# ============================================================
# STEP 2: CREATE THE REPO
# ============================================================

print("\nCreating repository on HuggingFace Hub...")
try:
    api.create_repo(
        repo_id=FULL_REPO,
        private=False,      # public so recruiters can see it
        exist_ok=True,      # don't fail if already exists
    )
    print(f"Repository ready: https://huggingface.co/{FULL_REPO}")
except Exception as e:
    print(f"Repo creation note: {e}")

# ============================================================
# STEP 3: PUSH LORA ADAPTERS
#
# We push ONLY the LoRA adapters, not the full model.
# The base model (Qwen2.5) already lives on HuggingFace.
# Our adapters are tiny (~100MB vs ~3GB for full model).
#
# This is standard practice — push adapters, reference base.
# ============================================================

print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)

print("\nPushing tokenizer...")
tokenizer.push_to_hub(FULL_REPO, token=HF_TOKEN)

print("Pushing LoRA adapters...")
model.push_to_hub(FULL_REPO, token=HF_TOKEN)

# ============================================================
# STEP 4: ADD MODEL CARD
#
# A model card is like a README for your model.
# HuggingFace requires it for good practice.
# This is what shows up on your model page.
# ============================================================

print("\nCreating model card...")

model_card = """---
language: en
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- peft
- lora
- customer-support
- novabrew
- qlora
---

# NovaBrew Customer Support Bot

Fine-tuned version of [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 
for NovaBrew customer support using QLoRA.

## Model Details

- **Base model**: Qwen/Qwen2.5-1.5B-Instruct
- **Fine-tuning method**: QLoRA (4-bit quantization + LoRA)
- **LoRA rank**: 16
- **Training examples**: 10 NovaBrew support Q&A pairs
- **Task**: Customer support for NovaBrew smart coffee machines

## Training

Trained using PEFT + TRL on a custom NovaBrew dataset.
Part of the [llm-finetune-journey](https://github.com/anaishadh/llm-finetune-journey) project.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "anaishadh/novabrew-support-bot")
tokenizer = AutoTokenizer.from_pretrained("anaishadh/novabrew-support-bot")
```

## Example

**User**: How do I reset my NovaBrew?

**Bot**: To factory reset your NovaBrew, hold the power button and 
the brew button simultaneously for 10 seconds until the LED flashes 
red three times.
"""

api.upload_file(
    path_or_fileobj=model_card.encode(),
    path_in_repo="README.md",
    repo_id=FULL_REPO,
    token=HF_TOKEN,
)

print("\n" + "=" * 50)
print("Model pushed successfully!")
print(f"View at: https://huggingface.co/{FULL_REPO}")
print("=" * 50) 

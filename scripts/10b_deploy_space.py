# ============================================================
# Stage 10b: Deploy NovaBrew Bot to HuggingFace Spaces
#
# HuggingFace Spaces hosts Gradio apps for free.
# We create a Space repo with:
# - app.py (the Gradio chat interface)
# - requirements.txt (dependencies)
# - README.md (Space configuration)
#
# HuggingFace builds and hosts it automatically.
# ============================================================

from huggingface_hub import HfApi, login
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "anaishadh"
SPACE_NAME = "novabrew-support-bot"
FULL_SPACE = f"{HF_USERNAME}/{SPACE_NAME}"

print("=" * 50)
print("Deploying NovaBrew Bot to HuggingFace Spaces")
print(f"Destination: {FULL_SPACE}")
print("=" * 50)

login(token=HF_TOKEN)
api = HfApi()

# ============================================================
# STEP 1: CREATE THE SPACE
# ============================================================

print("\nCreating Space...")
api.create_repo(
    repo_id=FULL_SPACE,
    repo_type="space",
    space_sdk="gradio",
    private=False,
    exist_ok=True,
)
print(f"Space created: https://huggingface.co/spaces/{FULL_SPACE}")

# ============================================================
# STEP 2: CREATE app.py
#
# This is the actual chat application.
# Gradio gives us a chat UI with zero frontend code.
# ============================================================

app_code = '''
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --------------------------------------------------------
# Page config
# --------------------------------------------------------
st.set_page_config(
    page_title="NovaBrew Support Bot",
    page_icon="☕",
    layout="centered"
)

st.title("☕ NovaBrew Customer Support")
st.caption("AI-powered support for NovaBrew smart coffee machines")

# --------------------------------------------------------
# Load model (cached so it only loads once)
# --------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FINETUNED_MODEL = "anaishadh/novabrew-support-bot"

SYSTEM_PROMPT = """You are a friendly and helpful customer support agent for NovaBrew,
a smart coffee machine company. Answer questions accurately and politely
based on NovaBrew product knowledge. If you don't know the answer,
direct the customer to support@novabrew.com."""

@st.cache_resource
def load_model():
    st.info("Loading NovaBrew model... this takes a minute on first load.")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,  # CPU needs float32
        device_map="cpu",
    )

    model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# --------------------------------------------------------
# Chat function
# --------------------------------------------------------
def generate_response(question: str) -> str:
    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Clean up role continuations
    for stop in ["Customer:", "Human:", "User:", "<|im_start|>"]:
        if stop in response:
            response = response.split(stop)[0].strip()

    return response

# --------------------------------------------------------
# Chat UI
# --------------------------------------------------------

# Example questions
st.subheader("Try asking:")
examples = [
    "How do I reset my NovaBrew?",
    "What pods are compatible with NovaBrew?",
    "My NovaBrew won't connect to WiFi.",
    "How do I clean my machine?",
    "What does the orange LED mean?",
]

cols = st.columns(2)
for i, example in enumerate(examples):
    if cols[i % 2].button(example, use_container_width=True):
        st.session_state.question = example

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
question = st.chat_input("Ask NovaBrew support...")

# Handle example button clicks
if "question" in st.session_state:
    question = st.session_state.question
    del st.session_state.question

if question:
    # Show user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Generate and show response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(question)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
'''

# ============================================================
# STEP 3: CREATE requirements.txt FOR THE SPACE
# ============================================================

requirements = """torch
transformers
peft
accelerate
bitsandbytes
streamlit
huggingface_hub
sentencepiece
"""
print("Uploading .python-version...")
api.upload_file(
    path_or_fileobj="3.10".encode(),
    path_in_repo=".python-version",
    repo_id=FULL_SPACE,
    repo_type="space",
    token=HF_TOKEN,
)

# ============================================================
# STEP 4: CREATE README.md (Space config)
#
# The YAML frontmatter configures the Space.
# This tells HuggingFace what hardware to use etc.
# ============================================================

readme = """---
title: NovaBrew Support Bot
emoji: ☕
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# NovaBrew Customer Support Bot ☕

An AI-powered customer support bot for NovaBrew smart coffee machines.

## About

This Space demonstrates a fine-tuned LLM for customer support.
The model is **Qwen2.5-1.5B-Instruct** fine-tuned with **QLoRA** on
a custom NovaBrew support dataset.

## Try it out

Ask questions like:
- "How do I reset my NovaBrew?"
- "What pods are compatible?"
- "My machine won't turn on"
- "How long is the warranty?"

## Built with

- QLoRA fine-tuning (PEFT + TRL)
- HuggingFace Transformers
- Gradio

Part of the [LLM Fine-Tuning Journey](https://github.com/anaishadh/llm-finetune-journey) project.
"""

# ============================================================
# STEP 5: UPLOAD ALL FILES TO SPACE
# ============================================================

print("\nUploading app.py...")
api.upload_file(
    path_or_fileobj=app_code.encode(),
    path_in_repo="app.py",
    repo_id=FULL_SPACE,
    repo_type="space",
    token=HF_TOKEN,
)

print("Uploading requirements.txt...")
api.upload_file(
    path_or_fileobj=requirements.encode(),
    path_in_repo="requirements.txt",
    repo_id=FULL_SPACE,
    repo_type="space",
    token=HF_TOKEN,
)

print("Uploading README.md...")
api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id=FULL_SPACE,
    repo_type="space",
    token=HF_TOKEN,
)

print("\n" + "=" * 50)
print("Space deployed successfully!")
print(f"URL: https://huggingface.co/spaces/{FULL_SPACE}")
print("\nNOTE: HuggingFace will now build your Space.")
print("This takes 3-5 minutes.")
print("Watch the build logs at the URL above.")
print("=" * 50) 

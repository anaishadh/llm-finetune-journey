# LLM Fine-Tuning Journey 🤖

A hands-on project demonstrating how to fine-tune a large language model (LLM) on custom data using QLoRA — the same technique used by companies to build domain-specific AI assistants.

## Project: NovaBrew Customer Support Bot
Fine-tuned **Qwen2.5-1.5B-Instruct** to act as a customer support agent for NovaBrew, a fictional smart coffee machine company. The base model has zero knowledge of NovaBrew — all product knowledge comes purely from fine-tuning.

## What This Demonstrates
- Setting up a professional ML development environment
- Understanding and applying **QLoRA** (Quantized Low-Rank Adaptation)
- Preparing and formatting custom training data in **ChatML format**
- Fine-tuning an LLM on domain-specific data
- Running inference with a fine-tuned model + LoRA adapters
- Before/after comparison showing fine-tuning effectiveness

## Tech Stack
| Tool | Purpose |
|------|---------|
| PyTorch | Deep learning framework |
| HuggingFace Transformers | Model loading and tokenization |
| PEFT | LoRA adapter implementation |
| TRL | SFT (Supervised Fine-Tuning) trainer |
| bitsandbytes | 4-bit quantization |
| Datasets | Data loading and processing |

## Hardware Used
- CPU: Intel i9
- GPU: NVIDIA RTX 4090 (25.8GB VRAM)
- RAM: 64GB

## Project Structure
llm-finetune-journey/
├── data/
│   ├── novabrew_dataset.jsonl     ← raw training examples
│   └── novabrew_prepared/         ← formatted dataset
├── scripts/
│   ├── 01_verify_setup.py         ← environment verification
│   ├── 02_prepare_data.py         ← dataset preparation
│   ├── 03_finetune.py             ← QLoRA fine-tuning
│   └── 04_inference.py            ← chat with the model
├── outputs/
│   └── novabrew-model/            ← saved LoRA adapters
├── requirements.txt
└── README.md

## What's Next
- [ ] Expand dataset to 100+ examples for better generalization
- [ ] Add evaluation metrics (ROUGE, BERTScore)
- [ ] Experiment with different LoRA ranks (r=8, r=32, r=64)
- [ ] Try larger base models (7B, 13B)
- [ ] Deploy as a REST API with FastAPI

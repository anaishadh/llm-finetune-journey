# LLM Fine-Tuning Journey 🤖

A hands-on project demonstrating how to fine-tune a large language model (LLM) on custom data using QLoRA, and how to build a RAG (Retrieval Augmented Generation) system — the same techniques used by companies to build domain-specific AI assistants.

## Project: NovaBrew Customer Support Bot

Fine-tuned **Qwen2.5-1.5B-Instruct** to act as a customer support agent for NovaBrew, a fictional smart coffee machine company. Also built a full RAG pipeline for the same use case — demonstrating both approaches and their tradeoffs.

The base model has zero knowledge of NovaBrew — all product knowledge comes purely from fine-tuning and/or retrieved documents.

## What This Demonstrates

- Setting up a professional ML development environment
- Understanding and applying **QLoRA** (Quantized Low-Rank Adaptation)
- Preparing and formatting custom training data in **ChatML format**
- Fine-tuning an LLM on domain-specific data
- Running inference with a fine-tuned model + LoRA adapters
- Building a **RAG pipeline** with semantic search
- Using **vector embeddings** to find relevant documents by meaning
- Storing and querying vectors with **ChromaDB**
- Understanding the tradeoffs between fine-tuning and RAG

## Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Deep learning framework |
| HuggingFace Transformers | Model loading and tokenization |
| PEFT | LoRA adapter implementation |
| TRL | SFT (Supervised Fine-Tuning) trainer |
| bitsandbytes | 4-bit quantization |
| Datasets | Data loading and processing |
| sentence-transformers | Text embedding model |
| ChromaDB | Local vector database |

## Hardware Used

- CPU: Intel i9
- GPU: NVIDIA RTX 4090 (25.8GB VRAM)
- RAM: 64GB

## Project Structure

```
llm-finetune-journey/
├── data/
│   ├── novabrew_dataset.jsonl     <- raw training examples (fine-tuning)
│   ├── novabrew_docs.txt          <- knowledge base (RAG)
│   ├── novabrew_prepared/         <- formatted dataset
│   └── novabrew_vectordb/         <- ChromaDB vector store
├── scripts/
│   ├── 01_verify_setup.py         <- environment verification
│   ├── 02_prepare_data.py         <- dataset preparation
│   ├── 03_finetune.py             <- QLoRA fine-tuning
│   ├── 04_inference.py            <- chat with fine-tuned model
│   ├── 05_build_rag.py            <- build vector database
│   └── 06_rag_inference.py        <- chat using RAG pipeline
├── outputs/
│   └── novabrew-model/            <- saved LoRA adapters
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/yourusername/llm-finetune-journey
cd llm-finetune-journey
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify your environment

```bash
python scripts/01_verify_setup.py
```

### 3. Prepare the dataset

```bash
python scripts/02_prepare_data.py
```

### 4. Fine-tune the model

```bash
python scripts/03_finetune.py
```

### 5. Chat with your fine-tuned bot

```bash
python scripts/04_inference.py
```

### 6. Build the RAG knowledge base

```bash
python scripts/05_build_rag.py
```

### 7. Chat using RAG

```bash
python scripts/06_rag_inference.py
```

## Key Concepts

### QLoRA
Quantized Low-Rank Adaptation. Instead of fine-tuning all 1.5 billion parameters, LoRA adds small trainable adapter layers (~4M parameters) on top of the frozen base model. Quantization compresses the base model to 4-bit precision, reducing VRAM usage significantly. This makes fine-tuning possible on consumer GPUs.

### ChatML Format
The prompt template used during both training and inference:

```
<|im_start|>system
You are a helpful assistant...<|im_end|>
<|im_start|>user
How do I reset NovaBrew?<|im_end|>
<|im_start|>assistant
To reset NovaBrew...<|im_end|>
```

### Supervised Fine-Tuning (SFT)
Training the model on labeled examples of desired behavior. The model learns to associate specific inputs with specific outputs by seeing many examples.

### Embeddings
Text converted into vectors (arrays of numbers) that capture semantic meaning. Similar meaning = similar vectors. This enables searching by meaning rather than exact keywords.

```
"machine won't turn on" -> [0.23, 0.87, 0.12, ...]
"power issues"          -> [0.25, 0.84, 0.11, ...]  <- similar!
"I like pizza"          -> [0.91, 0.02, 0.67, ...]  <- very different
```

### RAG (Retrieval Augmented Generation)
Instead of baking knowledge into model weights, RAG retrieves relevant documents at runtime and injects them into the prompt as context. The model answers based on what it was given, not what it memorized.

## Fine-Tuning vs RAG

| | Fine-Tuning | RAG |
|---|---|---|
| Knowledge location | Baked into weights | Fetched at runtime |
| Updating knowledge | Retrain required | Update docs only |
| Handles large doc sets | Limited | Scales easily |
| Response traceability | Hard | Easy (show source) |
| Tone and personality | Deeply learned | Relies on system prompt |
| Works offline | Yes | Needs vector DB |
| Best for | Behavior and style | Facts and knowledge |

**Production systems typically use both together.**

## Results

### Fine-tuned model
- Answers NovaBrew-specific questions it was trained on
- Handles questions it was **never trained on** by generalizing
- Stays in character as a NovaBrew support agent
- Redirects off-topic questions back to NovaBrew context

### RAG system
- Retrieves relevant docs by semantic meaning not keywords
- Answers grounded in actual documentation
- Handles rude or off-topic messages gracefully
- Transparent: you can see exactly which docs were retrieved

## Example Conversations

### Fine-tuned model

```
User: How do I reset NovaBrew to factory settings?
Bot:  Hold the power button and the brew button simultaneously
      for 10 seconds until the LED flashes red three times...

User: My NovaBrew is making a strange noise.
Bot:  I am sorry to hear that. Could you please tell me which
      part of the device it is coming from?

User: What is the turnover of NovaBrew?
Bot:  I cannot provide financial details. I would be happy to
      help with any questions about using your NovaBrew device!
```

### RAG system

```
User: context What do the LED colors mean?
Bot:  [Retrieved: ERROR CODES section]
      LED flashes red twice means the water tank is empty...

User: How do I get a refund?
Bot:  NovaBrew offers a 30-day money back guarantee. To initiate
      a return visit novabrew.com/returns and enter your order number...
```

## What's Next

- [ ] Expand dataset to 100+ examples for better generalization
- [ ] Add evaluation metrics (ROUGE, BERTScore)
- [ ] Experiment with different LoRA ranks (r=8, r=32, r=64)
- [ ] Try larger base models (7B, 13B)
- [ ] Deploy as a REST API with FastAPI
- [ ] Combine fine-tuning + RAG in a single pipeline
- [ ] Add conversation memory for multi-turn chat

## License

MIT

# ============================================================
# Stage 3: Fine-tuning NovaBrew Customer Support Bot
# Using: QLoRA (Quantized Low-Rank Adaptation)
# Model: Qwen2.5-1.5B-Instruct (small enough for fast training)
# ============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk

# ============================================================
# STEP 1: CONFIGURATION
# All your settings in one place — easy to experiment with
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # ~3GB download, fits easily on 4090
OUTPUT_DIR = "outputs/novabrew-model"
DATASET_PATH = "data/novabrew_prepared"

# QLoRA settings — quantize the base model to 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # load model in 4-bit precision
    bnb_4bit_quant_type="nf4",      # nf4 is best for LLMs
    bnb_4bit_compute_dtype=torch.float16,  # compute in float16
    bnb_4bit_use_double_quant=True, # quantize the quantization constants too
)

# LoRA settings
lora_config = LoraConfig(
    r=16,                    # rank — how much capacity to learn
    lora_alpha=32,           # scaling (usually 2x rank)
    target_modules=[         # which layers to add adapters to
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,       # prevent overfitting
    bias="none",             # don't train bias terms
    task_type="CAUSAL_LM",   # we're doing causal language modeling
)

# Training settings
training_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,          # go through dataset 10 times
                                  # small dataset = more epochs needed
    per_device_train_batch_size=2,# process 2 examples at a time
    gradient_accumulation_steps=4,# accumulate gradients over 4 steps
                                  # effective batch size = 2x4 = 8
    learning_rate=2e-4,           # how big each learning step is
    fp16=True,                    # use float16 for faster training
    logging_steps=1,              # print loss every step
    save_strategy="epoch",        # save checkpoint every epoch
    warmup_ratio=0.1,             # gradually increase lr at start
    lr_scheduler_type="cosine",   # learning rate schedule
    max_seq_length=512,           # max token length per example
    dataset_text_field="text",    # which field in dataset to train on
)

# ============================================================
# STEP 2: LOAD MODEL AND TOKENIZER
# ============================================================

print("=" * 50)
print("Loading model and tokenizer...")
print(f"Model: {MODEL_NAME}")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # use end token for padding

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,   # load in 4-bit
    device_map="auto",                # automatically use GPU
    trust_remote_code=True,
)

# ============================================================
# STEP 3: APPLY LORA
# ============================================================

print("\nPreparing model for QLoRA training...")
model = prepare_model_for_kbit_training(model)  # prepare 4-bit model
model = get_peft_model(model, lora_config)       # add LoRA adapters

# Show how many parameters we're actually training
model.print_trainable_parameters()

# ============================================================
# STEP 4: LOAD DATASET
# ============================================================

print("\nLoading NovaBrew dataset...")
dataset = load_from_disk(DATASET_PATH)
print(f"Train examples: {len(dataset['train'])}")
print(f"Test examples:  {len(dataset['test'])}")

# ============================================================
# STEP 5: TRAIN
# ============================================================

print("\nStarting training...")
print("Watch the 'loss' number — it should go DOWN over time.")
print("That means the model is learning NovaBrew!\n")

trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
)

trainer.train()

# ============================================================
# STEP 6: SAVE
# ============================================================

print("\nSaving fine-tuned model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("=" * 50)
print("Training complete!")
print(f"Model saved to: {OUTPUT_DIR}")
print("Run 04_inference.py to chat with your NovaBrew bot!")
print("=" * 50)
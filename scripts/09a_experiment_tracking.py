# ============================================================
# Stage 9a: Experiment Tracking with Weights & Biases
#
# Problem without tracking:
# "Which training run gave the best results?"
# "What hyperparameters did we use?"
# "Why did model v3 perform worse than v2?"
#
# With wandb you get:
# - Every training run logged automatically
# - Loss curves visualized in browser
# - Hyperparameters saved per run
# - Compare runs side by side
# - Share results with your team
# ============================================================

import wandb
import torch
import random
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
import os
from dotenv import load_dotenv

load_dotenv()  # loads WANDB_API_KEY from .env file

# ============================================================
# STEP 1: UNDERSTAND EXPERIMENT TRACKING
#
# The core idea: every training run is an "experiment"
# You log:
#   - hyperparameters (what settings you used)
#   - metrics (loss, accuracy over time)
#   - artifacts (the actual model files)
#
# Then you can compare experiments:
#   Run 1: r=8,  lr=2e-4 → final loss: 1.8
#   Run 2: r=16, lr=2e-4 → final loss: 1.6  ← better!
#   Run 3: r=16, lr=1e-4 → final loss: 1.5  ← best!
# ============================================================

# ============================================================
# STEP 2: DEFINE EXPERIMENTS TO COMPARE
#
# We'll run 3 mini training runs with different LoRA ranks
# to demonstrate how tracking helps you find best settings
# ============================================================

experiments = [
    {
        "name": "novabrew-r8",
        "r": 8,
        "lora_alpha": 16,
        "learning_rate": 2e-4,
        "epochs": 5,
    },
    {
        "name": "novabrew-r16",
        "r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "epochs": 5,
    },
    {
        "name": "novabrew-r16-lowlr",
        "r": 16,
        "lora_alpha": 32,
        "learning_rate": 1e-4,
        "epochs": 5,
    },
]

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH = "data/novabrew_prepared"

def run_experiment(config):
    """
    Run one training experiment with given config.
    Everything gets logged to wandb automatically.
    """
    print(f"\n{'='*50}")
    print(f"Starting experiment: {config['name']}")
    print(f"r={config['r']}, lr={config['learning_rate']}")
    print("="*50)

    # --------------------------------------------------------
    # Initialize wandb run
    # Each run gets a unique ID and is tracked separately
    # --------------------------------------------------------
    run = wandb.init(
        project="novabrew-finetuning",    # groups all runs together
        name=config["name"],              # this run's name
        config={                          # hyperparameters to track
            "model": MODEL_NAME,
            "r": config["r"],
            "lora_alpha": config["lora_alpha"],
            "learning_rate": config["learning_rate"],
            "epochs": config["epochs"],
            "dataset": "novabrew_10examples",
            "quantization": "4bit",
        },
        tags=["novabrew", "qlora", "qwen2.5"],  # searchable tags
    )

    # --------------------------------------------------------
    # Load model with QLoRA
    # --------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    lora_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Log trainable parameters to wandb
    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    wandb.log({
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": 100 * trainable / total
    })

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    dataset = load_from_disk(DATASET_PATH)

    training_config = SFTConfig(
        output_dir=f"outputs/{config['name']}",
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=config["learning_rate"],
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_seq_length=512,
        dataset_text_field="text",
        report_to="wandb",     # ← this line sends all metrics to wandb
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
    )

    # Train and capture results
    train_result = trainer.train()
    final_loss = train_result.training_loss

    # Log final metrics
    wandb.log({
        "final_loss": final_loss,
        "training_runtime": train_result.metrics["train_runtime"],
    })

    print(f"\nExperiment {config['name']} complete!")
    print(f"Final loss: {final_loss:.4f}")

    # Save model
    trainer.save_model(f"outputs/{config['name']}")

    # Close wandb run
    wandb.finish()

    # Clean up GPU memory between runs
    del model
    torch.cuda.empty_cache()

    return final_loss

# ============================================================
# STEP 3: RUN ALL EXPERIMENTS
# ============================================================

print("=" * 50)
print("NovaBrew Experiment Tracking")
print("Open wandb.ai to watch results in real time!")
print("=" * 50)

results = []
for config in experiments:
    loss = run_experiment(config)
    results.append({"name": config["name"], "loss": loss})

# ============================================================
# STEP 4: COMPARE RESULTS
# ============================================================

print("\n" + "=" * 50)
print("EXPERIMENT RESULTS SUMMARY")
print("=" * 50)

results.sort(key=lambda x: x["loss"])
for i, r in enumerate(results, 1):
    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
    print(f"{medal} {r['name']}: loss = {r['loss']:.4f}")

best = results[0]
print(f"\nBest config: {best['name']}")
print(f"Check wandb.ai for full charts and comparison!")

# ============================================================
# STEP 5: LOG FINAL COMPARISON TO WANDB
# ============================================================

run = wandb.init(
    project="novabrew-finetuning",
    name="experiment-comparison",
    job_type="comparison"
)

# Log comparison table
comparison_table = wandb.Table(
    columns=["experiment", "final_loss", "rank"],
    data=[[r["name"], r["loss"], i+1] for i, r in enumerate(results)]
)

wandb.log({"experiment_comparison": comparison_table})
wandb.finish()

print("\nAll results logged to wandb!")
print("Visit wandb.ai to see your experiment dashboard!") 

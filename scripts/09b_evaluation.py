# ============================================================
# Stage 9b: Model Evaluation
#
# Training loss going down is necessary but not sufficient.
# You need to verify the model actually answers correctly.
#
# We'll measure 4 things companies care about:
# 1. Relevance  - did it answer the actual question?
# 2. Faithfulness - did it stick to NovaBrew facts?
# 3. Character  - did it stay in support agent role?
# 4. Comparison - is it better than base model?
# ============================================================

import torch
import wandb
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# STEP 1: DEFINE EVALUATION DATASET
#
# These are test questions with:
# - expected_keywords: words that MUST appear in good answer
# - forbidden_keywords: words that should NOT appear
# - should_mention_novabrew: must stay in character
#
# In production companies have hundreds of these.
# We have 8 — enough to demonstrate the concept.
# ============================================================

eval_dataset = [
    {
        "question": "How do I reset NovaBrew to factory settings?",
        "expected_keywords": ["power button", "brew button", "10 seconds", "red"],
        "forbidden_keywords": ["I don't know", "I cannot", "unclear"],
        "should_mention_novabrew": True,
        "category": "troubleshooting"
    },
    {
        "question": "What coffee pods work with NovaBrew?",
        "expected_keywords": ["nespresso", "pods", "compatible"],
        "forbidden_keywords": ["I don't know", "I cannot"],
        "should_mention_novabrew": True,
        "category": "product_info"
    },
    {
        "question": "How do I connect NovaBrew to WiFi?",
        "expected_keywords": ["app", "sync", "blue"],
        "forbidden_keywords": ["I don't know", "I cannot"],
        "should_mention_novabrew": True,
        "category": "setup"
    },
    {
        "question": "What is the warranty period?",
        "expected_keywords": ["2", "year", "warranty"],
        "forbidden_keywords": ["I don't know", "I cannot"],
        "should_mention_novabrew": True,
        "category": "warranty"
    },
    {
        "question": "My NovaBrew won't turn on.",
        "expected_keywords": ["power", "cable", "outlet"],
        "forbidden_keywords": ["I don't know", "I cannot"],
        "should_mention_novabrew": True,
        "category": "troubleshooting"
    },
    {
        "question": "How do I schedule a brew?",
        "expected_keywords": ["app", "schedule"],
        "forbidden_keywords": ["I don't know", "I cannot"],
        "should_mention_novabrew": True,
        "category": "features"
    },
    {
        "question": "What is the capital of France?",
        "expected_keywords": ["novabrew", "support@novabrew.com"],
        "forbidden_keywords": ["Paris", "France", "capital"],
        "should_mention_novabrew": True,
        "category": "off_topic"
    },
    {
        "question": "How do I clean my machine?",
        "expected_keywords": ["water tank", "clean", "descale"],
        "forbidden_keywords": ["I don't know", "I cannot"],
        "should_mention_novabrew": True,
        "category": "maintenance"
    },
]

# ============================================================
# STEP 2: SCORING FUNCTIONS
#
# Each function returns a score between 0 and 1.
# 1.0 = perfect, 0.0 = completely wrong
# ============================================================

def score_keywords(response: str, keywords: list) -> float:
    """
    Check what percentage of expected keywords appear in response.
    0.0 = none found, 1.0 = all found
    """
    response_lower = response.lower()
    found = sum(1 for kw in keywords if kw.lower() in response_lower)
    return found / len(keywords) if keywords else 1.0

def score_forbidden(response: str, forbidden: list) -> float:
    """
    Check that forbidden words don't appear.
    1.0 = none found (good), 0.0 = all found (bad)
    """
    response_lower = response.lower()
    found = sum(1 for kw in forbidden if kw.lower() in response_lower)
    return 1.0 - (found / len(forbidden)) if forbidden else 1.0

def score_character(response: str, should_mention: bool) -> float:
    """
    Check if response stays in NovaBrew character.
    """
    novabrew_signals = [
        "novabrew",
        "support@novabrew.com",
        "our",
        "we",
    ]
    response_lower = response.lower()
    found = any(signal in response_lower for signal in novabrew_signals)
    if should_mention:
        return 1.0 if found else 0.0
    return 1.0

def evaluate_response(question: str, response: str, eval_item: dict) -> dict:
    """
    Score a single response on all metrics.
    Returns dict of scores.
    """
    keyword_score = score_keywords(response, eval_item["expected_keywords"])
    forbidden_score = score_forbidden(response, eval_item["forbidden_keywords"])
    character_score = score_character(response, eval_item["should_mention_novabrew"])

    # Overall score is average of all metrics
    overall = (keyword_score + forbidden_score + character_score) / 3

    return {
        "keyword_score": keyword_score,
        "forbidden_score": forbidden_score,
        "character_score": character_score,
        "overall_score": overall,
    }

# ============================================================
# STEP 3: LOAD MODELS
# We evaluate both base model and fine-tuned model
# to measure improvement
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FINETUNED_MODEL = "outputs/novabrew-r16"  # best from experiment

SYSTEM_PROMPT = """You are a friendly and helpful customer support agent for NovaBrew,
a smart coffee machine company. Answer questions accurately and politely
based on NovaBrew product knowledge. If you don't know the answer,
direct the customer to support@novabrew.com."""

def load_base_model():
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer

def load_finetuned_model(base_model, tokenizer):
    print("Loading fine-tuned model...")
    model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, question: str) -> str:
    """Generate a response for a given question"""
    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
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
    )
    return response.strip()

# ============================================================
# STEP 4: RUN EVALUATION
# ============================================================

def run_evaluation(model, tokenizer, model_name: str, wandb_run):
    """Evaluate a model on all test questions"""
    print(f"\nEvaluating {model_name}...")
    print("-" * 40)

    all_scores = []
    results = []

    for item in eval_dataset:
        question = item["question"]
        response = generate_response(model, tokenizer, question)
        scores = evaluate_response(question, response, item)

        all_scores.append(scores["overall_score"])
        results.append({
            "question": question,
            "category": item["category"],
            "response": response[:100] + "...",
            **scores
        })

        # Print result
        emoji = "✅" if scores["overall_score"] >= 0.7 else "⚠️" if scores["overall_score"] >= 0.4 else "❌"
        print(f"{emoji} [{item['category']}] Score: {scores['overall_score']:.2f}")
        print(f"   Q: {question[:50]}")
        print(f"   A: {response[:80]}...")
        print()

    # Calculate averages
    avg_score = sum(all_scores) / len(all_scores)

    # Log to wandb
    eval_table = wandb.Table(
        columns=["question", "category", "response",
                 "keyword_score", "forbidden_score",
                 "character_score", "overall_score"],
        data=[[r["question"], r["category"], r["response"],
               r["keyword_score"], r["forbidden_score"],
               r["character_score"], r["overall_score"]]
              for r in results]
    )

    wandb_run.log({
        f"{model_name}/avg_score": avg_score,
        f"{model_name}/eval_table": eval_table,
    })

    return avg_score, results

# ============================================================
# STEP 5: MAIN - COMPARE BASE VS FINE-TUNED
# ============================================================

print("=" * 50)
print("NovaBrew Model Evaluation")
print("=" * 50)

# Initialize wandb
run = wandb.init(
    project="novabrew-finetuning",
    name="model-evaluation",
    job_type="evaluation"
)

# Load base model
base_model, tokenizer = load_base_model()

# Evaluate base model
base_score, base_results = run_evaluation(
    base_model, tokenizer, "base_model", run
)

# Load fine-tuned adapters on top
finetuned_model, tokenizer = load_finetuned_model(base_model, tokenizer)

# Evaluate fine-tuned model
ft_score, ft_results = run_evaluation(
    finetuned_model, tokenizer, "finetuned_model", run
)

# ============================================================
# STEP 6: FINAL COMPARISON
# ============================================================

print("=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"Base model score:       {base_score:.2f}")
print(f"Fine-tuned model score: {ft_score:.2f}")
improvement = ((ft_score - base_score) / base_score) * 100
print(f"Improvement:            {improvement:+.1f}%")

if ft_score > base_score:
    print("\n✅ Fine-tuning improved the model!")
elif ft_score == base_score:
    print("\n➡️ No change — consider more training data")
else:
    print("\n⚠️ Fine-tuning hurt performance — check your data quality")

# Log comparison
run.log({
    "base_model_score": base_score,
    "finetuned_model_score": ft_score,
    "improvement_percent": improvement,
})

wandb.finish()

print("\nDetailed results logged to wandb!")
print("Check the eval_table in your wandb dashboard")
print("to see per-question scores for both models.") 

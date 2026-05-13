# ============================================================
# Stage 9c: Production Monitoring
#
# Problem without monitoring:
# - Users get bad responses and you never know
# - Costs spiral out of control
# - Model degrades over time undetected
#
# With monitoring you track every single call:
# - What was asked
# - What was answered
# - How long it took
# - How confident the model was
# - Whether it stayed in character
# ============================================================

import time
import json
import torch
import wandb
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# STEP 1: THE MONITOR CLASS
#
# In production this would write to a database.
# We write to a local JSON file + wandb.
# The concept is identical — log everything.
# ============================================================

class ProductionMonitor:
    """
    Logs every model call with:
    - input/output
    - latency
    - quality signals
    - anomaly flags
    """

    def __init__(self, log_file="outputs/production_logs.jsonl"):
        self.log_file = log_file
        self.calls = []
        self.session_start = datetime.now()

        # Initialize wandb for real-time monitoring dashboard
        self.wandb_run = wandb.init(
            project="novabrew-finetuning",
            name=f"production-monitor-{datetime.now().strftime('%Y%m%d-%H%M')}",
            job_type="monitoring"
        )

        # Running stats
        self.total_calls = 0
        self.total_latency = 0
        self.flagged_calls = 0

        print("Monitor initialized. Logging all calls...")

    def log_call(self, question: str, response: str,
                 latency_ms: float, model_version: str):
        """Log a single model call with quality analysis"""

        self.total_calls += 1
        self.total_latency += latency_ms

        # ------------------------------------------------
        # Quality signals — automatic checks on every call
        # ------------------------------------------------

        # Signal 1: Did it stay in character?
        in_character = any(word in response.lower() for word in
                          ["novabrew", "support@novabrew", "our", "we"])

        # Signal 2: Did it refuse to answer?
        refused = any(phrase in response.lower() for phrase in
                     ["i cannot", "i don't know", "i am unable",
                      "as an ai", "i'm not able"])

        # Signal 3: Is response too short? (likely unhelpful)
        too_short = len(response.split()) < 10

        # Signal 4: Is response too long? (likely rambling)
        too_long = len(response.split()) > 200

        # Signal 5: Did it mention contact info?
        has_contact = "support@novabrew.com" in response.lower()

        # Flag call if quality signals are bad
        flagged = refused or too_short or too_long or not in_character
        if flagged:
            self.flagged_calls += 1

        # Build log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "call_id": self.total_calls,
            "model_version": model_version,
            "question": question,
            "response": response,
            "metrics": {
                "latency_ms": round(latency_ms, 2),
                "response_words": len(response.split()),
                "in_character": in_character,
                "refused": refused,
                "too_short": too_short,
                "too_long": too_long,
                "has_contact": has_contact,
                "flagged": flagged,
            }
        }

        self.calls.append(log_entry)

        # Write to local file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")

        # Log to wandb in real time
        self.wandb_run.log({
            "latency_ms": latency_ms,
            "response_words": len(response.split()),
            "in_character": int(in_character),
            "refused": int(refused),
            "flagged": int(flagged),
            "total_calls": self.total_calls,
            "flag_rate": self.flagged_calls / self.total_calls,
            "avg_latency": self.total_latency / self.total_calls,
        })

        # Print live status
        flag_icon = "🚨" if flagged else "✅"
        print(f"{flag_icon} Call #{self.total_calls} | "
              f"{latency_ms:.0f}ms | "
              f"{'in-character' if in_character else 'OUT OF CHARACTER'} | "
              f"{'FLAGGED' if flagged else 'ok'}")

        return log_entry

    def print_summary(self):
        """Print session summary"""
        duration = (datetime.now() - self.session_start).seconds

        print("\n" + "=" * 50)
        print("MONITORING SESSION SUMMARY")
        print("=" * 50)
        print(f"Total calls:      {self.total_calls}")
        print(f"Flagged calls:    {self.flagged_calls} "
              f"({100*self.flagged_calls/max(self.total_calls,1):.1f}%)")
        print(f"Avg latency:      {self.total_latency/max(self.total_calls,1):.0f}ms")
        print(f"Session duration: {duration}s")

        # Category breakdown
        flagged = [c for c in self.calls if c["metrics"]["flagged"]]
        if flagged:
            print(f"\n⚠️  Flagged calls need review:")
            for call in flagged:
                print(f"   - {call['question'][:60]}")
                reasons = []
                m = call["metrics"]
                if m["refused"]:      reasons.append("refused to answer")
                if m["too_short"]:    reasons.append("too short")
                if m["too_long"]:     reasons.append("too long")
                if not m["in_character"]: reasons.append("out of character")
                print(f"     Reason: {', '.join(reasons)}")

        # Log summary to wandb
        self.wandb_run.log({
            "summary/total_calls": self.total_calls,
            "summary/flag_rate": self.flagged_calls / max(self.total_calls, 1),
            "summary/avg_latency_ms": self.total_latency / max(self.total_calls, 1),
        })

        self.wandb_run.finish()
        print(f"\nFull logs saved to: {self.log_file}")
        print("Check wandb dashboard for monitoring charts!")

# ============================================================
# STEP 2: LOAD MODEL
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FINETUNED_MODEL = "outputs/novabrew-r16"

SYSTEM_PROMPT = """You are a friendly and helpful customer support agent for NovaBrew,
a smart coffee machine company. Answer questions accurately and politely.
If you don't know the answer, direct to support@novabrew.com."""

print("=" * 50)
print("NovaBrew Production Monitor")
print("=" * 50)

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
model.eval()
print("Model loaded!")

# ============================================================
# STEP 3: MONITORED INFERENCE FUNCTION
#
# This wraps our normal generate function with monitoring.
# Every call is automatically logged.
# This is the pattern used in production APIs.
# ============================================================

monitor = ProductionMonitor()

def monitored_chat(question: str) -> str:
    """
    Generate response AND log everything about the call.
    In production this function sits behind your API endpoint.
    """
    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Measure latency
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    latency_ms = (time.time() - start_time) * 1000

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Log everything
    monitor.log_call(
        question=question,
        response=response,
        latency_ms=latency_ms,
        model_version="novabrew-r16-qlora"
    )

    return response

# ============================================================
# STEP 4: SIMULATE PRODUCTION TRAFFIC
#
# Mix of normal, edge case, and problematic questions.
# This is how you stress-test before real deployment.
# ============================================================

print("\nSimulating production traffic...")
print("-" * 50)

production_questions = [
    # Normal support questions
    "How do I reset my NovaBrew?",
    "What pods are compatible with NovaBrew?",
    "My NovaBrew won't connect to WiFi.",

    # Edge cases
    "My NovaBrew is making a weird noise.",
    "Can I use NovaBrew to make tea?",

    # Off-topic — should redirect
    "What's the weather like today?",
    "Who is the president of the United States?",

    # Potentially problematic
    "I want a refund immediately, this is terrible!",
    "Your product is garbage and broke after one day.",

    # Normal again
    "How often should I descale my machine?",
    "How do I adjust the brew strength?",
    "What does the orange LED mean?",
]

responses = {}
for question in production_questions:
    response = monitored_chat(question)
    responses[question] = response

# ============================================================
# STEP 5: PRINT SUMMARY AND FLAGGED CALLS
# ============================================================

monitor.print_summary()

# Show a few actual responses
print("\n" + "=" * 50)
print("SAMPLE RESPONSES")
print("=" * 50)

sample_questions = [
    "I want a refund immediately, this is terrible!",
    "What's the weather like today?",
    "What does the orange LED mean?",
]

for q in sample_questions:
    print(f"\nQ: {q}")
    print(f"A: {responses[q][:150]}") 

from datasets import Dataset
import json

# ============================================================
# WHAT IS HAPPENING HERE:
# Models don't just take "prompt" and "completion" raw.
# They expect a specific format called a "chat template".
# This is what tells the model:
#   - who is speaking (user vs assistant)
#   - where the question ends and answer begins
# ============================================================

def load_raw_data(filepath):
    """Load our hand-crafted jsonl dataset"""
    examples = []
    with open(filepath, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples

def format_as_chat(example):
    """
    Convert this:
        {"prompt": "How do I reset?", "completion": "Hold the button..."}
    
    Into this:
        <|im_start|>system
        You are a helpful NovaBrew customer support agent...
        <|im_end|>
        <|im_start|>user
        How do I reset?
        <|im_end|>
        <|im_start|>assistant
        Hold the button...
        <|im_end|>
    
    This format is called "ChatML" - most modern models understand it.
    """
    system_prompt = (
        "You are a friendly and helpful customer support agent for NovaBrew, "
        "a smart coffee machine company. Answer questions accurately and politely "
        "based on NovaBrew product knowledge. If you don't know the answer, "
        "direct the customer to support@novabrew.com."
    )
    
    text = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{example['prompt']}<|im_end|>
<|im_start|>assistant
{example['completion']}<|im_end|>"""
    
    return {"text": text}

def main():
    print("Loading raw dataset...")
    raw_data = load_raw_data('data/novabrew_dataset.jsonl')
    print(f"Loaded {len(raw_data)} examples")
    
    print("\nFormatting into chat template...")
    formatted = [format_as_chat(ex) for ex in raw_data]
    
    print("\nSample formatted example:")
    print("-" * 40)
    print(formatted[0]['text'])
    print("-" * 40)
    
    print("\nConverting to HuggingFace Dataset...")
    dataset = Dataset.from_list(formatted)
    
    # Split into train and test (80/20)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    print(f"\nTrain examples: {len(dataset['train'])}")
    print(f"Test examples:  {len(dataset['test'])}")
    
    # Save to disk
    dataset.save_to_disk('data/novabrew_prepared')
    print("\nDataset saved to data/novabrew_prepared")
    print("Ready for training!")

if __name__ == "__main__":
    main()
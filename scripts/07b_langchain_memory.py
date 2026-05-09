# ============================================================
# Stage 7b: LangChain + Memory + RAG (Fixed)
#
# Changes from previous version:
# - History formatted as plain text instead of message objects
# - Added stop tokens so model knows when to stop
# - Cleaner prompt structure for smaller models
# ============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
import chromadb

# ============================================================
# STEP 1: LOAD EVERYTHING
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("=" * 50)
print("Loading NovaBrew Support Bot with Memory")
print("=" * 50)

print("\n[1/3] Loading language model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.3,          # lower = more focused responses
    do_sample=True,
    return_full_text=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

print("[2/3] Loading RAG system...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="data/novabrew_vectordb")
collection = client.get_collection("novabrew_docs")
print(f"      Loaded {collection.count()} chunks")

print("[3/3] Building chain...")

# ============================================================
# STEP 2: SIMPLER PROMPT — history as plain text
#
# Instead of MessagesPlaceholder which confuses small models,
# we format history as readable text directly in the prompt.
# This is cleaner and works better with smaller models.
# ============================================================

prompt = PromptTemplate.from_template("""<|im_start|>system
You are a NovaBrew customer support agent. Answer ONLY the customer's latest question.
Use the context below. If unsure, direct to support@novabrew.com.
Give a short, direct answer. Do not ask questions back.

Context:
{context}

Conversation so far:
{history}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
""")

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# ============================================================
# STEP 3: MEMORY AS PLAIN TEXT
#
# We store history as a simple formatted string.
# Each turn is "Customer: ... \nAgent: ..."
# Much easier for small models to understand.
# ============================================================

conversation_history = []  # list of (question, answer) tuples

def format_history():
    """Convert history list to readable text for the prompt"""
    if not conversation_history:
        return "No previous messages."
    
    formatted = []
    for human, ai in conversation_history:
        formatted.append(f"Customer: {human}")
        formatted.append(f"Agent: {ai}")
    return "\n".join(formatted)

def chat_with_memory(question, show_context=False):
    """Full RAG + memory chat pipeline"""
    
    # Retrieve relevant docs
    query_vector = embedding_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=2)
    context = "\n\n".join(results['documents'][0])

    if show_context:
        print(f"  [Context]: {context[:80]}...")

    # Build response
    response = chain.invoke({
        "context": context,
        "history": format_history(),
        "question": question
    })

    # Clean up — stop at any role markers the model might generate
    response = response.strip()
    for stop in ["Customer:", "Human:", "User:", "<|im_start|>", "\n\n\n"]:
        if stop in response:
            response = response.split(stop)[0].strip()

    # Save to history
    conversation_history.append((question, response))

    return response

# ============================================================
# STEP 4: DEMONSTRATE MEMORY
# ============================================================

print("\nReady!")
print("=" * 50)
print("MEMORY DEMONSTRATION")
print("=" * 50)

demo = [
    "My NovaBrew won't turn on.",
    "I already checked the cable and it's fine.",
    "What should I try next?",
    "How long is the warranty?",
    "Does it cover the power issue I described?",
]

for message in demo:
    print(f"\nYou: {message}")
    response = chat_with_memory(message)
    print(f"Bot: {response}")

print("\n" + "=" * 50)
print(f"History stored: {len(conversation_history)} exchanges")
print("=" * 50)

# Reset for interactive mode
conversation_history.clear()

# ============================================================
# STEP 5: INTERACTIVE CHAT
# ============================================================

print("\nNovaBrew Support Bot")
print("Commands: 'reset' 'history' 'quit'")
print("=" * 50)

while True:
    user_input = input("\nYou: ").strip()

    if not user_input:
        continue

    if user_input.lower() == 'quit':
        break

    if user_input.lower() == 'reset':
        conversation_history.clear()
        print("History cleared!")
        continue

    if user_input.lower() == 'history':
        if not conversation_history:
            print("No history yet.")
        else:
            print("\n--- Conversation History ---")
            for i, (q, a) in enumerate(conversation_history, 1):
                print(f"\nTurn {i}:")
                print(f"  You: {q}")
                print(f"  Bot: {a}")
        continue

    response = chat_with_memory(user_input)
    print(f"Bot: {response}")
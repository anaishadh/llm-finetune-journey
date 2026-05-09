# ============================================================
# Stage 6: NovaBrew RAG Inference
#
# Flow:
# User question
#   → convert to vector
#   → search ChromaDB for relevant docs
#   → send docs + question to LLM
#   → get grounded answer
# ============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb

# ============================================================
# STEP 1: LOAD EVERYTHING
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

print("=" * 50)
print("Loading NovaBrew RAG System")
print("=" * 50)

# Load embedding model for search
print("\n[1/3] Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load ChromaDB
print("[2/3] Loading vector database...")
client = chromadb.PersistentClient(path="data/novabrew_vectordb")
collection = client.get_collection("novabrew_docs")
print(f"      Loaded {collection.count()} chunks")

# Load LLM
print("[3/3] Loading language model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("\nRAG system ready!")

# ============================================================
# STEP 2: RAG FUNCTION
# ============================================================

def retrieve(question, n_results=2):
    """
    Convert question to vector and find most relevant docs.
    """
    query_vector = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_vector,
        n_results=n_results
    )
    # Return the matched chunks as a list
    return results['documents'][0]

def rag_chat(question, show_context=False):
    """
    Full RAG pipeline:
    1. Retrieve relevant docs
    2. Build prompt with context
    3. Generate answer
    """
    # Step 1: Retrieve
    relevant_docs = retrieve(question)
    context = "\n\n".join(relevant_docs)
    
    if show_context:
        print(f"\n  [Retrieved context]:")
        for i, doc in enumerate(relevant_docs):
            print(f"  {i+1}. {doc[:100]}...")
    
    # Step 2: Build prompt with context injected
    # This is the key difference from fine-tuning:
    # we GIVE the model the answer, it just formats it
    prompt = f"""<|im_start|>system
You are a helpful NovaBrew customer support agent.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say you don't have that information
and direct them to support@novabrew.com.

Context:
{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

    # Step 3: Generate
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()

# ============================================================
# STEP 3: COMPARE RAG VS BASE MODEL
# ============================================================

print("\n" + "=" * 50)
print("TEST: RAG vs Base Model (no fine-tuning)")
print("=" * 50)

test_questions = [
    "What do the LED colors mean on my NovaBrew?",
    "How do I get a refund?",
    "My NovaBrew is making a strange noise.",
]

for question in test_questions:
    print(f"\nQuestion: {question}")
    print("-" * 40)
    response = rag_chat(question, show_context=True)
    print(f"RAG Answer: {response}")
    print()

# ============================================================
# STEP 4: INTERACTIVE CHAT
# ============================================================

print("=" * 50)
print("NovaBrew RAG Support Bot")
print("Type 'context' before your question to see retrieved docs")
print("Type 'quit' to exit")
print("=" * 50)

while True:
    user_input = input("\nYou: ").strip()
    
    if user_input.lower() == 'quit':
        break
    
    if not user_input:
        continue
    
    # Allow user to see context if curious
    show_ctx = False
    if user_input.lower().startswith('context '):
        show_ctx = True
        user_input = user_input[8:]
    
    response = rag_chat(user_input, show_context=show_ctx)
    print(f"Bot: {response}") 

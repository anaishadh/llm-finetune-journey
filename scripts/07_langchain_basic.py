# ============================================================
# Stage 7a: Introduction to LangChain
#
# Core concept: LCEL (LangChain Expression Language)
# Everything is a chain built with the | pipe operator
#
# prompt | model | output_parser
#
# This is exactly how companies build LLM pipelines
# ============================================================
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ============================================================
# STEP 1: LOAD THE MODEL INTO LANGCHAIN
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("=" * 50)
print("Loading model into LangChain...")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Wrap model in a HuggingFace pipeline
# This is how LangChain talks to local models
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    return_full_text=False,  # only return new tokens
)

# Wrap in LangChain's interface
llm = HuggingFacePipeline(pipeline=hf_pipeline)
print("Model loaded into LangChain!\n")

# ============================================================
# STEP 2: BUILD YOUR FIRST CHAIN
#
# A chain is: prompt | model | output_parser
# The | operator connects components like unix pipes
# ============================================================

# Define a prompt template
# {variable} placeholders get filled at runtime
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful NovaBrew customer support agent.
Answer questions politely and accurately about NovaBrew products.
If you don't know, direct to support@novabrew.com."""),
    ("human", "{question}")
])

# Output parser converts model output to clean string
output_parser = StrOutputParser()

# Build the chain with | operator
# This is LCEL - LangChain Expression Language
chain = prompt | llm | output_parser

print("Chain built: prompt | llm | output_parser")
print("=" * 50)

# ============================================================
# STEP 3: INVOKE THE CHAIN
# ============================================================

print("\nTesting the chain...\n")

questions = [
    "What is NovaBrew?",
    "How do I clean my machine?",
    "My app won't connect.",
]

for question in questions:
    print(f"Q: {question}")
    response = chain.invoke({"question": question})
    print(f"A: {response}")
    print()

# ============================================================
# STEP 4: BATCH PROCESSING
# LangChain can process multiple inputs at once
# This is how companies handle high traffic efficiently
# ============================================================

print("=" * 50)
print("Testing batch processing...")
print("=" * 50)

batch_questions = [
    {"question": "How do I schedule a brew?"},
    {"question": "What pods are compatible?"},
]

responses = chain.batch(batch_questions)
for q, r in zip(batch_questions, responses):
    print(f"\nQ: {q['question']}")
    print(f"A: {r}")

print("\n" + "=" * 50)
print("Stage 7a complete!")
print("You just built your first LangChain pipeline!")
print("=" * 50) 

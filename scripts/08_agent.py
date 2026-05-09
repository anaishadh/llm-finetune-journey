# ============================================================
# Stage 8: NovaBrew AI Agent with LangGraph
#
# LangGraph models the agent as a graph:
# - Nodes = actions (think, use tool)
# - Edges = transitions between actions
# - State = everything the agent knows so far
#
# This is the modern industry standard for AI agents
# ============================================================

import torch
import json
from datetime import datetime, date
from typing import Annotated, TypedDict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from sentence_transformers import SentenceTransformer
import chromadb

# ============================================================
# STEP 1: LOAD MODEL AND RAG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("=" * 50)
print("Loading NovaBrew AI Agent (LangGraph)")
print("=" * 50)

print("\n[1/2] Loading language model...")
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
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True,
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

print("[2/2] Loading RAG system...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="data/novabrew_vectordb")
collection = client.get_collection("novabrew_docs")
print(f"      Loaded {collection.count()} chunks")

# ============================================================
# STEP 2: DEFINE TOOLS
# ============================================================

@tool
def search_knowledge_base(query: str) -> str:
    """Search NovaBrew documentation for product info,
    troubleshooting, cleaning, warranty, WiFi setup, and pods."""
    query_vector = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_vector, n_results=2)
    return "\n\n".join(results['documents'][0])

@tool
def check_order_status(order_number: str) -> str:
    """Check the status of a NovaBrew order by order number."""
    fake_orders = {
        "NB-1001": {"status": "Delivered",   "item": "NovaBrew Pro",            "purchase_date": "2026-04-10"},
        "NB-1002": {"status": "In Transit",  "item": "NovaBrew Starter Kit",    "purchase_date": "2026-05-08"},
        "NB-1003": {"status": "Processing",  "item": "NovaBrew Descaling x3",   "purchase_date": "2026-05-09"},
    }
    order = fake_orders.get(order_number.upper())
    if order:
        return f"Order {order_number}: {order['item']} — {order['status']} (purchased {order['purchase_date']})"
    return f"Order {order_number} not found. Please verify the order number."

@tool
def check_warranty_status(purchase_date: str) -> str:
    """Check if a NovaBrew product is under warranty. Input: purchase date as YYYY-MM-DD."""
    try:
        purchased = datetime.strptime(purchase_date, "%Y-%m-%d").date()
        days_owned = (date.today() - purchased).days
        warranty_days = 365 * 2
        if days_owned <= warranty_days:
            return f"WARRANTY ACTIVE — {warranty_days - days_owned} days remaining."
        return f"WARRANTY EXPIRED — expired {days_owned - warranty_days} days ago."
    except ValueError:
        return "Invalid date. Use YYYY-MM-DD format."

@tool
def escalate_to_human(reason: str) -> str:
    """Escalate to a human agent when customer is frustrated,
    issue is unresolvable, or customer asks for a human."""
    ticket_id = f"TICKET-{abs(hash(reason)) % 10000:04d}"
    return f"Ticket {ticket_id} created. A human agent will respond within 24 hours."

tools = [search_knowledge_base, check_order_status, check_warranty_status, escalate_to_human]

# ============================================================
# STEP 3: DEFINE GRAPH STATE
#
# State is what gets passed between nodes in the graph.
# Every node reads from state and writes back to state.
# add_messages is a reducer — it appends to the list
# instead of replacing it, building conversation history.
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# ============================================================
# STEP 4: DEFINE NODES
#
# Node 1: agent — LLM decides what to do
# Node 2: tools — executes the chosen tool
# ============================================================

system_message = SystemMessage(content="""You are a helpful NovaBrew customer support agent.
You have access to tools to help customers. Use them when needed.
Always search the knowledge base first for product questions.
Be concise and helpful.""")

def agent_node(state: AgentState):
    """
    The brain of the agent.
    Takes current state (conversation so far),
    asks the LLM what to do next,
    returns either a tool call or a final answer.
    """
    messages = [system_message] + state["messages"]
    
    # Format messages as plain text for local model
    conversation = ""
    for msg in messages:
        if isinstance(msg, SystemMessage):
            conversation += f"System: {msg.content}\n"
        elif isinstance(msg, HumanMessage):
            conversation += f"Customer: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            conversation += f"Agent: {msg.content}\n"
        elif isinstance(msg, ToolMessage):
            conversation += f"Tool Result: {msg.content}\n"

    # Build tool descriptions
    tool_descriptions = "\n".join([
        f"- {t.name}: {t.description}" for t in tools
    ])

    prompt = f"""{conversation}
Available tools:
{tool_descriptions}

To use a tool, respond EXACTLY like this:
USE_TOOL: tool_name
INPUT: your input here

If you have enough information to answer without a tool, respond with:
FINAL: your answer here

Agent:"""

    response = llm.invoke(prompt)
    response = response.strip()

    # Parse response
    if response.startswith("USE_TOOL:"):
        lines = response.split("\n")
        tool_name = lines[0].replace("USE_TOOL:", "").strip()
        tool_input = lines[1].replace("INPUT:", "").strip() if len(lines) > 1 else ""
        
        # Format as tool call message
        ai_message = AIMessage(
            content=response,
            additional_kwargs={"tool_name": tool_name, "tool_input": tool_input}
        )
    else:
        # Clean up final answer
        final = response.replace("FINAL:", "").strip()
        ai_message = AIMessage(content=final)

    return {"messages": [ai_message]}

def should_continue(state: AgentState):
    """
    Router — decides which node to go to next.
    If last message has a tool call → go to tools node
    Otherwise → end (return answer to user)
    """
    last_message = state["messages"][-1]
    
    if (isinstance(last_message, AIMessage) and 
        "USE_TOOL:" in last_message.content):
        return "tools"
    return END

def tool_node(state: AgentState):
    """
    Executes the tool the agent chose.
    Reads tool name and input from last message,
    runs the tool, adds result to state.
    """
    last_message = state["messages"][-1]
    
    tool_name = last_message.additional_kwargs.get("tool_name", "")
    tool_input = last_message.additional_kwargs.get("tool_input", "")
    
    # Find and run the tool
    tool_map = {t.name: t for t in tools}
    
    if tool_name in tool_map:
        result = tool_map[tool_name].invoke(tool_input)
    else:
        result = f"Tool '{tool_name}' not found."
    
    tool_message = ToolMessage(
        content=str(result),
        tool_call_id="local_tool_call"
    )
    return {"messages": [tool_message]}

# ============================================================
# STEP 5: BUILD THE GRAPH
#
# This is where LangGraph shines — you define the flow
# visually as nodes and edges.
# ============================================================

graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "agent")          # always start at agent

graph_builder.add_conditional_edges(            # after agent:
    "agent",                                     # from agent node
    should_continue,                             # run this function
    {                                            # map return values to nodes
        "tools": "tools",                        # "tools" → go to tools node
        END: END                                 # END → finish
    }
)

graph_builder.add_edge("tools", "agent")        # after tools → back to agent

# Compile the graph
agent_graph = graph_builder.compile()

print("\nAgent graph built!")
print("Flow: START → agent → (tools → agent)* → END")

# ============================================================
# STEP 6: RUN THE AGENT
# ============================================================

def run_agent(question: str, verbose: bool = True):
    """Run the agent on a question and return final answer"""
    if verbose:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print("="*50)

    result = agent_graph.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # Get final answer from last message
    final_answer = result["messages"][-1].content

    if verbose:
        # Show the agent's thinking steps
        print("\nAgent reasoning:")
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and "USE_TOOL:" in msg.content:
                lines = msg.content.split("\n")
                tool_used = lines[0].replace("USE_TOOL:", "").strip()
                print(f"  → Used tool: {tool_used}")
            elif isinstance(msg, ToolMessage):
                print(f"  → Tool returned: {msg.content[:80]}...")

        print(f"\nFinal Answer: {final_answer}")

    return final_answer

# ============================================================
# STEP 7: TEST
# ============================================================

print("\n" + "=" * 50)
print("RUNNING AGENT TESTS")
print("=" * 50)

tests = [
    "How do I clean my NovaBrew?",
    "Check my order NB-1002 please.",
    "I bought my NovaBrew on 2025-01-15, is it under warranty?",
    "My machine won't turn on and I bought it on 2024-06-01. Am I covered?",
]

for question in tests:
    run_agent(question)

# ============================================================
# STEP 8: INTERACTIVE MODE
# ============================================================

print("\n" + "=" * 50)
print("NovaBrew AI Agent — Interactive Mode")
print("Type 'quit' to exit")
print("=" * 50)

while True:
    user_input = input("\nYou: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "quit":
        break
    run_agent(user_input)
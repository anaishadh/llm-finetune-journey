# ============================================================
# Stage 6: Build NovaBrew RAG System
# 
# What this script does:
# 1. Reads NovaBrew docs
# 2. Splits them into chunks
# 3. Converts chunks to vectors (embeddings)
# 4. Stores vectors in ChromaDB
# ============================================================

from sentence_transformers import SentenceTransformer
import chromadb
import os

# ============================================================
# STEP 1: LOAD AND CHUNK THE DOCS
# ============================================================

def load_and_chunk_docs(filepath):
    """
    Split the docs file into individual chunks.
    Each paragraph becomes one chunk.
    Smaller chunks = more precise retrieval.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by double newline (each section is a chunk)
    chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
    
    print(f"Loaded {len(chunks)} chunks from docs")
    for i, chunk in enumerate(chunks):
        # Print first line of each chunk
        print(f"  Chunk {i+1}: {chunk[:60]}...")
    
    return chunks

# ============================================================
# STEP 2: CREATE EMBEDDINGS
# ============================================================

def load_embedding_model():
    """
    Sentence transformers convert text to vectors.
    'all-MiniLM-L6-v2' is small, fast, and good enough for our use case.
    Companies use larger models for production.
    """
    print("\nLoading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded!")
    return model

# ============================================================
# STEP 3: STORE IN VECTOR DATABASE
# ============================================================

def build_vector_db(chunks, embedding_model):
    """
    ChromaDB stores our vectors locally.
    No server needed — just files on disk.
    """
    print("\nBuilding vector database...")
    
    # Initialize ChromaDB - saves to disk
    client = chromadb.PersistentClient(path="data/novabrew_vectordb")
    
    # Delete existing collection if rebuilding
    try:
        client.delete_collection("novabrew_docs")
        print("Cleared existing collection")
    except:
        pass
    
    # Create fresh collection
    collection = client.create_collection(
        name="novabrew_docs",
        metadata={"hnsw:space": "cosine"}  # cosine similarity for text
    )
    
    # Convert chunks to vectors
    print("Converting chunks to vectors...")
    embeddings = embedding_model.encode(chunks).tolist()
    
    # Store in ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    print(f"Stored {len(chunks)} chunks in vector database")
    return collection

# ============================================================
# STEP 4: TEST THE SEARCH
# ============================================================

def test_search(collection, embedding_model, query):
    """
    Test that our vector search works correctly.
    """
    query_vector = embedding_model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_vector,
        n_results=2  # get top 2 most relevant chunks
    )
    
    print(f"\nQuery: '{query}'")
    print("Top matching chunks:")
    for i, doc in enumerate(results['documents'][0]):
        print(f"\n  Result {i+1}:")
        print(f"  {doc[:150]}...")

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 50)
    print("Building NovaBrew RAG Knowledge Base")
    print("=" * 50)
    
    # Load docs
    chunks = load_and_chunk_docs('data/novabrew_docs.txt')
    
    # Load embedding model
    embedding_model = load_embedding_model()
    
    # Build vector DB
    collection = build_vector_db(chunks, embedding_model)
    
    # Test searches
    print("\n" + "=" * 50)
    print("Testing vector search...")
    print("=" * 50)
    
    test_search(collection, embedding_model, "my machine won't turn on")
    test_search(collection, embedding_model, "how do I get a refund")
    test_search(collection, embedding_model, "what do the LED colors mean")
    
    print("\n" + "=" * 50)
    print("RAG knowledge base built successfully!")
    print("Run 06_rag_inference.py to chat with RAG!")
    print("=" * 50)

if __name__ == "__main__":
    main() 

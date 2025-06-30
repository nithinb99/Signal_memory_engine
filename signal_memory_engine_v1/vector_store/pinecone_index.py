# vector_store/pinecone_index.py

import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ─── 1. Environment & Config ─────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME       = os.getenv("PINECONE_INDEX", "signal-engine")

if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY in environment variables")

# ─── 2. Determine the correct embed dim ───────────────────
_model    = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = _model.get_sentence_embedding_dimension()  # → 384

# ─── 3. Init Pinecone client ──────────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# ─── 4. Ensure Index Exists with Correct Dim ──────────────
existing = pc.list_indexes().names()
if INDEX_NAME in existing:
    # Check its dimension
    desc = pc.describe_index(name=INDEX_NAME)
    current_dim = desc.dimension
    if current_dim != EMBED_DIM:
        # Delete & recreate if dimensions mismatch
        print(f"Index '{INDEX_NAME}' exists with dim={current_dim}, expected {EMBED_DIM}. Recreating.")
        pc.delete_index(name=INDEX_NAME)
        existing.remove(INDEX_NAME)
# Create if still missing
if INDEX_NAME not in existing:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    print(f"Created index '{INDEX_NAME}' with dimension {EMBED_DIM}")

# ─── 5. Expose the live index client ──────────────────────
index = pc.Index(INDEX_NAME)
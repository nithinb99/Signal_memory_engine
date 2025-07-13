# memory_router.py

import os
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from vector_store.embeddings import get_embedder
from vector_store.pinecone_index import init_pinecone_index

router = APIRouter()

# ── 0) Load & validate env ─────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

if not (PINECONE_API_KEY and PINECONE_INDEX and OPENAI_API_KEY):
    raise RuntimeError("Set PINECONE_API_KEY, PINECONE_INDEX and OPENAI_API_KEY in .env")

# ── 1) Init Pinecone once ────────────────────────────────────
pinecone_index = init_pinecone_index(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV,
    index_name=PINECONE_INDEX,
    dimension=384,
    metric="cosine",
)

# ── 2) Init embedder once ────────────────────────────────────
embedder = get_embedder(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-ada-002",
)

class MemoryMatch(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]

@router.post("/memory/search", response_model=List[MemoryMatch])
def search_memory(
    q: str = Query(..., description="Your natural-language query"),
    top_k: int = Query(3, ge=1, le=10, description="How many results to return"),
):
    # 1) Embed the query
    try:
        q_vec = embedder.embed_query(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    # 2) Query Pinecone
    try:
        resp = pinecone_index.query(
            vector=q_vec,
            top_k=top_k,
            include_metadata=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query error: {e}")

    # 3) Normalize and return
    matches = []
    for m in getattr(resp, "matches", resp.get("matches", [])):
        matches.append(MemoryMatch(
            id=m.id if hasattr(m, "id") else m["id"],
            score=m.score if hasattr(m, "score") else m["score"],
            metadata=(m.metadata if hasattr(m, "metadata") else m["metadata"])
        ))
    return matches
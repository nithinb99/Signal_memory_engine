# vector_store_router.py

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import os

from vector_store.pinecone_index import init_pinecone_index
from vector_store.embeddings import get_embedder

router = APIRouter()

# ── 0) Load and validate env ─────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

if not (PINECONE_API_KEY and PINECONE_INDEX and OPENAI_API_KEY):
    raise RuntimeError("Set PINECONE_API_KEY, PINECONE_INDEX and OPENAI_API_KEY in .env")

# ── 1) Initialize Pinecone index ──────────────────────────────
#    (returns a pinecone.Index instance)
pinecone_index = init_pinecone_index(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV,
    index_name=PINECONE_INDEX,
    dimension=384,       # match your index
    metric="cosine",
)

# ── 2) Prepare embedder ───────────────────────────────────────
embedder = get_embedder(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-ada-002",
)

class MemoryMatch(BaseModel):
    id: str
    score: float
    agent: Optional[str]
    text: str
    tags: Optional[List[str]]

@router.get("/query", response_model=List[MemoryMatch])
def query_memory(
    q: str = Query(..., description="Your natural-language query"),
    top_k: int = Query(3, ge=1, le=10, description="Number of results to return")
):
    # 1) Embed the query
    try:
        q_vec = embedder.embed_query(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embed error: {e}")

    # 2) Run Pinecone query
    try:
        resp = pinecone_index.query(
            vector=q_vec,
            top_k=top_k,
            include_metadata=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone error: {e}")

    # 3) Format response
    results: List[MemoryMatch] = []
    for match in getattr(resp, "matches", resp.get("matches", [])):
        meta = getattr(match, "metadata", match.get("metadata", {}))
        results.append(MemoryMatch(
            id=match.id if hasattr(match, "id") else match["id"],
            score=match.score if hasattr(match, "score") else match["score"],
            agent=meta.get("agent"),
            text=meta.get("content") or meta.get("text"),
            tags=meta.get("tags"),
        ))

    return results
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

from vector_store.pinecone_index import index
from vector_store.embeddings      import get_embedder

router = APIRouter()

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
        q_vec = get_embedder(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embed error: {e}")

    # 2) Run Pinecone query
    try:
        resp = index.query(vector=q_vec, top_k=top_k, include_metadata=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone error: {e}")

    # 3) Format response
    results = []
    for match in resp["matches"]:
        meta = match["metadata"]
        results.append(MemoryMatch(
            id=match["id"],
            score=match["score"],
            agent=meta.get("agent"),
            text=meta.get("content") or meta.get("text"),
            tags=meta.get("tags")
        ))
    return results
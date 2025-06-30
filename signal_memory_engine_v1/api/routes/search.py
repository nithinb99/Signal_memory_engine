from fastapi import APIRouter
from vector_store.embeddings       import get_embedding
from vector_store.pinecone_index   import index

router = APIRouter()

@router.post("/memory/search")
def search_memory(query: str, top_k: int = 3):
    emb     = get_embedding(query)
    results = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return {"matches": results["matches"]}
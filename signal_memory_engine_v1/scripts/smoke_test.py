from vector_store.pinecone_index import index
from vector_store.embeddings      import get_embedding

def retrieve(query: str, top_k: int = 3):
    q_vec   = get_embedding(query)
    results = index.query(vector=q_vec, top_k=top_k, include_metadata=True)
    return results["matches"]

if __name__ == "__main__":
    for match in retrieve("emotional recursion during feedback loops"):
        print(f"[{match['metadata']['agent']}] {match['metadata']['content']} (score {match['score']:.3f})")
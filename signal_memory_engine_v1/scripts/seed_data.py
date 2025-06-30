# scripts/seed_data.py

import os
from dotenv import load_dotenv
load_dotenv()

# Import Pinecone client instance and index directly
from vector_store.pinecone_index import pc, index
from ingestion.batch_loader import BatchLoader
from vector_store.embeddings import get_embedding

def upsert_seed_memories(seed_path="data/seed/seed_memories.json"):
    loader = BatchLoader(seed_path)
    memories = loader.load()

    vectors = [
        (m["id"], get_embedding(m["content"]), m)
        for m in memories
    ]
    index.upsert(vectors=vectors)
    print(f"Upserted {len(vectors)} seed memories.")

if __name__ == "__main__":
    upsert_seed_memories()
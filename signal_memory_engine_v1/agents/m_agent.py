# agents/sentinel.py
import os
from scripts.langchain_retrieval import build_qa_chain

INDEX_NAME = os.getenv("PINECONE_INDEX_SENTINEL", "sentinel-memory")
qa_sentinel, store_sentinel = build_qa_chain(
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    pinecone_env=os.getenv("PINECONE_ENV", "us-east-1"),
    index_name=INDEX_NAME,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="gpt-3.5-turbo",
)
ROLE_SENTINEL = "M™ Shadow Sentinel"
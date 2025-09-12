# agents/oria_agent.py
import os
from dotenv import load_dotenv
from scripts.langchain_retrieval import build_qa_chain

load_dotenv()

PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

INDEX_NAME = os.getenv("PINECONE_INDEX_ORIA", "oria-memory")
qa_oria, store_oria = build_qa_chain(
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    pinecone_env=PINECONE_ENV,
    index_name=INDEX_NAME,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model=OPENAI_MODEL,
    k=3,
)
ROLE_ORIA = "Oria™ HR Oracle"
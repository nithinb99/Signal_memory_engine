# agents/axis_agent.py
import os
from scripts.langchain_retrieval import build_qa_chain

# Environment-specific index name for Axis™
INDEX_NAME = os.getenv("PINECONE_INDEX_AXIS", "axis-memory")

# Build the QA chain and vectorstore for Axis™
qa_axis, store_axis = build_qa_chain(
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    pinecone_env=os.getenv("PINECONE_ENV", "us-east-1"),
    index_name=INDEX_NAME,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="gpt-3.5-turbo",
)

# Human‐readable role for Axis™
ROLE_AXIS = "Axis™ Relationship Architect"
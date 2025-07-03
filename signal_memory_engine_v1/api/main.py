#!/usr/bin/env python
# api/main.py

"""
FastAPI server wrapping your RAG pipeline via Hugging Face Inference API
for zero-local-ML, low-latency queries.
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import pinecone
import httpx

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ── 1) Load env & init Pinecone ────────────────────────────
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME       = os.getenv("PINECONE_INDEX", "signal-engine")
HF_TOKEN         = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not PINECONE_API_KEY or not HF_TOKEN:
    raise RuntimeError("Set PINECONE_API_KEY and HUGGINGFACEHUB_API_TOKEN in .env")

# Monkey-patch Pinecone for langchain-community
if not hasattr(pinecone, "__version__"):
    pinecone.__version__ = "3.0.0"
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
from pinecone.db_data.index import Index as PineconeIndexClass
pinecone.Index = PineconeIndexClass

# ── 2) HTTP client for HF inference ─────────────────────────
hf_headers = {"Authorization": f"Bearer {HF_TOKEN}"}
async_client = httpx.AsyncClient(headers=hf_headers)

# ── 3) Embedder via HF Inference ───────────────────────────
# Note: HF Inference embedding endpoint for sentence-transformers…
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
async def embed_text(text: str) -> list[float]:
    resp = await async_client.post(
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBED_MODEL}",
        json={"inputs": text},
        timeout=30.0,
    )
    resp.raise_for_status()
    # HF returns list of token vectors; average them
    vecs = resp.json()
    # if shape [tokens][dim], average:
    import statistics
    return [statistics.mean(col) for col in zip(*vecs)]

# Wrap into a LangChain-compatible embedder
embeddings = HuggingFaceEmbeddings(
    client=async_client,       # passes through to HF
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

# ── 4) Vector store & retriever ────────────────────────────
vectorstore = LC_Pinecone.from_existing_index(
    embedding=embeddings,
    index_name=INDEX_NAME,
    text_key="content",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── 5) Generation via HF Inference “text2text-generation” endpoint ───────────
# Switch to a free, instruction-tuned Flan-T5 model
GEN_MODEL = "google-t5/t5-base"

async def llm_call(prompt: str) -> str:
    resp = await async_client.post(
        # Use the explicitly-named pipeline endpoint for seq2seq
        f"https://api-inference.huggingface.co/models/{GEN_MODEL}",
        json={
            "inputs": prompt,
            "parameters": {
                "temperature": 0.1,
                "max_new_tokens": 256,
            },
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    # “text2text-generation” returns a list of strings
    return resp.json()[0]["generated_text"].strip()
# ── 6) RAG helper ───────────────────────────────────────────
async def rag_answer(query: str, k: int = 3) -> str:
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(d.page_content for d in docs[:k])
    prompt  = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{query}"
    return await llm_call(prompt)

# ── 7) FastAPI wiring ──────────────────────────────────────
app = FastAPI(title="Signal Memory RAG API")

class QueryRequest(BaseModel):
    query: str
    k: int = 3

class Chunk(BaseModel):
    content: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    chunks: list[Chunk]

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    try:
        answer = await rag_answer(req.query, req.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    docs_and_scores = vectorstore.similarity_search_with_score(req.query, k=req.k)
    chunks = [Chunk(content=d.page_content, score=s) for d, s in docs_and_scores]
    return QueryResponse(answer=answer, chunks=chunks)
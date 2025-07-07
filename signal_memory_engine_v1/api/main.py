#!/usr/bin/env python
# api/main.py

"""
FastAPI server wrapping your RAG pipeline via OpenAI + Pinecone,
with core logic factored into build_qa_chain and init_pinecone_index,
and a multi-agent endpoint for agent-to-agent handoff.
"""
from dotenv import load_dotenv
load_dotenv()
import os
import logging
from typing import Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

# pull in our refactored pipeline builders
from scripts.langchain_retrieval import build_qa_chain
from vector_store import init_pinecone_index
from vector_store.embeddings import get_embedder

# pre-built per-agent RetrievalQA chains + stores + role names
from agents.axis_agent import qa_axis,     store_axis,     ROLE_AXIS
from agents.oria_agent import qa_oria,     store_oria,     ROLE_ORIA
from agents.m_agent    import qa_sentinel, store_sentinel, ROLE_SENTINEL

# ── Configure logging ─────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── 1) Load env ───────────────────────────────────────────
logger.debug("Loading environment variables")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV",     "us-east-1")
INDEX_NAME       = os.getenv("PINECONE_INDEX",   "signal-engine")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    logger.error("Missing Pinecone or OpenAI API key")
    raise RuntimeError("Set PINECONE_API_KEY and OPENAI_API_KEY in .env")

# ── 2) Init the _default_ Pinecone index for one-off queries ──────────────────────
# (you may still need this for the /query endpoint)
logger.debug("Initializing Pinecone index via helper")
init_pinecone_index(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV,
    index_name=INDEX_NAME,
    dimension=384,      # match your existing index
    metric="cosine",
)
logger.debug("Pinecone index '%s' ready", INDEX_NAME)

# ── 3) Prepare embeddings (if needed elsewhere) ─────────────────────────────────
embeddings = get_embedder(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-ada-002",           # <-- use `model`, not `model_name`
)

# ── 4) Build your “default” QA chain + store for backwards compatibility ────────
logger.debug("Building default QA chain via build_qa_chain()")
qa, vectorstore = build_qa_chain(
    pinecone_api_key=PINECONE_API_KEY,
    pinecone_env=PINECONE_ENV,
    index_name=INDEX_NAME,
    openai_api_key=OPENAI_API_KEY,
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="gpt-3.5-turbo",
)
logger.debug("Default QA chain and vectorstore ready")

# ── 5) Flagging logic ──────────────────────────────────────
def flag_from_score(score: float) -> str:
    if score > 0.8:
        return "concern"
    if score > 0.5:
        return "drifting"
    return "stable"

SUGGESTIONS = {
    "stable":   "No action needed.",
    "drifting": "Consider sending a check-in message.",
    "concern":  "Recommend escalation or a one-on-one conversation."
}

# ── 6) FastAPI wiring ─────────────────────────────────────
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
    flag: str
    suggestion: str

# ── Single‐agent endpoint ──────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    logger.debug("Received /query request: %s", req)
    try:
        answer = qa.run(req.query)
    except Exception as e:
        logger.exception("Error during RAG run")
        raise HTTPException(status_code=500, detail=str(e))

    hits = vectorstore.similarity_search_with_score(req.query, k=req.k)
    top_score = max((s for (_, s) in hits), default=0.0)
    flag       = flag_from_score(top_score)
    suggestion = SUGGESTIONS[flag]
    chunks     = [Chunk(content=d.page_content, score=s) for d, s in hits]

    return QueryResponse(
        answer=answer,
        chunks=chunks,
        flag=flag,
        suggestion=suggestion
    )

# ── per‐agent response schema ──────────────────────────────
class AgentResponse(BaseModel):
    answer: str
    chunks: list[Chunk]
    flag: str
    suggestion: str

class MultiQueryResponse(BaseModel):
    agents: Dict[str, AgentResponse]

# def notify_human_loop(query: str, agents: list[str]) -> None:
#     """
#     Background task that sends a handoff notification to a human
#     or to another endpoint/service.
#     """
#     # e.g. POST to your escalation service:
#     import httpx
#     payload = {"query": query, "agents": agents}
#     httpx.post("https://your-escalation-endpoint.example/handoff", json=payload, timeout=5.0)

# ── 7) Multi‐agent fan‐out endpoint ────────────────────────
@app.post("/multi_query", response_model=MultiQueryResponse)
def multi_query(req: QueryRequest, bg: BackgroundTasks):
    logger.debug("Received /multi_query request: %s", req)
    out: Dict[str, AgentResponse] = {}

    for role, qa_chain, store in (
        (ROLE_AXIS,     qa_axis,     store_axis),
        (ROLE_ORIA,     qa_oria,     store_oria),
        (ROLE_SENTINEL, qa_sentinel, store_sentinel),
    ):
        ans = qa_chain.run(req.query)
        docs_and_scores = store.similarity_search_with_score(req.query, k=req.k)
        top_score = max((s for (_, s) in docs_and_scores), default=0.0)
        flag      = flag_from_score(top_score)
        suggestion = SUGGESTIONS[flag]
        chunks    = [Chunk(content=d.page_content, score=s) for d, s in docs_and_scores]

        out[role] = AgentResponse(
            answer=ans,
            chunks=chunks,
            flag=flag,
            suggestion=suggestion,
        )

    # 1) Check if any agent flagged “concern”
    high_agents = [role for role, resp in out.items() if resp.flag == "concern"]
    if high_agents:
        logger.warning("High-severity detected from %s – kicking off handoff", high_agents)
        # 2) Schedule background notification
        # bg.add_task(notify_human_loop, req.query, high_agents)

    return MultiQueryResponse(agents=out)
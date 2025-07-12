#!/usr/bin/env python
# api/main.py

"""
FastAPI server wrapping your RAG pipeline via OpenAI + Pinecone,
with core logic factored into build_qa_chain and init_pinecone_index,
and multi-agent endpoints including biometrics context and event-to-memory mapping.
Additionally injects live biometric readings directly into the query string so that RetrievalQA.run() can accept it as a single input,
and triggers a handoff when any agent flags a high-severity ("concern").
"""
from dotenv import load_dotenv
load_dotenv()
import os
import logging
from typing import Dict
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# pull in our refactored pipeline builders
from scripts.langchain_retrieval import build_qa_chain
from vector_store import init_pinecone_index
from vector_store.embeddings import get_embedder
from sensors.biometric import sample_all_signals

# import our event mapper
from coherence.commons import map_events_to_memory

# pre-built per-agent RetrievalQA chains + stores + role names
from agents.axis_agent import qa_axis, store_axis, ROLE_AXIS
from agents.oria_agent import qa_oria, store_oria, ROLE_ORIA
from agents.m_agent import qa_sentinel, store_sentinel, ROLE_SENTINEL

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

# ── 2) Init Pinecone index ─────────────────────────────────
logger.debug("Initializing Pinecone index via helper")
init_pinecone_index(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV,
    index_name=INDEX_NAME,
    dimension=384,
    metric="cosine",
)
logger.debug("Pinecone index '%s' ready", INDEX_NAME)

# ── 3) Prepare embeddings (if needed elsewhere) ─────────────
embeddings = get_embedder(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-ada-002",
)

# ── 4) Build default QA chain + vectorstore ────────────────
qa, vectorstore = build_qa_chain(
    pinecone_api_key=PINECONE_API_KEY,
    pinecone_env=PINECONE_ENV,
    index_name=INDEX_NAME,
    openai_api_key=OPENAI_API_KEY,
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="gpt-3.5-turbo",
    k=3,
)

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

# ── Hand off helper ───────────────────────────────────────
def notify_human_loop(query: str, agents: list[str]) -> None:
    """
    Send an HTTP handoff to your escalation service.
    """
    import httpx
    payload = {"query": query, "agents": agents}
    # replace with your real escalation URL
    httpx.post(
        "https://your-escalation-endpoint.example/handoff",
        json=payload,
        timeout=5.0,
    )

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

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    logger.debug("Received /query request: %s", req)

    # 1) Fetch biometrics snapshot
    signals = sample_all_signals()
    logger.debug("Biometric signals: %s", signals)

    # 2) Build combined prompt string (single input)
    system_prefix = (
        "Current biometric readings:\n"
        f"• HRV: {signals['hrv']:.1f} ms\n"
        f"• Temp: {signals['temperature']:.1f} °C\n"
        f"• Blink rate: {signals['blink_rate']:.1f} bpm\n\n"
        "Use this context to answer the question below."
    )
    combined = f"{system_prefix}\n\nQuestion: {req.query}"

    # 3) Run the chain with a single string input
    try:
        answer = qa.run(combined)
    except Exception as e:
        logger.exception("Error during RAG run")
        raise HTTPException(status_code=500, detail=str(e))

    # 4) Retrieve & map memory hits
    raw_hits      = vectorstore.similarity_search_with_score(req.query, k=req.k)
    memory_events = map_events_to_memory(raw_hits)
    top_score     = max((evt["score"] for evt in memory_events), default=0.0)
    flag          = flag_from_score(top_score)
    suggestion    = SUGGESTIONS[flag]
    chunks        = [Chunk(content=evt["content"], score=evt["score"]) for evt in memory_events]

    return QueryResponse(answer=answer, chunks=chunks, flag=flag, suggestion=suggestion)

class AgentResponse(BaseModel):
    answer: str
    chunks: list[Chunk]
    flag: str
    suggestion: str

class MultiQueryResponse(BaseModel):
    agents: Dict[str, AgentResponse]

@app.post("/multi_query", response_model=MultiQueryResponse)
def multi_query(req: QueryRequest, bg: BackgroundTasks):
    logger.debug("Received /multi_query request: %s", req)

    # sample biometrics once
    signals = sample_all_signals()
    logger.debug("Biometric signals: %s", signals)

    out: Dict[str, AgentResponse] = {}
    for role, qa_chain, store in (
        (ROLE_AXIS,     qa_axis,     store_axis),
        (ROLE_ORIA,     qa_oria,     store_oria),
        (ROLE_SENTINEL, qa_sentinel, store_sentinel),
    ):
        system_prefix = (
            "Current biometric readings:\n"
            f"• HRV: {signals['hrv']:.1f} ms\n"
            f"• Temp: {signals['temperature']:.1f} °C\n"
            f"• Blink rate: {signals['blink_rate']:.1f} bpm\n\n"
            "Use this context to answer the question below."
        )
        combined = f"{system_prefix}\n\nQuestion: {req.query}"

        try:
            ans = qa_chain.run(combined)
        except Exception as e:
            logger.exception("Agent %s QA failed", role)
            raise HTTPException(status_code=500, detail=str(e))

        raw_hits      = vectorstore.similarity_search_with_score(req.query, k=req.k)
        logger.debug("%s returned %d hits", role, len(raw_hits))
        memory_events = map_events_to_memory(raw_hits)
        top_score     = max((evt["score"] for evt in memory_events), default=0.0)
        flag          = flag_from_score(top_score)
        suggestion    = SUGGESTIONS[flag]
        chunks        = [Chunk(content=evt["content"], score=evt["score"]) for evt in memory_events]

        out[role] = AgentResponse(answer=ans, chunks=chunks, flag=flag, suggestion=suggestion)

    # If any agent is at concern, schedule handoff
    high_agents = [role for role, resp in out.items() if resp.flag == "concern"]
    if high_agents:
        logger.warning("High-severity detected from %s - kicking off handoff", high_agents)
        bg.add_task(notify_human_loop, req.query, high_agents)

    return MultiQueryResponse(agents=out)
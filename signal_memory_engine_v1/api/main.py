#!/usr/bin/env python
# api/main.py

"""
FastAPI server wrapping your RAG pipeline via OpenAI + Pinecone,
with multi-agent endpoints, biometrics context, event‐to‐memory mapping,
JSON trace logging, and human‐in‐the‐loop escalation.
"""
from dotenv import load_dotenv
load_dotenv()
import os
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import httpx

# response models
from api.models import (
    Chunk,
    QueryResponse,
    BaseAgentResponse as AgentResponse,
    MultiQueryResponse,
)

# core pipeline builders
from scripts.langchain_retrieval import build_qa_chain
from vector_store import init_pinecone_index
from vector_store.embeddings import get_embedder
from sensors.biometric import sample_all_signals
from coherence.commons import map_events_to_memory

# per-agent chains & stores
from agents.axis_agent import qa_axis,     store_axis,     ROLE_AXIS
from agents.oria_agent import qa_oria,     store_oria,     ROLE_ORIA
from agents.m_agent    import qa_sentinel, store_sentinel, ROLE_SENTINEL

# ── Logging setup ───────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── 1) Load & validate environment ─────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME       = os.getenv("PINECONE_INDEX", "signal-engine")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
if not (PINECONE_API_KEY and OPENAI_API_KEY):
    logger.error("Missing Pinecone or OpenAI API key")
    raise RuntimeError("Set PINECONE_API_KEY and OPENAI_API_KEY in .env")

# ── 2) Init Pinecone index ──────────────────────────────────
init_pinecone_index(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV,
    index_name=INDEX_NAME,
    dimension=384,
    metric="cosine",
)
logger.debug("Pinecone index '%s' ready", INDEX_NAME)

# ── 3) Prepare embedder (for any direct uses) ──────────────
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

# ── 5) Flagging logic & suggestions ────────────────────────
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

# ── 6) Trace‐logger for analytics ───────────────────────────
TRACE_LOG_FILE = "trace.log"

def trace_log(agent: str, query: str, flag: str, score: float, request_id: str) -> None:
    """
    Append a JSON record with timestamp, request_id, agent, query, flag, trust_score.
    """
    rec = {
        "timestamp":   datetime.utcnow().isoformat(),
        "request_id":  request_id,
        "agent":       agent,
        "query":       query,
        "flag":        flag,
        "trust_score": score,
    }
    with open(TRACE_LOG_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")

# ── 7) Human‐in‐the‐loop escalation helper ──────────────────
def notify_human_loop(query: str, agents: List[str]) -> None:
    payload = {"query": query, "agents": agents}
    try:
        httpx.post(
            "https://your-escalation-endpoint.example/handoff",
            json=payload,
            timeout=5.0,
        )
    except Exception as e:
        logger.error("Failed to notify human loop: %s", e)

# ── 8) FastAPI app & models ────────────────────────────────
app = FastAPI(title="Signal Memory RAG API")

class QueryRequest(BaseModel):
    query: str
    k: int = 3

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    request_id = uuid.uuid4().hex
    logger.debug("Received /query [%s]: %s", request_id, req)

    # a) sample biometrics
    signals = sample_all_signals()
    logger.debug("Biometrics [%s]: %s", request_id, signals)

    # b) build single‐string prompt
    prefix = (
        "Current biometric readings:\n"
        f"• HRV: {signals['hrv']:.1f} ms\n"
        f"• Temp: {signals['temperature']:.1f} °C\n"
        f"• Blink rate: {signals['blink_rate']:.1f} bpm\n\n"
        "Use this context to answer the question below."
    )
    full_prompt = f"{prefix}\n\nQuestion: {req.query}"

    # c) run RAG
    try:
        answer = qa.run(full_prompt)
    except Exception as e:
        logger.exception("Error during RAG run [%s]", request_id)
        raise HTTPException(status_code=500, detail=str(e))

    # d) memory hits → events → scoring
    raw_hits      = vectorstore.similarity_search_with_score(req.query, k=req.k)
    events        = map_events_to_memory(raw_hits)
    top_score     = max((e["score"] for e in events), default=0.0)
    flag          = flag_from_score(top_score)
    suggestion    = SUGGESTIONS[flag]
    chunks        = [Chunk(content=e["content"], score=e["score"]) for e in events]

    # e) tracelog & respond
    trace_log("single-agent", req.query, flag, top_score, request_id)
    return QueryResponse(
        answer=answer,
        chunks=chunks,
        flag=flag,
        suggestion=suggestion,
        trust_score=top_score,
    )

@app.post("/multi_query", response_model=MultiQueryResponse)
def multi_query(req: QueryRequest, bg: BackgroundTasks):
    request_id = uuid.uuid4().hex
    logger.debug("Received /multi_query [%s]: %s", request_id, req)

    # a) sample biometrics once
    signals = sample_all_signals()
    logger.debug("Biometrics [%s]: %s", request_id, signals)

    out: Dict[str, AgentResponse] = {}
    for role, qa_chain, store in (
        (ROLE_AXIS,     qa_axis,     store_axis),
        (ROLE_ORIA,     qa_oria,     store_oria),
        (ROLE_SENTINEL, qa_sentinel, store_sentinel),
    ):
        # per-agent prompt
        prefix = (
            "Current biometric readings:\n"
            f"• HRV: {signals['hrv']:.1f} ms\n"
            f"• Temp: {signals['temperature']:.1f} °C\n"
            f"• Blink rate: {signals['blink_rate']:.1f} bpm\n\n"
            "Use this context to answer the question below."
        )
        full_prompt = f"{prefix}\n\nQuestion: {req.query}"

        # QA with graceful fallback
        try:
            ans = qa_chain.run(full_prompt)
        except Exception as e:
            logger.error("Agent %s failed [%s]: %s", role, request_id, e)
            out[role] = AgentResponse(
                answer="(error during processing)",
                chunks=[],
                flag="stable",
                suggestion="No action (agent failed)",
                trust_score=0.0,
            )
            trace_log(role, req.query, "stable", 0.0, request_id)
            continue

        # memory hits → events → scoring
        raw_hits      = store.similarity_search_with_score(req.query, k=req.k)
        events        = map_events_to_memory(raw_hits)
        top_score     = max((e["score"] for e in events), default=0.0)
        flag          = flag_from_score(top_score)
        suggestion    = SUGGESTIONS[flag]
        chunks        = [Chunk(content=e["content"], score=e["score"]) for e in events]

        out[role] = AgentResponse(
            answer=ans,
            chunks=chunks,
            flag=flag,
            suggestion=suggestion,
            trust_score=top_score,
        )
        trace_log(role, req.query, flag, top_score, request_id)

    # b) if any concern → schedule handoff
    high_agents = [r for r, resp in out.items() if resp.flag == "concern"]
    if high_agents:
        logger.warning("Escalation [%s]: %s", request_id, high_agents)
        bg.add_task(notify_human_loop, req.query, high_agents)

    # c) return aggregated responses
    return MultiQueryResponse(agents=out)
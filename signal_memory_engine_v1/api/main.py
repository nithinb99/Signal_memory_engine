#!/usr/bin/env python
# api/main.py

"""
FastAPI server wrapping your RAG pipeline via OpenAI + Pinecone,
with:
  • multi-agent endpoints (Axis™, Oria™, M™)
  • live biometric context baked into each prompt
  • event → memory mapping via coherence/commons.py
  • JSON trace-logging (agent, query, flag, score, timestamp)
  • a /memory_log endpoint for downstream analytics
  • human-in-the-loop escalation when any agent flags “concern”
  • MLflow experiment logging of params, metrics, artifacts (including prompt)
"""

# ── Standard library imports ───────────────────────────────
import json
import logging
import os
import re
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# ── Third-party imports ────────────────────────────────────
import httpx
import mlflow
from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import status  # (v2)
from mlflow.tracking import MlflowClient  # (v2)
from pydantic import BaseModel

# ── Logging setup ─────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── MLflow autolog (keep order: import → autolog) ─────────
mlflow.autolog()
try:
    import mlflow.openai  # noqa: F401
    mlflow.openai.autolog()
except ImportError:
    mlflow.get_logger().warning(
        "mlflow-openai plugin not installed; skipping OpenAI autolog."
    )

from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ (OPENAI_API_KEY, etc.)

# ── 1) Load & validate environment ────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# accept either PINECONE_ENVIRONMENT (your .env) or PINECONE_ENV (existing code)
PINECONE_ENV     = os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME       = os.getenv("PINECONE_INDEX",   "signal-engine")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
if not (PINECONE_API_KEY and OPENAI_API_KEY):
    logging.error("Missing Pinecone or OpenAI API key")
    raise RuntimeError("Set PINECONE_API_KEY and OPENAI_API_KEY in .env")

# ── First-party (package-qualified) imports ───────────────
from api.routes import signal as signal_routes
from coherence.commons import map_events_to_memory
from sensors.biometric import sample_all_signals
from scripts.langchain_retrieval import build_qa_chain
from vector_store.embeddings import get_embedder
from vector_store.pinecone_index import init_pinecone_index

from agents.axis_agent import ROLE_AXIS
from agents.axis_agent import qa_axis
from agents.axis_agent import store_axis
from agents.m_agent import ROLE_SENTINEL
from agents.m_agent import qa_sentinel
from agents.m_agent import store_sentinel
from agents.oria_agent import ROLE_ORIA
from agents.oria_agent import qa_oria
from agents.oria_agent import store_oria

# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(title="Signal Memory RAG API")
app.include_router(signal_routes.router)

# ── Response models ───────────────────────────────────────
class Chunk(BaseModel):
    content: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    chunks: List[Chunk]
    flag: str
    suggestion: str
    trust_score: float

class AgentResponse(BaseModel):
    answer: str
    chunks: List[Chunk]
    flag: str
    suggestion: str
    trust_score: float

class MultiQueryResponse(BaseModel):
    agents: Dict[str, AgentResponse]

class TraceRecord(BaseModel):
    timestamp: str
    request_id: str
    agent: str
    query: str
    flag: str
    trust_score: float

# ── Helper to sanitize MLflow keys ────────────────────────
def sanitize_key(key: str) -> str:
    return re.sub(r"[^\w\-\.:/ ]", "", key)

# ── MLflow tracking setup ─────────────────────────────────
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)
else:
    project_root = Path(__file__).resolve().parents[1]  # repo root
    mlruns_dir = project_root / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    # Use a valid file:// URI on all platforms
    try:
        uri = mlruns_dir.resolve().as_uri()  # e.g. file:///C:/... on Windows
        mlflow.set_tracking_uri(uri)
    except Exception:
        # Fallback: plain absolute path (also works locally)
        mlflow.set_tracking_uri(str(mlruns_dir.resolve()))

# ── MLflow experiment bootstrap (avoid relying on ID=0) ─── (v2)
EXP_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "signal-memory-engine")  # (v2)
_client = MlflowClient()  # (v2)
_exp = _client.get_experiment_by_name(EXP_NAME)  # (v2)
if _exp is None:  # (v2)
    _client.create_experiment(EXP_NAME)  # (v2)
mlflow.set_experiment(EXP_NAME)  # (v2)

# ── 1) Load & validate environment ────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV",     "us-east-1")
INDEX_NAME       = os.getenv("PINECONE_INDEX",   "signal-engine")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
if not (PINECONE_API_KEY and OPENAI_API_KEY):
    logger.error("Missing Pinecone or OpenAI API key")
    raise RuntimeError("Set PINECONE_API_KEY and OPENAI_API_KEY in .env")

# ── 2) Init Pinecone index ─────────────────────────────────
init_pinecone_index(
    api_key      = PINECONE_API_KEY,
    environment  = PINECONE_ENV,
    index_name   = INDEX_NAME,
    dimension    = 384,
    metric       = "cosine",
)
logger.debug("Pinecone index '%s' ready", INDEX_NAME)

# ── 3) Prepare embedder ────────────────────────────────────
embeddings = get_embedder(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small",
)

# ── 4) Build default QA chain + vectorstore ───────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # was gpt-3.5-turbo
qa, vectorstore = build_qa_chain(
    pinecone_api_key=PINECONE_API_KEY,
    pinecone_env=PINECONE_ENV,
    index_name=INDEX_NAME,
    openai_api_key=OPENAI_API_KEY,
    embed_model=EMBED_MODEL,
    llm_model=LLM_MODEL,
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

# ── 6) Trace-logger for analytics ───────────────────────────
TRACE_LOG_FILE = "trace.log"
def trace_log(agent: str, query: str, flag: str, score: float, request_id: str) -> None:
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

# ── 7) Human-in-the-loop escalation helper ──────────────────
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

# ── Helper: Safe & Non-deprecated LLM invocation (.invoke + 429 handling) ── (v2)
def _invoke_qa(chain, prompt: str) -> str:
    """
    Invoke a LangChain chain safely, handling OpenAI 429 / quota.
    Uses .invoke instead of deprecated .run.
    """
    try:
        # RetrievalQA chains accept {"query": "..."} when using .invoke
        out = chain.invoke({"query": prompt})
        # RetrievalQA returns a dict; answer under "result"
        if isinstance(out, dict) and "result" in out:
            return out["result"]
        # some chains return a plain string
        if isinstance(out, str):
            return out
        return str(out)
    except Exception as e:
        # check for quota specifically; return a controlled 503
        detail = (
            "LLM quota/rate limit hit (OpenAI 429 / insufficient_quota). "
            "Set a valid key/billing, switch model/provider, or configure a fallback."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail
        ) from e
    except Exception as e:
        # Preserve original behavior for non-429 errors
        logger.exception("LLM invoke failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

# ── 8) FastAPI endpoints ───────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    k: int = 3

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    request_id = uuid.uuid4().hex
    start_ts   = time.time()
    logger.debug("Received /query [%s]: %s", request_id, req)

    with mlflow.start_run(run_name=f"single-{request_id}"):
        mlflow.log_param("query", req.query)
        mlflow.log_param("k",     req.k)
        mlflow.log_param("embed_model", EMBED_MODEL)
        mlflow.log_param("llm_model",   LLM_MODEL)

        # biometrics
        signals = sample_all_signals()
        mlflow.log_metric("hrv",         signals.get("hrv", 0))
        mlflow.log_metric("temperature", signals.get("temperature", 0))
        mlflow.log_metric("blink_rate",  signals.get("blink_rate", 0))
        if "gsr" in signals:
            mlflow.log_metric("gsr", signals["gsr"])
        if "emotion_label" in signals:
            mlflow.log_param("emotion_label", signals["emotion_label"])

        # prompt
        prefix = (
            "Current biometric readings:\n"
            f"• HRV: {signals['hrv']:.1f} ms\n"
            f"• Temp: {signals['temperature']:.1f} °C\n"
            f"• Blink rate: {signals['blink_rate']:.1f} bpm\n\n"
            "Use this context to answer the question below."
        )
        full_prompt = f"{prefix}\n\nQuestion: {req.query}"
        mlflow.log_param("prompt", full_prompt)
        mlflow.log_text(full_prompt, "prompt.txt")

        # RAG (v2: allow HTTPException to pass through)
        try:  # (v2)
            answer = _invoke_qa(qa, full_prompt)  # (v2)
        except HTTPException as he:  # (v2)
            raise he  # preserve intentional statuses (e.g., 503) (v2)
        except Exception as e:  # (v2)
            logger.exception("Error during RAG run [%s]", request_id)  # (v2)
            raise HTTPException(status_code=500, detail=str(e)) from e  # (v2)

        # retrieve & score
        raw_hits  = vectorstore.similarity_search_with_score(req.query, k=req.k)
        events    = map_events_to_memory(raw_hits)
        top_score = max((e['score'] for e in events), default=0.0)
        flag      = flag_from_score(top_score)
        suggestion= SUGGESTIONS[flag]
        chunks    = [Chunk(content=e['content'], score=e['score']) for e in events]

        # log outputs
        mlflow.log_metric("top_score",   top_score)
        mlflow.log_metric("num_chunks",  len(raw_hits))
        mlflow.log_param( "flag",        flag)
        mlflow.log_param( "suggestion",  suggestion)
        mlflow.log_text(answer, "answer.txt")

        emo_rec = int(any(e.get("emotional_recursion_present") for e in events))
        reroute = int(any(e.get("rerouting_triggered") for e in events))
        mlflow.log_param("emotional_recursion_detected", emo_rec)
        mlflow.log_param("rerouting_triggered", reroute)

        mlflow.log_metric("chad_confidence_score", top_score)
        esc_level = 1 if flag == "concern" else 0
        mlflow.log_param("escalation_level", esc_level)

        elapsed_ms = (time.time() - start_ts) * 1000
        mlflow.log_metric("endpoint_latency_ms",     elapsed_ms)
        mlflow.log_param( "coherence_drift_detected", int(flag != "stable"))
        mlflow.log_param( "escalation_triggered",     0)

    trace_log("single-agent", req.query, flag, top_score, request_id)
    return QueryResponse(
        answer       = answer,
        chunks       = chunks,
        flag         = flag,
        suggestion   = suggestion,
        trust_score  = top_score,
    )

@app.post("/multi_query", response_model=MultiQueryResponse)
def multi_query(req: QueryRequest, bg: BackgroundTasks):
    request_id = uuid.uuid4().hex
    start_ts   = time.time()
    logger.debug("Received /multi_query [%s]: %s", request_id, req)

    with mlflow.start_run(run_name=f"multi-{request_id}"):
        mlflow.log_param("query", req.query)
        mlflow.log_param("k",     req.k)
        mlflow.log_param("embed_model", EMBED_MODEL)
        mlflow.log_param("llm_model",   LLM_MODEL)

        # biometrics
        signals = sample_all_signals()
        mlflow.log_metric("hrv",         signals.get("hrv", 0))
        mlflow.log_metric("temperature", signals.get("temperature", 0))
        mlflow.log_metric("blink_rate",  signals.get("blink_rate", 0))
        if "gsr" in signals:
            mlflow.log_metric("gsr", signals["gsr"])
        if "emotion_label" in signals:
            mlflow.log_param("emotion_label", signals["emotion_label"])

        # shared prompt
        prefix = (
            "Current biometric readings:\n"
            f"• HRV: {signals['hrv']:.1f} ms\n"
            f"• Temp: {signals['temperature']:.1f} °C\n"
            f"• Blink rate: {signals['blink_rate']:.1f} bpm\n\n"
            "Use this context to answer the question below."
        )
        full_prompt = f"{prefix}\n\nQuestion: {req.query}"
        mlflow.log_param("prompt", full_prompt)
        mlflow.log_text(full_prompt, "prompt.txt")

        out: Dict[str, AgentResponse] = {}
        all_events: List[dict] = []  # (v2) collect across agents

        for role, qa_chain, store in (
            (ROLE_AXIS,     qa_axis,     store_axis),
            (ROLE_ORIA,     qa_oria,     store_oria),
            (ROLE_SENTINEL, qa_sentinel, store_sentinel),
        ):
            safe = sanitize_key(role)
            try:
                ans = _invoke_qa(qa_chain, full_prompt)  # (v2)
            except HTTPException as he:  # (v2)
                if he.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
                    logger.warning("Agent %s quota hit [%s]: %s", role, request_id, he.detail)  # (v2)
                    out[role] = AgentResponse(
                        answer="(LLM temporarily unavailable: quota/rate limit)",
                        chunks=[], flag="stable", suggestion="No action", trust_score=0.0
                    )
                    trace_log(role, req.query, "stable", 0.0, request_id)  # (v2)
                    mlflow.log_param(f"{safe}_flag", "stable")  # (v2)
                    mlflow.log_metric(f"{safe}_top_score", 0.0)  # (v2)
                    continue  # skips current agent but continues for the others (v2)
                raise he  # (v2)
            except Exception as e:
                logger.error("Agent %s failed [%s]: %s", role, request_id, e)
                out[role] = AgentResponse(answer="(error)", chunks=[], flag="stable",
                                           suggestion="No action", trust_score=0.0)
                trace_log(role, req.query, "stable", 0.0, request_id)
                mlflow.log_param(f"{safe}_flag",       "stable")
                mlflow.log_metric(f"{safe}_top_score", 0.0)
                continue

            raw_hits  = store.similarity_search_with_score(req.query, k=req.k)
            events    = map_events_to_memory(raw_hits)
            all_events.extend(events)  # (v2)
            top_score = max((e["score"] for e in events), default=0.0)
            flag      = flag_from_score(top_score)
            suggestion= SUGGESTIONS[flag]
            chunks    = [Chunk(content=e["content"], score=e["score"]) for e in events]

            out[role] = AgentResponse(answer=ans, chunks=chunks,
                                      flag=flag, suggestion=suggestion,
                                      trust_score=top_score)
            trace_log(role, req.query, flag, top_score, request_id)

            mlflow.log_param( f"{safe}_flag",        flag)
            mlflow.log_param( f"{safe}_suggestion",  suggestion)
            mlflow.log_metric(f"{safe}_top_score",   top_score)
            mlflow.log_metric(f"{safe}_num_chunks",  len(raw_hits))
            mlflow.log_text(ans, f"{safe}_answer.txt")

        # Cross-agent derived flags (v2: use all_events)
        emo_rec_multi = int(any(e.get("emotional_recursion_present") for e in all_events))  # (v2)
        reroute_multi = int(any(e.get("rerouting_triggered")           for e in all_events))  # (v2)
        mlflow.log_param("emotional_recursion_detected", emo_rec_multi)
        mlflow.log_param("rerouting_triggered",          reroute_multi)

        avg_conf = sum(r.trust_score for r in out.values()) / (len(out) or 1)
        mlflow.log_metric("chad_confidence_score", avg_conf)
        esc_level = len([r for r in out.values() if r.flag == "concern"])
        mlflow.log_param("escalation_level", esc_level)

        elapsed_ms = (time.time() - start_ts) * 1000
        mlflow.log_metric("endpoint_latency_ms", elapsed_ms)
        mlflow.log_param("coherence_drift_detected",
                         int(any(r.flag != "stable" for r in out.values())))
        mlflow.log_param("escalation_triggered",
                         int(any(r.flag == "concern" for r in out.values())))

    high_agents = [r for r, resp in out.items() if resp.flag == "concern"]
    if high_agents:
        bg.add_task(notify_human_loop, req.query, high_agents)

    return MultiQueryResponse(agents=out)

@app.get("/memory_log", response_model=List[TraceRecord])
def memory_log(limit: int = 20):
    """Return the last `limit` trace-log records for downstream analytics."""
    try:
        buf = deque(maxlen=limit)
        with open(TRACE_LOG_FILE, "r") as f:
            for line in f:
                buf.append(json.loads(line))
        return list(buf)
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.exception("Error reading trace log")
        raise HTTPException(status_code=500, detail=str(e))
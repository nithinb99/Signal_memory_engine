import uuid
import time
from fastapi import APIRouter
from sensors.biometric import sample_all_signals
from api.models import QueryRequest, QueryResponse, Chunk
from api import deps
from utils.llm import invoke_chain
from utils.tracing import trace_log

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    request_id = uuid.uuid4().hex
    start_ts = time.time()

    # biometrics
    signals = sample_all_signals()

    # prompt
    prefix = (
        "Current biometric readings:\n"
        f"• HRV: {signals['hrv']:.1f} ms\n"
        f"• Temp: {signals['temperature']:.1f} °C\n"
        f"• Blink rate: {signals['blink_rate']:.1f} bpm\n\n"
        "Use this context to answer the question below."
    )
    full_prompt = f"{prefix}\n\nQuestion: {req.query}"

    # RAG (bubbles 503 on quota)
    answer = invoke_chain(deps.qa, full_prompt)

    # retrieve & score
    raw_hits = deps.vectorstore.similarity_search_with_score(req.query, k=req.k)
    events = [
        {"content": h[0].page_content, "score": float(h[1])} for h in raw_hits
    ]  # lightweight mapping; keep your map_events_to_memory if preferred
    top_score = max((e["score"] for e in events), default=0.0)
    flag = deps.flag_from_score(top_score)
    suggestion = deps.SUGGESTIONS[flag]
    chunks = [Chunk(content=e["content"], score=e["score"]) for e in events]

    trace_log("single-agent", req.query, flag, top_score, request_id)

    return QueryResponse(
        answer=answer,
        chunks=chunks,
        flag=flag,
        suggestion=suggestion,
        trust_score=top_score,
    )

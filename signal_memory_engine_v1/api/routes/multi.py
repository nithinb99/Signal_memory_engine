# ============================================================================
# api/routes/multi.py  →  POST /multi_query
# ============================================================================
import uuid
import time
from typing import Dict
from fastapi import APIRouter, BackgroundTasks
from sensors.biometric import sample_all_signals
from api.models import QueryRequest, MultiQueryResponse, AgentResponse, Chunk
from api import deps
from utils.llm import invoke_chain
from utils.tracing import trace_log
import httpx

router = APIRouter()


def notify_human_loop(query: str, agents: list[str]) -> None:
    payload = {"query": query, "agents": agents}
    try:
        httpx.post(
            "https://your-escalation-endpoint.example/handoff",
            json=payload,
            timeout=5.0,
        )
    except Exception:
        pass


@router.post("/multi_query", response_model=MultiQueryResponse)
def multi_query(req: QueryRequest, bg: BackgroundTasks):
    request_id = uuid.uuid4().hex
    start_ts = time.time()

    signals = sample_all_signals()
    prefix = (
        "Current biometric readings:\n"
        f"• HRV: {signals['hrv']:.1f} ms\n"
        f"• Temp: {signals['temperature']:.1f} °C\n"
        f"• Blink rate: {signals['blink_rate']:.1f} bpm\n\n"
        "Use this context to answer the question below."
    )
    full_prompt = f"{prefix}\n\nQuestion: {req.query}"

    out: Dict[str, AgentResponse] = {}
    all_events = []

    for role, qa_chain, store in deps.AGENTS:
        try:
            ans = invoke_chain(qa_chain, full_prompt)
        except (
            Exception
        ):  # invoke_chain already maps quota → 503; keep going for others
            out[role] = AgentResponse(
                answer="(LLM temporarily unavailable)",
                chunks=[],
                flag="stable",
                suggestion="No action",
                trust_score=0.0,
            )
            trace_log(role, req.query, "stable", 0.0, request_id)
            continue

        raw_hits = store.similarity_search_with_score(req.query, k=req.k)
        events = [
            {"content": h[0].page_content, "score": float(h[1])} for h in raw_hits
        ]
        all_events.extend(events)
        top_score = max((e["score"] for e in events), default=0.0)
        flag = deps.flag_from_score(top_score)
        suggestion = deps.SUGGESTIONS[flag]
        chunks = [Chunk(content=e["content"], score=e["score"]) for e in events]

        out[role] = AgentResponse(
            answer=ans,
            chunks=chunks,
            flag=flag,
            suggestion=suggestion,
            trust_score=top_score,
        )
        trace_log(role, req.query, flag, top_score, request_id)

    high_agents = [r for r, resp in out.items() if resp.flag == "concern"]
    if high_agents:
        bg.add_task(notify_human_loop, req.query, high_agents)

    return MultiQueryResponse(agents=out)

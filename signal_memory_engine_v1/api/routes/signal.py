import os
import json as _json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

# from storage.sqlite_store import init_db, insert_event, list_by_user
# from agents.router_stub import route_agent, log_routing_decision
# from utils.dashboard import send_to_dashboard
from storage.sqlite_store import init_db, insert_event, list_by_user
from agents.router_stub import route_agent, log_routing_decision
from utils.dashboard import send_to_dashboard

DB_PATH = os.getenv("SME_DB_PATH", "./data/signal.db")
init_db(DB_PATH)

router = APIRouter()

# ---------- Models (Pydantic v2) ----------
class SignalEventIn(BaseModel):
    user_id: str
    user_query: str
    signal_type: str
    drift_score: float = Field(ge=0, le=1)
    emotional_tone: Optional[float] = Field(default=0.0, ge=0, le=1)
    agent_id: Optional[str] = None  # optional override; if absent we route
    payload: Optional[dict] = None
    relationship_context: Optional[str] = None
    diagnostic_notes: Optional[str] = None


class SignalEventOut(SignalEventIn):
    id: int
    timestamp: str
    escalate_flag: int


# ---------- Endpoints ----------
@router.post("/signal", response_model=SignalEventOut)
def log_signal(event: SignalEventIn):
    """
    Logs a signal. If agent_id is not provided, the router decides.
    Persists the event to SQLite, appends JSONL, and prints a dashboard stub.
    """
    # Decide agent
    if event.agent_id:
        selected_agent = event.agent_id
        reason = "Agent override"
    else:
        decision = route_agent(
            event.user_query,
            event.emotional_tone or 0.0,
            event.signal_type,
            event.drift_score,
        )
        selected_agent = decision["selected_agent"]
        reason = decision["reason"]

    # Compute escalate flag + timestamp
    escalate = 1 if float(event.drift_score) > 0.5 else 0
    ts = datetime.utcnow().isoformat()

    # Build row to persist
    stored = event.model_dump()
    stored["timestamp"] = ts
    stored["escalate_flag"] = escalate
    stored["agent_id"] = selected_agent

    # Persist to DB
    event_id = insert_event(DB_PATH, stored)
    stored["id"] = event_id

    # Keep JSONL routing log (handy during dev)
    log_routing_decision(
        {
            "selected_agent": selected_agent,
            "reason": reason,
            "user_id": event.user_id,
            "signal_type": event.signal_type,
            "drift_score": event.drift_score,
        }
    )

    # Future-proof hook (console for now)
    send_to_dashboard({**stored})

    return stored


@router.get("/drift/{user_id}", response_model=list[SignalEventOut])
def get_drift(user_id: str, limit: int = Query(10, ge=1, le=100)):
    """
    Returns most-recent events for a user. Normalizes `payload` to a dict.
    """
    rows = list_by_user(DB_PATH, user_id=user_id, limit=limit)

    # Normalize payload back to a dict (DB stores it as TEXT)
    for r in rows:
        if isinstance(r.get("payload"), str):
            try:
                r["payload"] = _json.loads(r["payload"])
            except Exception:
                r["payload"] = None

    return rows
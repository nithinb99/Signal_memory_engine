# ============================================================================
# api/models.py (consolidated models)
# ============================================================================
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    content: str
    score: float


class QueryRequest(BaseModel):
    query: str
    k: int = 3


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


class MemoryMatch(BaseModel):
    id: str
    score: float
    text: Optional[str] = None
    agent: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class SignalEventIn(BaseModel):
    user_id: str
    user_query: str
    signal_type: str
    drift_score: float = Field(ge=0, le=1)
    emotional_tone: Optional[float] = Field(default=0.0, ge=0, le=1)
    agent_id: Optional[str] = None
    payload: Optional[dict] = None
    relationship_context: Optional[str] = None
    diagnostic_notes: Optional[str] = None


class SignalEventOut(SignalEventIn):
    id: int
    timestamp: str
    escalate_flag: int

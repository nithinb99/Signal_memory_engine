# api/models.py
from pydantic import BaseModel
from typing import List, Optional

class Chunk(BaseModel):
    content: str
    score: float

class BaseAgentResponse(BaseModel):
    answer: str
    chunks: List[Chunk]
    flag: str
    suggestion: str
    trust_score: float  # new field

class MultiQueryResponse(BaseModel):
    agents: dict[str, BaseAgentResponse]

class QueryResponse(BaseModel):
    answer: str
    chunks: List[Chunk]
    flag: str
    suggestion: str
    trust_score: float  # new field for single‐agent
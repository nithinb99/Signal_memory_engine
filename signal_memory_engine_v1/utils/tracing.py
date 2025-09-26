# utils/tracing.py
import json, logging
from datetime import datetime

TRACE_LOG_FILE = "trace.log"
_log = logging.getLogger(__name__)

def trace_log(agent: str, query: str, flag: str, score: float, request_id: str) -> None:
    try:
        rec = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "agent": agent,
            "query": query,
            "flag": flag,
            "trust_score": score,
        }
        with open(TRACE_LOG_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        _log.error("trace_log write failed: %s", e)

def read_trace_tail(limit: int = 20):
    from collections import deque
    try:
        buf = deque(maxlen=limit)
        with open(TRACE_LOG_FILE, "r") as f:
            for line in f:
                buf.append(json.loads(line))
        return list(buf)
    except FileNotFoundError:
        return []

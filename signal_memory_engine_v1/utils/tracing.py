# ============================================================================
# utils/tracing.py
# ============================================================================

import json
from datetime import datetime

TRACE_LOG_FILE = "trace.log"

def trace_log(agent: str, query: str, flag: str, score: float, request_id: str) -> None:
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

def read_trace_tail(limit: int = 20):
    from collections import deque
    buf = deque(maxlen=limit)
    with open(TRACE_LOG_FILE, "r") as f:
        for line in f:
            buf.append(json.loads(line))
    return list(buf)
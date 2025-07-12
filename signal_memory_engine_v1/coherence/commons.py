# coherence/commons.py
"""
Common utilities for mapping RAG hits (Document, score) pairs into normalized event-memory structures.
"""
import hashlib
from datetime import datetime
from typing import List, Tuple, Dict, Any

from langchain.schema import Document


def generate_event_id(content: str, timestamp: str = "") -> str:
    """
    Deterministically generate a unique ID for an event based on its content and optional timestamp.
    """
    base = content + timestamp
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def normalize_timestamp(ts: Any) -> str:
    """
    Normalize a timestamp to ISO 8601 string if possible.
    Accepts datetime or ISO-like string.
    """
    if isinstance(ts, datetime):
        return ts.isoformat()
    try:
        # attempt parse string
        dt = datetime.fromisoformat(str(ts))
        return dt.isoformat()
    except Exception:
        # fallback to raw string
        return str(ts)


def map_events_to_memory(
    docs_and_scores: List[Tuple[Document, float]]
) -> List[Dict[str, Any]]:
    """
    Convert a list of (Document, score) into a list of normalized event dictionaries.

    Each event will include:
      - id: unique hash
      - content: the document page_content
      - score: the similarity score
      - timestamp: if present in document.metadata (normalized)
      - metadata: other metadata fields, if any
    """
    events: List[Dict[str, Any]] = []
    for doc, score in docs_and_scores:
        meta = doc.metadata or {}
        ts_raw = meta.get("timestamp")
        timestamp = normalize_timestamp(ts_raw) if ts_raw is not None else None

        event: Dict[str, Any] = {
            "id": generate_event_id(doc.page_content, timestamp or ""),
            "content": doc.page_content.strip(),
            "score": score,
        }
        if timestamp:
            event["timestamp"] = timestamp
        # include any other metadata
        extra_meta = {k: v for k, v in meta.items() if k != "timestamp"}
        if extra_meta:
            event["metadata"] = extra_meta

        events.append(event)
    return events

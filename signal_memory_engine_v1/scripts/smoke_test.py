#!/usr/bin/env python
import sys
import time
import argparse
from pathlib import Path

# ── make sure project root is on the import path ───────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from api.main import app  # now it will find api/main.py

# ── in-process client ─────────────────────────────────────────
client = TestClient(app)

def test_endpoint(name, method, path, payload=None, params=None):
    start = time.time()
    try:
        if method.lower() == "get":
            resp = client.get(path, params=params)
        else:
            resp = client.post(path, json=payload)
        resp.raise_for_status()
        latency = (time.time() - start) * 1000
    except Exception as e:
        print(f"[{name}] ERROR: {e}")
        return None, None, None
    print(f"[{name}] {resp.status_code} in {latency:.0f}ms")
    return resp.json(), resp.status_code, latency

def validate_query_response(data):
    assert isinstance(data, dict), "Response must be an object"
    assert "answer" in data
    assert isinstance(data["chunks"], list)
    assert data["flag"] in {"stable","drifting","concern"}
    assert isinstance(data["trust_score"], (int,float))
    return True

def validate_multi_response(data):
    assert "agents" in data and isinstance(data["agents"], dict)
    for role, resp in data["agents"].items():
        validate_query_response(resp)
    return True

def validate_memory_log(data):
    assert isinstance(data, list)
    for rec in data:
        assert "timestamp" in rec and "agent" in rec and "trust_score" in rec
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-k",     type=int, default=3)
    p.add_argument("--limit", type=int, default=5)
    args = p.parse_args()

    payload = {"query": "What is emotional recursion?", "k": args.k}

    # 1) SINGLE-QUERY
    data, _, _ = test_endpoint("SINGLE-QUERY", "post", "/query", payload=payload)
    if data:
        validate_query_response(data)
        print("[SINGLE-QUERY] ✅ schema OK\n")

    # 2) MULTI-QUERY
    data, _, _ = test_endpoint("MULTI-QUERY", "post", "/multi_query", payload=payload)
    if data:
        validate_multi_response(data)
        print("[MULTI-QUERY] ✅ schema OK\n")

    # 3) MEMORY-LOG
    data, _, _ = test_endpoint("MEMORY-LOG", "get", "/memory_log", params={"limit": args.limit})
    if data is not None:
        validate_memory_log(data)
        print("[MEMORY-LOG] ✅ schema OK\n")

    print("✔ All smoke tests passed!")

if __name__ == "__main__":
    main()
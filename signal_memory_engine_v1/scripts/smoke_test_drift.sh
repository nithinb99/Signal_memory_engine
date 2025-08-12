#!/usr/bin/env bash
set -euo pipefail

# cd to repo root
cd "$(dirname "$0")/.."

# default DB path if not provided
export SME_DB_PATH="${SME_DB_PATH:-./data/signal.db}"

python - <<'PY'
from agents.router_stub import route_and_log_event
from storage.sqlite_store import list_recent, init_db
import os

db = os.getenv("SME_DB_PATH", "./data/signal.db")
init_db(db)

print(route_and_log_event("u_123", "Team feels tense", 0.85, "relational", 0.1))
print(route_and_log_event("u_123", "Prod looks off", 0.2, "relational", 0.65))
print(route_and_log_event("u_999", "Is this legally allowed?", 0.3, "compliance", 0.2))

print("Recent:", list_recent(db, limit=5))
PY
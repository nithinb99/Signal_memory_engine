# tests/conftest.py

import os
import sys
from pathlib import Path

# Ensures "signal_memory_engine_v1" is importable when running tests inside this subfolder.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def pytest_sessionstart(session):
    # Tell ingest_memory.py to use its dummy embedder/index
    os.environ.setdefault("SME_TEST_MODE", "1")
    # Quiet/skip heavy libs if they get imported elsewhere
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
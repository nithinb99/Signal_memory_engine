# sensors/biometric.py
"""
Simulated biometric signal stubs for downstream agent processing.
Provides heart-rate variability (HRV), skin temperature, blink rate, and batch sampling.
"""
import random
import time
import json
from datetime import datetime
from pathlib import Path

DRIFT_FILE = Path(__file__).parent.parent / "data/drift_simulation.jsonl"
_drift_events = None

def simulate_hrv() -> float:
    """Heart-rate variability in milliseconds."""
    return random.uniform(20.0, 100.0)


def simulate_temperature() -> float:
    """Skin temperature in °C."""
    return random.uniform(35.5, 37.5)


def simulate_blink_rate() -> float:
    """Blink rate in blinks per minute."""
    return random.uniform(10.0, 30.0)

def _load_drift():
    global _drift_events
    if _drift_events is None:
        _drift_events = [json.loads(l) for l in open(DRIFT_FILE)]
    return _drift_events

def sample_all_signals():
    """
    Returns either a fresh random stub _or_ the next record from drift_simulation.jsonl.
    """
    events = _load_drift()
    # simple rotating cursor
    idx = int(datetime.utcnow().timestamp()) % len(events)
    rec = events[idx]
    return {
        "hrv": rec["hrv"],
        "temperature": rec.get("temperature", 36.5),
        "blink_rate": rec.get("blink_rate", 15.0),
        **{"flag_raised": rec.get("flag_raised"), "timestamp": rec.get("timestamp")},
    }

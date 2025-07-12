# sensors/biometric.py
"""
Simulated biometric signal stubs for downstream agent processing.
Provides heart-rate variability (HRV), skin temperature, blink rate, and batch sampling.
"""
import random
import time

def simulate_hrv() -> float:
    """Heart-rate variability in milliseconds."""
    return random.uniform(20.0, 100.0)


def simulate_temperature() -> float:
    """Skin temperature in °C."""
    return random.uniform(35.5, 37.5)


def simulate_blink_rate() -> float:
    """Blink rate in blinks per minute."""
    return random.uniform(10.0, 30.0)


def sample_all_signals() -> dict:
    """
    Return a timestamped sample of all biometric signals.
    {
        "timestamp": 1672531200.123,
        "hrv": 65.3,
        "temperature": 36.8,
        "blink_rate": 18.7
    }
    """
    ts = time.time()
    return {
        "timestamp": ts,
        "hrv": simulate_hrv(),
        "temperature": simulate_temperature(),
        "blink_rate": simulate_blink_rate(),
    }
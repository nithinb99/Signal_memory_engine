import json
import pytest
from agents.router_stub import route_agent, log_routing_decision


def test_axis_high_emotion():
    decision = route_agent("Why is my team so tense?", emotional_tone=0.9, signal_type="relational")
    assert decision["selected_agent"] == "Axis"
    assert "High emotional tone" in decision["reason"]


def test_m_compliance():
    decision = route_agent("Is this legally allowed?", emotional_tone=0.3, signal_type="compliance")
    assert decision["selected_agent"] == "M"
    assert "Compliance" in decision["reason"]


def test_oria_default():
    decision = route_agent("What’s our next meeting?", emotional_tone=0.4, signal_type="general")
    assert decision["selected_agent"] == "Oria"
    assert "Default" in decision["reason"]


def test_logging(tmp_path):
    # Use a temporary log file to avoid side effects
    log_file = tmp_path / "router_log.jsonl"
    decision = {"selected_agent": "Axis", "reason": "High emotional tone"}
    log_routing_decision(decision, logfile=str(log_file))

    # Read and verify the logged entry
    lines = log_file.read_text().splitlines()
    assert len(lines) == 1, "Expected one log entry"
    entry = json.loads(lines[-1])
    assert entry["selected_agent"] == "Axis"
    assert entry["reason"] == "High emotional tone"
    assert "timestamp" in entry, "Log entry must include a timestamp"


if __name__ == "__main__":
    pytest.main([__file__])
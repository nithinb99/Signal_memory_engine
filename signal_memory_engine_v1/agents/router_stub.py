import json
from datetime import datetime
from typing import Dict, Any

def route_agent(user_query: str, emotional_tone: float, signal_type: str) -> Dict[str, Any]:
    """
    Decide which agent should handle the incoming query based on emotional tone and signal type.

    Args:
        user_query (str): The text of the user's query.
        emotional_tone (float): A score between 0 and 1 indicating emotional intensity.
        signal_type (str): One of "biometric", "relational", or "compliance".

    Returns:
        Dict[str, Any]: A decision dict containing:
            - selected_agent (str): Name of the chosen agent.
            - reason (str): Explanation for the routing decision.
    """
    # Routing rules
    if emotional_tone > 0.7:
        return {"selected_agent": "Axis", "reason": "High emotional tone"}
    if signal_type.lower() == "compliance":
        return {"selected_agent": "M", "reason": "Compliance signal"}
    # Default case
    return {"selected_agent": "Oria", "reason": "Default routing"}


def log_routing_decision(decision: Dict[str, Any], logfile: str = "router_log.jsonl") -> None:
    """
    Append a routing decision to a JSONL log file, including a timestamp.

    Args:
        decision (Dict[str, Any]): The routing decision dict.
        logfile (str): Path to the log file.
    """
    entry = {
        "timestamp": datetime.now().astimezone().isoformat(),
        **decision
    }
    with open(logfile, 'a') as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    # Quick manual tests
    tests = [
        ("Why is my team so tense?", 0.9, "relational"),
        ("Is this legally allowed?", 0.3, "compliance"),
        ("What’s our next meeting?", 0.4, "general")
    ]
    for query, tone, sig in tests:
        decision = route_agent(query, tone, sig)
        print(query, "->", decision)
        log_routing_decision(decision)
    print("Manual tests completed. Logs written to router_log.jsonl.")
# ingestion/batch_loader.py

import json
import uuid
from typing import List, Dict


class BatchLoader:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> List[Dict]:
        with open(self.path, "r") as f:
            data = json.load(f)

        normalized = []
        for entry in data:
            content = entry.get("content", "").strip()
            if not content:
                continue  # skip empty

            # Generate a unique ID if missing
            entry_id = entry.get("id") or str(uuid.uuid4())

            normalized.append(
                {
                    "id": entry_id,
                    "content": content,
                    "tags": entry.get("tags", []),
                    "agent": entry.get("agent"),
                }
            )

        return normalized

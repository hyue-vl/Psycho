"""MemGPT-backed personalized knowledge base manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PKBManager:
    memgpt_client: object

    def load_core_memory(self, user_id: str) -> Dict[str, str]:
        return self.memgpt_client.get_core_memory(user_id)

    def update_core_memory(self, user_id: str, payload: Dict[str, str]) -> None:
        self.memgpt_client.update_core_memory(user_id, payload)

    def maybe_update_on_event(self, user_id: str, prev_belief: dict, next_belief: dict) -> None:
        if prev_belief.get("cognitive_distortion") and not next_belief.get("cognitive_distortion"):
            self.update_core_memory(
                user_id,
                {
                    "therapy_progress": "Successfully challenged core distortion",
                    "latest_emotion": next_belief.get("emotion"),
                },
            )

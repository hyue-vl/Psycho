"""
MemGPT-backed personalized knowledge base manager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PKBManager:
    """
    Wraps MemGPT to keep per-user psychological state synchronized.
    """

    memgpt_client: "MemGPTClient"
    default_template: Dict = field(
        default_factory=lambda: {
            "user_profile": "Name: TBD",
            "psychological_state": "Core Belief: TBD",
            "therapy_progress": "",
        }
    )

    def load_user(self, user_id: str) -> Dict:
        summary = self.memgpt_client.get_core_memory(user_id)
        return summary or self.default_template.copy()

    def update_belief_shift(self, user_id: str, belief_delta: Dict) -> None:
        payload = {
            "type": "belief_update",
            "delta": belief_delta,
        }
        self.memgpt_client.append_event(user_id, payload)

    def log_session(self, user_id: str, transcript: str, reward: float) -> None:
        payload = {
            "type": "session_summary",
            "transcript": transcript,
            "reward": reward,
        }
        self.memgpt_client.append_event(user_id, payload)


__all__ = ["PKBManager"]

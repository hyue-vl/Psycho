"""Structured data models shared across Psycho-World modules."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence


@dataclass
class BeliefState:
    """Represents the latent user belief factors."""

    emotion: str
    intensity: float
    cognitive_distortion: Optional[str]
    intent: Optional[str]
    risk_level: str
    psychodynamics: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "BeliefState":
        return cls(
            emotion=str(payload.get("emotion", "unknown")),
            intensity=float(payload.get("intensity", 0.0)),
            cognitive_distortion=payload.get("cognitive_distortion"),
            intent=payload.get("intent"),
            risk_level=str(payload.get("risk_level", "Low")),
            psychodynamics={
                str(k): float(v)
                for k, v in (payload.get("psychodynamics") or {}).items()
            },
        )

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class StrategyAction:
    """Agent action containing the meta strategy and utterance."""

    strategy: str
    utterance: str

    def to_prompt(self) -> str:
        return f"Strategy: {self.strategy}\nUtterance: {self.utterance}"


@dataclass
class PsychoState:
    """Full simulator state containing history + belief."""

    history: List[str]
    belief: BeliefState


@dataclass
class UserProfile:
    user_id: str
    demographics: Dict[str, str]
    triggers: Sequence[str]
    core_beliefs: Sequence[str]
    initial_state_desc: str


@dataclass
class ConversationTurn:
    speaker: str
    text: str


@dataclass
class WorldModelOutput:
    user_belief: BeliefState
    user_text: str


@dataclass
class RewardBreakdown:
    delta_belief: float
    risk_penalty: float
    empathy_score: float

    def total(self, alpha: float, beta: float, gamma: float) -> float:
        return alpha * self.delta_belief + beta * self.risk_penalty + gamma * self.empathy_score

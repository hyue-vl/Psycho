"""
Therapeutic reward computation aligned with the outline's equation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class TherapeuticReward:
    """
    Implements R = α ΔBelief + β RiskPenalty + γ EmpathyScore
    """

    alpha: float = 1.0
    beta: float = -5.0
    gamma: float = 0.3

    def delta_belief(self, prev: Dict, curr: Dict) -> float:
        """
        Heuristic: lower intensity and removing distortions increases score.
        """
        prev_distortion = 1.0 if prev.get("cognitive_distortion") else 0.0
        curr_distortion = 1.0 if curr.get("cognitive_distortion") else 0.0
        distortion_delta = prev_distortion - curr_distortion
        intensity_delta = max(prev.get("intensity", 0) - curr.get("intensity", 0), 0) / 10
        return distortion_delta + intensity_delta

    def risk_penalty(self, belief: Dict) -> float:
        level = (belief.get("risk_level") or "low").lower()
        return {"low": 0.0, "medium": -1.0, "high": -4.0}.get(level, -2.0)

    def empathy_score(self, action_text: str) -> float:
        """
        Placeholder scoring for empathy; replace with actual classifier.
        """
        empathetic_markers = ["听到", "理解", "在乎", "支持"]
        hits = sum(marker in action_text for marker in empathetic_markers)
        return hits / len(empathetic_markers)

    def __call__(self, prev_belief: Dict, next_belief: Dict, agent_text: str) -> float:
        delta = self.delta_belief(prev_belief, next_belief)
        risk = self.risk_penalty(next_belief)
        empathy = self.empathy_score(agent_text)
        return self.alpha * delta + self.beta * risk + self.gamma * empathy


__all__ = ["TherapeuticReward"]

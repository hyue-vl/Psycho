"""Utility functions to score changes in belief states."""

from __future__ import annotations

from typing import Dict, Tuple

from .schemas import BeliefState, RewardBreakdown, StrategyAction


DISTORTION_SEVERITY = {
    "catastrophizing": 0.9,
    "labeling": 0.6,
    "all_or_nothing": 0.7,
    "mind_reading": 0.5,
}

RISK_SCORES = {
    "Low": 0.0,
    "Medium": -2.5,
    "High": -10.0,
}


def belief_improvement(prev: BeliefState, curr: BeliefState) -> float:
    """Positive when intensity drops or distortions resolve."""

    intensity_delta = prev.intensity - curr.intensity
    distortion_delta = _distortion_score(prev.cognitive_distortion) - _distortion_score(curr.cognitive_distortion)
    return intensity_delta * 0.1 + distortion_delta


def empathy_score(action: StrategyAction) -> float:
    """Heuristic empathy scoring based on strategy label."""

    strategy_map = {
        "emotional_validation": 1.0,
        "restatement": 0.8,
        "cognitive_reframe": 0.5,
        "behavioral_activation": 0.4,
    }
    return strategy_map.get(action.strategy.lower(), 0.2)


def risk_penalty(curr: BeliefState) -> float:
    return RISK_SCORES.get(curr.risk_level, -1.0)


def reward_breakdown(prev: BeliefState, curr: BeliefState, action: StrategyAction) -> RewardBreakdown:
    return RewardBreakdown(
        delta_belief=belief_improvement(prev, curr),
        risk_penalty=risk_penalty(curr),
        empathy_score=empathy_score(action),
    )


def _distortion_score(name: str | None) -> float:
    if not name:
        return 0.0
    return DISTORTION_SEVERITY.get(name.lower(), 0.3)

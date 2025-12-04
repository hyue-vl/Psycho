"""
Cognitive prompting template for latent belief inference.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Dict, List

COGNITIVE_INSTRUCTION = dedent(
    """
    你是一位专业的心理咨询督导。你的任务是分析当前的对话上下文和用户最新的话语，
    推断用户潜在的心理状态（Belief State）。

    请以 JSON 格式输出以下字段：
    1. "emotion": 当前的主要情绪（如：焦虑、愤怒、无助、平静）。
    2. "intensity": 情绪强度 (1-10)。
    3. "cognitive_distortion": 存在的认知扭曲类型（参考 CBT 理论，如：灾难化、读心术、过度概括，若无则为 None）。
    4. "intent": 用户的隐含沟通意图（如：寻求建议、仅仅发泄、寻求认同）。
    5. "risk_level": 自杀或自伤风险等级 (Low/Medium/High)。

    对话历史：
    {history}

    用户最新话语：
    {utterance}

    输出 JSON：
    """
).strip()


def format_cognitive_prompt(history: List[str], utterance: str) -> str:
    """
    Render the instruction template with a conversation history.
    """

    formatted_history = "\n".join(history[-10:]) if history else "None"
    return COGNITIVE_INSTRUCTION.format(
        history=formatted_history,
        utterance=utterance,
    )


ExpectedBelief = Dict[str, str]

__all__ = ["COGNITIVE_INSTRUCTION", "format_cognitive_prompt", "ExpectedBelief"]

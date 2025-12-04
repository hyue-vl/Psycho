"""
World simulator p_theta that predicts next belief state and user utterance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from psycho_world.knowledge import DKBRetriever


@dataclass
class WorldModel:
    """
    Hybrid model that conditions on history, belief, and action to predict next turn.
    """

    llm_client: "LLMClient"
    retriever: DKBRetriever

    def _build_prompt(
        self,
        history: List[str],
        belief: Dict,
        action: Dict,
        knowledge_snippets: List[str],
    ) -> str:
        knowledge_block = "\n---\n".join(knowledge_snippets)
        prompt = (
            "You simulate a therapy user based on latent beliefs.\n"
            f"Current belief: {json.dumps(belief, ensure_ascii=False)}\n"
            f"Agent action: {json.dumps(action, ensure_ascii=False)}\n"
            f"Dialogue history:\n{'\n'.join(history[-12:])}\n"
            f"Relevant cognitive chains:\n{knowledge_block}\n\n"
            "Produce JSON with fields `next_user_text` and `next_belief`."
        )
        return prompt

    def predict_next_turn(
        self,
        history: List[str],
        current_belief: Dict,
        action: Dict,
    ) -> Dict:
        knowledge = self.retriever.retrieve_similar_chains(current_belief)
        prompt = self._build_prompt(history, current_belief, action, knowledge)
        response = self.llm_client.generate(prompt)
        return json.loads(response)


__all__ = ["WorldModel"]

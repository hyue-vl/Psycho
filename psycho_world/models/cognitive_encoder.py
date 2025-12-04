"""
Wrapper around an instruction-following LLM that infers latent belief states.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

from psycho_world.prompts import format_cognitive_prompt


@dataclass
class CognitiveEncoder:
    """
    State estimator q_phi(h_t) → b_t.
    """

    llm_client: "LLMClient"

    def infer(self, history: List[str], utterance: str) -> Dict:
        prompt = format_cognitive_prompt(history, utterance)
        response = self.llm_client.generate(prompt)
        return json.loads(response)


__all__ = ["CognitiveEncoder"]

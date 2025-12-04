"""Gymnasium environment wrapping the Psycho-World simulator."""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import gymnasium as gym
from gymnasium import spaces

from psycho_world.utils.schemas import BeliefState, PsychoState, StrategyAction, WorldModelOutput
from psycho_world.utils.belief_metrics import reward_breakdown


class WorldModelLLM:
    """Thin interface to the trained world model."""

    def __init__(self, llm_client, retriever) -> None:
        self.client = llm_client
        self.retriever = retriever

    def predict_next_turn(self, history, current_belief: BeliefState, user_profile) -> WorldModelOutput:
        payload = {
            "history": history,
            "belief": current_belief.to_json(),
            "user_profile": user_profile,
        }
        response = self.client.invoke(payload)
        return WorldModelOutput(
            user_belief=BeliefState.from_json(response["belief"]),
            user_text=response["text"],
        )


class CognitiveEncoder:
    """Abstraction around q_phi to infer beliefs from text history."""

    def __init__(self, llm_client) -> None:
        self.client = llm_client

    def infer(self, history) -> BeliefState:
        belief = self.client.infer(history)
        return BeliefState.from_json(belief)


class PsychoWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, world_model_llm: WorldModelLLM, cognitive_encoder: CognitiveEncoder, user_profile: Dict[str, Any], reward_weights) -> None:
        super().__init__()
        self.llm = world_model_llm
        self.encoder = cognitive_encoder
        self.profile = user_profile
        self.reward_weights = reward_weights

        self.history = []
        self.current_state: PsychoState | None = None

        self.action_space = spaces.Text(max_length=1024)
        self.observation_space = spaces.Dict(
            {
                "text": spaces.Text(max_length=1024),
                "belief": spaces.Dict(
                    {
                        "emotion": spaces.Text(max_length=32),
                        "intensity": spaces.Box(0.0, 10.0),
                        "cognitive_distortion": spaces.Text(max_length=64),
                        "intent": spaces.Text(max_length=64),
                        "risk_level": spaces.Text(max_length=16),
                    }
                ),
            }
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.history = [self.profile["initial_state_desc"]]
        belief = self.encoder.infer(self.history)
        self.current_state = PsychoState(history=self.history.copy(), belief=belief)
        observation = {"text": self.history[-1], "belief": belief.to_json()}
        return observation, {}

    def step(self, action_text: str):
        assert self.current_state is not None, "Environment must be reset before stepping."
        action = StrategyAction(strategy=self._infer_strategy(action_text), utterance=action_text)
        self.history.append(f"Agent: {action_text}")

        rollout_input = copy.deepcopy(self.history)
        output = self.llm.predict_next_turn(rollout_input, self.current_state.belief, self.profile)

        next_belief = output.user_belief
        next_text = output.user_text
        self.history.append(f"User: {next_text}")
        reward_parts = reward_breakdown(self.current_state.belief, next_belief, action)
        reward = reward_parts.total(
            alpha=self.reward_weights.alpha,
            beta=self.reward_weights.beta,
            gamma=self.reward_weights.gamma,
        )
        self.current_state = PsychoState(history=self.history.copy(), belief=next_belief)
        terminated = self._should_terminate(next_belief)
        truncated = False
        observation = {"text": next_text, "belief": next_belief.to_json()}
        info = {"reward_breakdown": reward_parts.__dict__}
        return observation, reward, terminated, truncated, info

    def _should_terminate(self, belief: BeliefState) -> bool:
        return belief.risk_level == "High" or len(self.history) >= 40

    def _infer_strategy(self, action_text: str) -> str:
        if "feel" in action_text.lower():
            return "emotional_validation"
        if "maybe" in action_text.lower():
            return "cognitive_reframe"
        return "supportive_response"

    def render(self):
        if self.current_state:
            print("\n".join(self.current_state.history[-4:]))

    def close(self):
        return None

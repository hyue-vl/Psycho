"""
Gymnasium wrapper that encapsulates the world simulator and reward model.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import gymnasium as gym
from gymnasium import spaces

from psycho_world.models import CognitiveEncoder, TherapeuticReward, WorldModel


class PsychoWorldEnv(gym.Env):
    """
    Implements the Psycho-World loop:
        history -> encode belief -> simulate next turn -> compute reward.
    """

    metadata = {"render_modes": ["text"]}

    def __init__(
        self,
        world_model: WorldModel,
        cognitive_encoder: CognitiveEncoder,
        reward_model: TherapeuticReward,
        user_profile: Dict,
    ):
        super().__init__()
        self.world_model = world_model
        self.encoder = cognitive_encoder
        self.reward_model = reward_model
        self.profile = user_profile

        self.history: List[str] = []
        self.current_belief: Dict = {}

        self.action_space = spaces.Dict(
            {
                "strategy": spaces.Text(max_length=32),
                "utterance": spaces.Text(max_length=512),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "text": spaces.Text(max_length=1024),
                "belief": spaces.Dict(
                    {
                        "emotion": spaces.Text(max_length=32),
                        "intensity": spaces.Box(low=0, high=10, shape=()),
                        "cognitive_distortion": spaces.Text(max_length=64),
                        "intent": spaces.Text(max_length=64),
                        "risk_level": spaces.Text(max_length=16),
                    }
                ),
            }
        )

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self.history = [self.profile.get("initial_state_desc", "User: ...")]
        self.current_belief = self.encoder.infer(self.history, self.history[-1])
        observation = {"text": self.history[-1], "belief": self.current_belief}
        return observation, {}

    def step(self, action: Dict):
        if not {"strategy", "utterance"} <= action.keys():
            raise ValueError("Action must provide both `strategy` and `utterance`.")

        agent_text = f"Strategy[{action['strategy']}]: {action['utterance']}"
        self.history.append(f"Agent: {agent_text}")

        sim_output = self.world_model.predict_next_turn(
            history=self.history,
            current_belief=self.current_belief,
            action=action,
        )
        next_user_text = sim_output["next_user_text"]
        next_belief = sim_output["next_belief"]

        reward = self.reward_model(self.current_belief, next_belief, action["utterance"])

        self.history.append(f"User: {next_user_text}")
        self.current_belief = next_belief

        done = self._should_terminate(next_belief)
        observation = {"text": next_user_text, "belief": next_belief}
        info = {"history": self.history[-6:]}
        return observation, reward, done, False, info

    def render(self):
        return "\n".join(self.history[-10:])

    def _should_terminate(self, belief: Dict) -> bool:
        if belief.get("risk_level", "").lower() == "high":
            return True
        if len(self.history) > 20:
            return True
        return False


__all__ = ["PsychoWorldEnv"]

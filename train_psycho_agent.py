"""Entrypoint for training the dreaming agent inside Psycho-World env."""

from __future__ import annotations

from psycho_world.agents.dreaming_agent import DreamingAgent
from psycho_world.envs.psycho_world import CognitiveEncoder, PsychoWorldEnv, WorldModelLLM
from psycho_world.memory.pkb_manager import PKBManager
from psycho_world.configs import RewardWeights


class DummyLLM:
    def infer(self, history):
        return {
            "emotion": "anxiety",
            "intensity": 7,
            "cognitive_distortion": "catastrophizing",
            "intent": "seeking_confirmation",
            "risk_level": "Low",
        }

    def invoke(self, payload):
        return {
            "text": "I still feel overwhelmed but maybe there is hope.",
            "belief": {
                "emotion": "hopeful",
                "intensity": 4,
                "cognitive_distortion": None,
                "intent": "collaborate",
                "risk_level": "Low",
            },
        }

    def generate(self, prompt: str) -> str:
        return "It sounds heavy; would grounding exercises feel manageable for you?"


if __name__ == "__main__":
    llm = DummyLLM()
    world_model = WorldModelLLM(llm, retriever=None)
    encoder = CognitiveEncoder(llm)
    env = PsychoWorldEnv(
        world_model_llm=world_model,
        cognitive_encoder=encoder,
        user_profile={"initial_state_desc": "User: I cannot stop worrying."},
        reward_weights=RewardWeights(),
    )
    agent = DreamingAgent(env, language_model=llm)
    obs, _ = env.reset()
    for _ in range(3):
        action = agent.select_action(obs)
        obs, reward, done, _, info = env.step(action)
        print("reward", reward, "info", info)
        if done:
            break

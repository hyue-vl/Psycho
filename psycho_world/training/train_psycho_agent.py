"""
Reinforcement learning / dreaming loop for policy optimization.
"""

from __future__ import annotations

from typing import List

from psycho_world.agents import DreamingAgent
from psycho_world.config import DreamingConfig
from psycho_world.envs import PsychoWorldEnv


def train_psycho_agent(
    env: PsychoWorldEnv,
    config: DreamingConfig,
    candidate_action_generator,
    episodes: int = 100,
):
    agent = DreamingAgent(env, config)

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        trajectory = []

        while not done:
            candidates = candidate_action_generator(state)
            action = agent.select_action(state, candidates)
            next_state, reward, done, _, info = env.step(action)
            trajectory.append((state, action, reward, next_state))
            state = next_state

        yield {
            "episode": episode,
            "trajectory": trajectory,
            "final_reward": sum(t[2] for t in trajectory),
        }


__all__ = ["train_psycho_agent"]

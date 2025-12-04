"""MCTS-style dreaming agent for inference-time planning."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from psycho_world.envs.psycho_world import PsychoWorldEnv


@dataclass
class PlannerConfig:
    simulations_per_step: int = 8
    rollout_depth: int = 3
    exploration_coef: float = 1.4


@dataclass
class TreeNode:
    state_summary: str
    visits: int = 0
    value: float = 0.0
    children: dict | None = None

    def q(self) -> float:
        return self.value / self.visits if self.visits else 0.0


class DreamingAgent:
    def __init__(self, env: PsychoWorldEnv, language_model, planner_cfg: PlannerConfig | None = None):
        self.env = env
        self.lm = language_model
        self.cfg = planner_cfg or PlannerConfig()

    def select_action(self, observation: dict, run_simulation: bool = True) -> str:
        if not run_simulation:
            return self._sample_action(observation)
        root = TreeNode(state_summary=self._state_fingerprint(observation))
        for _ in range(self.cfg.simulations_per_step):
            self._simulate(root)
        best_child = max(root.children.items(), key=lambda item: item[1].visits)[1]
        return best_child.state_summary.split("|action=")[1]

    def _simulate(self, node: TreeNode) -> float:
        if node.children is None:
            node.children = {}
            return self._expand(node)
        action, child = self._select_child(node)
        reward = self._simulate(child)
        node.visits += 1
        node.value += reward
        return reward

    def _expand(self, node: TreeNode) -> float:
        observation = self._parse_summary(node.state_summary)
        candidates = [self._sample_action(observation) for _ in range(3)]
        rewards = []
        for action in candidates:
            sim_env = copy.deepcopy(self.env)
            obs, reward, _, _, _ = sim_env.step(action)
            summary = self._state_fingerprint(obs, action)
            node.children[action] = TreeNode(state_summary=summary)
            rewards.append(reward)
        avg_reward = sum(rewards) / max(len(rewards), 1)
        node.visits += 1
        node.value += avg_reward
        return avg_reward

    def _select_child(self, node: TreeNode) -> Tuple[str, TreeNode]:
        total_visits = sum(child.visits for child in node.children.values()) + 1
        best, best_score = None, -float("inf")
        for action, child in node.children.items():
            exploit = child.q()
            explore = math.sqrt(math.log(total_visits) / (child.visits + 1e-6))
            score = exploit + self.cfg.exploration_coef * explore
            if score > best_score:
                best, best_score = action, score
        return best, node.children[best]

    def _sample_action(self, observation: dict) -> str:
        prompt = self._build_action_prompt(observation)
        return self.lm.generate(prompt)

    def _build_action_prompt(self, observation: dict) -> str:
        belief = observation["belief"]
        return (
            "You are a therapist planning a short reply. Current belief: "
            f"emotion={belief['emotion']} intensity={belief['intensity']} distortion={belief['cognitive_distortion']}. "
            "Produce a sentence under 60 tokens."
        )

    def _state_fingerprint(self, observation: dict, action: str | None = None) -> str:
        belief = observation["belief"]
        summary = f"emotion={belief['emotion']}|int={belief['intensity']}"
        if action:
            summary += f"|action={action}"
        return summary

    def _parse_summary(self, summary: str) -> dict:
        parts = dict(part.split("=") for part in summary.split("|") if "=" in part)
        belief = {"emotion": parts.get("emotion", ""), "intensity": float(parts.get("int", 0))}
        return {"belief": belief, "text": ""}

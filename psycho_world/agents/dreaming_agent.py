"""
Dreaming agent that performs limited-depth MCTS rollout before acting.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from psycho_world.config import DreamingConfig
from psycho_world.envs import PsychoWorldEnv


@dataclass
class Node:
    state: Dict
    parent: "Node | None"
    action: Dict | None
    visits: int = 0
    total_value: float = 0.0
    children: List["Node"] = None

    def __post_init__(self):
        self.children = self.children or []

    @property
    def value(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0


class DreamingAgent:
    """
    MCTS-style planner that uses the Psycho-World env for internal simulation.
    """

    def __init__(self, env: PsychoWorldEnv, config: DreamingConfig):
        self.env = env
        self.config = config

    def select_action(self, state: Dict, candidate_actions: List[Dict]) -> Dict:
        root = Node(state=state, parent=None, action=None)

        for _ in range(self.config.num_simulations):
            env_copy = self._clone_env()
            node = root
            depth = 0

            # Selection
            while node.children and depth < self.config.rollout_horizon:
                node = self._select_child(node)
                _, _, done, _, _ = env_copy.step(node.action)
                depth += 1
                if done:
                    break

            # Expansion
            if depth < self.config.rollout_horizon:
                actions = self._expand_actions(candidate_actions)
                for action in actions:
                    child_state, reward, done, _, _ = env_copy.step(action)
                    child = Node(state=child_state, parent=node, action=action)
                    child.visits += 1
                    child.total_value += reward
                    node.children.append(child)
                    if done:
                        break

            # Backpropagation
            value = node.value
            while node:
                node.visits += 1
                node.total_value += value
                node = node.parent

        best_child = max(root.children, key=lambda c: c.visits, default=None)
        return best_child.action if best_child else random.choice(candidate_actions)

    def _select_child(self, node: Node) -> Node:
        def ucb_score(child: Node) -> float:
            if child.visits == 0:
                return float("inf")
            exploitation = child.value
            exploration = math.sqrt(math.log(node.visits) / child.visits)
            return exploitation + self.config.ucb_exploration * exploration

        return max(node.children, key=ucb_score)

    def _expand_actions(self, candidates: List[Dict]) -> List[Dict]:
        random.shuffle(candidates)
        return candidates[: self.config.branching_factor]

    def _clone_env(self) -> PsychoWorldEnv:
        import copy

        return copy.deepcopy(self.env)


__all__ = ["DreamingAgent"]

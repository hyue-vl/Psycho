"""
Global configuration dataclasses shared across Psycho-World modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class KnowledgeBaseConfig:
    """Configuration for the dual knowledge base (DKB + PKB)."""

    coke_json_path: Path
    chroma_dir: Path
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    memgpt_api_key: Optional[str] = None


@dataclass
class AnnotationConfig:
    """LLM annotation parameters for hindsight labeling."""

    dataset_name: str
    max_turns: int = 12
    teacher_model: str = "gpt-4o-mini"
    temperature: float = 0.2
    batch_size: int = 4
    save_dir: Path = Path("./data/processed")
    prompt_template: Path = Path("./psycho_world/prompts/cognitive.py")


@dataclass
class TrainingConfig:
    """Shared hyperparameters for supervised or RL training."""

    model_name: str
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    gradient_accumulation: int = 2
    max_steps: int = 20_000
    save_steps: int = 1_000
    log_steps: int = 50
    output_dir: Path = Path("./artifacts")
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class DreamingConfig:
    """Configuration for the dreaming/MCTS agent."""

    rollout_horizon: int = 3
    num_simulations: int = 20
    branching_factor: int = 4
    gamma: float = 0.95
    ucb_exploration: float = 1.4
    reward_shaping_weights: List[float] = field(
        default_factory=lambda: [1.0, -5.0, 0.3]
    )  # α, β, γ in the spec


__all__ = [
    "KnowledgeBaseConfig",
    "AnnotationConfig",
    "TrainingConfig",
    "DreamingConfig",
]

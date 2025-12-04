"""Centralized configuration dataclasses for Psycho-World."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class COKEPaths:
    raw_path: Path = Path("./data/raw/coke.json")
    vectordb_dir: Path = Path("./data/coke_db")
    graph_uri: str = "bolt://localhost:7687"
    graph_user: str = "neo4j"
    graph_password: str = "password"


@dataclass
class TrainingConfig:
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    learning_rate: float = 1e-4
    batch_size: int = 4
    epochs: int = 3
    warmup_ratio: float = 0.05
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    output_dir: Path = Path("./outputs/cognitive_encoder")


@dataclass
class WorldModelConfig(TrainingConfig):
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    output_dir: Path = Path("./outputs/world_model")


@dataclass
class RewardWeights:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5

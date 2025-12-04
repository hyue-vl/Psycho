"""
Global configuration dataclasses shared across Psycho-World modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


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


@dataclass
class LLMConfig:
    """Provider-agnostic parameters for text generation backends."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_output_tokens: int = 1_024
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class RuntimeConfig:
    """
    Canonical bundle of subsystem configs. Keeps scripts in sync and clarifies wiring.
    """

    llm: LLMConfig
    knowledge_base: KnowledgeBaseConfig
    annotation: AnnotationConfig
    training: TrainingConfig
    dreaming: DreamingConfig

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RuntimeConfig":
        """Instantiate the typed config bundle from a plain dict (e.g., JSON)."""
        llm_cfg = LLMConfig(**(raw.get("llm") or {}))

        kb_raw = raw.get("knowledge_base") or {}
        knowledge_cfg = KnowledgeBaseConfig(
            coke_json_path=_require_path(
                kb_raw.get("coke_json_path"), "knowledge_base.coke_json_path"
            ),
            chroma_dir=_require_path(
                kb_raw.get("chroma_dir"), "knowledge_base.chroma_dir"
            ),
            neo4j_uri=kb_raw.get("neo4j_uri", "bolt://localhost:7687"),
            neo4j_user=kb_raw.get("neo4j_user", "neo4j"),
            neo4j_password=kb_raw.get("neo4j_password", "please-change-me"),
            memgpt_api_key=kb_raw.get("memgpt_api_key"),
        )

        annotation_raw = raw.get("annotation") or {}
        annotation_cfg = AnnotationConfig(
            dataset_name=annotation_raw.get("dataset_name", "psyqa-esconv"),
            max_turns=annotation_raw.get("max_turns", 12),
            teacher_model=annotation_raw.get("teacher_model", "gpt-4o-mini"),
            temperature=annotation_raw.get("temperature", 0.2),
            batch_size=annotation_raw.get("batch_size", 4),
            save_dir=_optional_path(
                annotation_raw.get("save_dir"), Path("./data/processed")
            ),
            prompt_template=_optional_path(
                annotation_raw.get("prompt_template"),
                Path("./psycho_world/prompts/cognitive.py"),
            ),
        )

        training_raw = raw.get("training") or {}
        training_cfg = TrainingConfig(
            model_name=training_raw.get("model_name", "Qwen/Qwen2-7B-Instruct"),
            learning_rate=training_raw.get("learning_rate", 2e-5),
            weight_decay=training_raw.get("weight_decay", 0.01),
            warmup_ratio=training_raw.get("warmup_ratio", 0.05),
            gradient_accumulation=training_raw.get("gradient_accumulation", 2),
            max_steps=training_raw.get("max_steps", 20_000),
            save_steps=training_raw.get("save_steps", 1_000),
            log_steps=training_raw.get("log_steps", 50),
            output_dir=_optional_path(
                training_raw.get("output_dir"), Path("./artifacts")
            ),
            use_lora=training_raw.get("use_lora", True),
            lora_r=training_raw.get("lora_r", 16),
            lora_alpha=training_raw.get("lora_alpha", 32),
            lora_dropout=training_raw.get("lora_dropout", 0.05),
        )

        dreaming_raw = raw.get("dreaming") or {}
        dreaming_cfg = DreamingConfig(
            rollout_horizon=dreaming_raw.get("rollout_horizon", 3),
            num_simulations=dreaming_raw.get("num_simulations", 20),
            branching_factor=dreaming_raw.get("branching_factor", 4),
            gamma=dreaming_raw.get("gamma", 0.95),
            ucb_exploration=dreaming_raw.get("ucb_exploration", 1.4),
            reward_shaping_weights=dreaming_raw.get(
                "reward_shaping_weights", [1.0, -5.0, 0.3]
            ),
        )

        return cls(
            llm=llm_cfg,
            knowledge_base=knowledge_cfg,
            annotation=annotation_cfg,
            training=training_cfg,
            dreaming=dreaming_cfg,
        )


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """
    Load a `RuntimeConfig` from a JSON file.

    Example:
        runtime_cfg = load_runtime_config("configs/default.json")
    """

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix != ".json":
        raise ValueError(
            f"Unsupported config format '{suffix}'. Provide a JSON file instead."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = json.load(fh)
    return RuntimeConfig.from_dict(raw)


def _require_path(value: str | Path | None, field_name: str) -> Path:
    if value is None:
        raise ValueError(f"Missing required config field: {field_name}")
    return Path(value).expanduser()


def _optional_path(value: str | Path | None, default: Path) -> Path:
    return Path(value).expanduser() if value is not None else default


__all__ = [
    "KnowledgeBaseConfig",
    "AnnotationConfig",
    "TrainingConfig",
    "DreamingConfig",
    "LLMConfig",
    "RuntimeConfig",
    "load_runtime_config",
]

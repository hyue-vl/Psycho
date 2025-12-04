"""
Supervised fine-tuning loop for the Cognitive Encoder (q_phi).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from transformers import Trainer, TrainingArguments

from psycho_world.config import TrainingConfig


def train_cognitive_encoder(
    dataset_path: Path,
    config: TrainingConfig,
):
    """
    Load annotated dialogues and run instruction fine-tuning.
    """
    dataset = load_dataset("json", data_files=str(dataset_path))

    def format_example(example):
        return {
            "input_ids": example["history"],
            "labels": json.dumps(example["belief"], ensure_ascii=False),
        }

    tokenized = dataset.map(format_example)

    args = TrainingArguments(
        output_dir=str(config.output_dir / "cognitive_encoder"),
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        logging_steps=config.log_steps,
        save_steps=config.save_steps,
        weight_decay=config.weight_decay,
    )

    trainer = Trainer(
        args=args,
        train_dataset=tokenized["train"],
    )
    trainer.train()
    trainer.save_model()


__all__ = ["train_cognitive_encoder"]

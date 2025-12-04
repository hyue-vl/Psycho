"""Supervised training pipeline for the world simulator p_theta."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from psycho_world.configs import WorldModelConfig, COKEPaths
from psycho_world.knowledge.retriever import DKBRetriever
from psycho_world.utils.schemas import BeliefState, StrategyAction


PROMPT_TEMPLATE = """
[History]\n{history}\n\n[Current Belief]\n{belief}\n\n[Agent Action]\n{action}\n\nUsing clinical reasoning and retrieved analogues:\n{retrieved}\n\nPredict the next user utterance and belief JSON.
"""


class WorldModelDataset(Dataset):
    def __init__(self, path: Path, tokenizer, retriever: DKBRetriever, max_length: int = 4096):
        self.samples = json.loads(path.read_text())
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        history = "\n".join(sample["history"])
        belief = BeliefState.from_json(sample["user_belief"])
        action = StrategyAction(strategy=sample["agent_strategy"], utterance=sample["agent_response"])
        similar = self.retriever.retrieve_similar_chains(belief)
        prompt = PROMPT_TEMPLATE.format(
            history=history,
            belief=json.dumps(belief.to_json(), ensure_ascii=False),
            action=action.to_prompt(),
            retrieved="\n---\n".join(similar),
        )
        target = sample["next_user_text"] + "\n" + json.dumps(sample["next_belief"], ensure_ascii=False)
        text = f"{prompt}\n{target}"
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in tokens.items()}


@dataclass
class WorldModelTrainer:
    config: WorldModelConfig
    dataset_path: Path
    coke_paths: COKEPaths

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        lora_cfg = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(model, lora_cfg)
        self.retriever = DKBRetriever(
            persist_directory=str(self.coke_paths.vectordb_dir),
            graph_uri=self.coke_paths.graph_uri,
            graph_user=self.coke_paths.graph_user,
            graph_password=self.coke_paths.graph_password,
        )

    def run(self) -> None:
        dataset = WorldModelDataset(self.dataset_path, self.tokenizer, self.retriever)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        num_steps = len(dataloader) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_steps * self.config.warmup_ratio),
            num_training_steps=num_steps,
        )
        self.model.train()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        for epoch in range(self.config.epochs):
            for step, batch in enumerate(dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = self.model(**batch).loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if step % 10 == 0:
                    print(f"world-model epoch={epoch} step={step} loss={loss.item():.3f}")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)


if __name__ == "__main__":
    trainer = WorldModelTrainer(
        WorldModelConfig(),
        Path("./data/world_model_supervision.json"),
        COKEPaths(),
    )
    trainer.run()

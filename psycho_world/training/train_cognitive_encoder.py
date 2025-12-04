"""LoRA fine-tuning script for the cognitive encoder q_phi."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from psycho_world.configs import TrainingConfig
from psycho_world.prompts.cognitive import COGNITIVE_INSTRUCTION


def build_prompt(history: str, latest_user: str) -> str:
    return COGNITIVE_INSTRUCTION.format(history=history, utterance=latest_user)


class BeliefDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int = 4096):
        self.examples = json.loads(path.read_text())
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        example = self.examples[idx]
        history = "\n".join(f"{t['speaker']}: {t['text']}" for t in example["history"])
        prompt = build_prompt(history, example["history"][-1]["text"])
        target = json.dumps(example["user_belief"], ensure_ascii=False)
        input_text = f"{prompt}\n{target}"
        tokens = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in tokens.items()}


@dataclass
class Trainer:
    config: TrainingConfig
    dataset_path: Path

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.model.gradient_checkpointing_enable()
        lora_cfg = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_cfg)

    def run(self) -> None:
        dataset = BeliefDataset(self.dataset_path, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        num_training_steps = len(dataloader) * self.config.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * self.config.warmup_ratio),
            num_training_steps=num_training_steps,
        )
        self.model.train()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        for epoch in range(self.config.epochs):
            for step, batch in enumerate(dataloader):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if step % 10 == 0:
                    print(f"epoch={epoch} step={step} loss={loss.item():.4f}")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)


if __name__ == "__main__":
    trainer = Trainer(TrainingConfig(), Path("./data/annotated_beliefs.json"))
    trainer.run()

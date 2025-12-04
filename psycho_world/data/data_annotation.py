"""
LLM-powered hindsight labeling pipeline for ESConv/PsyQA conversations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm import tqdm

from psycho_world.prompts import format_cognitive_prompt


@dataclass
class DialogueAnnotator:
    """
    Annotate raw dialogue logs with latent belief labels using an LLM teacher.
    """

    teacher_client: "LLMClient"
    save_dir: Path
    max_turns: int = 12

    def _prepare_example(self, history: List[Dict]) -> Iterable[Dict]:
        """
        Yield user turns paired with preceding agent turns.
        """
        for idx, turn in enumerate(history):
            if turn["role"] != "user":
                continue
            context = history[max(0, idx - self.max_turns) : idx]
            context_text = [f"{t['role'].title()}: {t['content']}" for t in context]
            yield {
                "history": context_text,
                "utterance": turn["content"],
            }

    def annotate_dialogue(self, dialogue: List[Dict]) -> List[Dict]:
        """
        Annotate a single dialogue.
        """
        annotations = []
        for item in self._prepare_example(dialogue):
            prompt = format_cognitive_prompt(
                history=item["history"],
                utterance=item["utterance"],
            )
            response = self.teacher_client.generate(prompt)
            belief = json.loads(response)
            annotations.append(
                {
                    "history": item["history"],
                    "utterance": item["utterance"],
                    "belief": belief,
                }
            )
        return annotations

    def run(self, dataset: Iterable[List[Dict]], save_path: Path) -> Path:
        """
        Iterate over a dataset (iterator of dialogues) and persist annotations.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        all_annotations = []
        for dialogue in tqdm(dataset, desc="Annotating dialogues"):
            annotated_dialogue = self.annotate_dialogue(dialogue)
            all_annotations.append(annotated_dialogue)

        with save_path.open("w", encoding="utf-8") as file:
            json.dump(all_annotations, file, ensure_ascii=False, indent=2)
        return save_path


class LLMClient:
    """
    Thin wrapper for whichever LLM backend you use.
    """

    def __init__(self, openai_client):
        self.client = openai_client

    def generate(self, prompt: str):
        completion = self.client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            temperature=0.2,
        )
        return completion.output[0].content[0].text


__all__ = ["DialogueAnnotator", "LLMClient"]

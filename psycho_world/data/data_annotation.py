"""Produce hindsight belief annotations for ESConv / PsyQA datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from openai import OpenAI

from psycho_world.prompts.cognitive import COGNITIVE_INSTRUCTION


@dataclass
class DialogueTurn:
    speaker: str
    text: str


@dataclass
class AnnotatedExample:
    history: List[DialogueTurn]
    user_belief: dict
    agent_strategy: str
    agent_response: str

    def to_json(self) -> dict:
        return {
            "history": [turn.__dict__ for turn in self.history],
            "user_belief": self.user_belief,
            "agent_strategy": self.agent_strategy,
            "agent_response": self.agent_response,
        }


class DialogueAnnotator:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.2) -> None:
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def annotate_dialogue(self, turns: Sequence[DialogueTurn]) -> List[AnnotatedExample]:
        annotated: List[AnnotatedExample] = []
        history_buffer: List[DialogueTurn] = []
        for idx in range(1, len(turns), 2):
            history_buffer.append(turns[idx - 1])
            user_turn = turns[idx]
            belief = self._call_teacher(history_buffer, user_turn)
            strategy = turns[idx + 1].speaker if idx + 1 < len(turns) else ""
            response = turns[idx + 1].text if idx + 1 < len(turns) else ""
            annotated.append(
                AnnotatedExample(
                    history=history_buffer.copy(),
                    user_belief=belief,
                    agent_strategy=strategy,
                    agent_response=response,
                )
            )
        return annotated

    def _call_teacher(self, history: Sequence[DialogueTurn], user_turn: DialogueTurn) -> dict:
        rendered_history = "\n".join(f"{t.speaker}: {t.text}" for t in history)
        prompt = COGNITIVE_INSTRUCTION.format(history=rendered_history, utterance=user_turn.text)
        completion = self.client.responses.create(
            model=self.model,
            temperature=self.temperature,
            input=[{"role": "user", "content": prompt}],
        )
        content = completion.output[0].content[0].text
        return json.loads(content)


def annotate_dataset(dialogues_path: Path, output_path: Path) -> None:
    annotator = DialogueAnnotator()
    dialogues = json.loads(dialogues_path.read_text())
    all_annotations: List[dict] = []
    for dialogue in dialogues:
        turns = [DialogueTurn(**turn) for turn in dialogue["turns"]]
        examples = annotator.annotate_dialogue(turns)
        all_annotations.extend(example.to_json() for example in examples)
    output_path.write_text(json.dumps(all_annotations, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    annotate_dataset(Path("./data/dialogues.json"), Path("./data/annotated_beliefs.json"))

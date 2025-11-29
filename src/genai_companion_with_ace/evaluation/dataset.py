"""Evaluation dataset management for the IBM Gen AI Companion."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

CourseId = Literal[
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
]

DIFFICULTIES = ("basic", "intermediate", "advanced")
QUESTION_TYPES = ("conceptual", "code_example", "comparison", "debugging")


@dataclass(slots=True)
class EvaluationExample:
    id: str
    question: str
    course: CourseId
    module: str
    difficulty: str
    question_type: str
    golden_answer: str
    evaluation_criteria: dict[str, list[str]] = field(default_factory=dict)


COURSE_TOPICS: dict[CourseId, list[str]] = {
    "1": ["AI history", "AI definition"],
    "2": ["GenAI applications", "GenAI workflow"],
    "3": ["Prompt structure", "Prompt refinement"],
    "4": ["Python basics", "Data structures"],
    "5": ["Flask routing", "REST APIs"],
    "6": ["Generative apps", "LLM integration"],
    "7": ["Data visualization", "Pandas"],
    "8": ["Supervised learning", "Model evaluation"],
    "9": ["Backpropagation", "Neural networks"],
    "10": ["LLM architecture", "Data preparation"],
    "11": ["Foundational models", "Language understanding"],
    "12": ["Transformers", "Self-attention"],
    "13": ["Fine-tuning", "PEFT"],
    "14": ["Advanced tuning", "LoRA"],
    "15": ["RAG agents", "LangChain"],
    "16": ["Capstone", "Project planning"],
}


def _build_example(course: CourseId, topic: str, index: int, rng: random.Random) -> EvaluationExample:
    difficulty = rng.choice(DIFFICULTIES)
    qtype = rng.choice(QUESTION_TYPES)
    question = f"[{difficulty.title()}] Explain {topic} in the context of Course {course}."
    golden = f"Provide a detailed answer about {topic} referencing Course {course}."
    example_id = f"course{course}_{topic.replace(' ', '_')}_{index}"
    criteria = {
        "must_include": [topic],
        "should_cite": [f"Course {course}"],
    }
    return EvaluationExample(
        id=example_id,
        question=question,
        course=course,
        module=topic,
        difficulty=difficulty,
        question_type=qtype,
        golden_answer=golden,
        evaluation_criteria=criteria,
    )


def generate_synthetic_dataset(samples_per_course: int = 6, seed: int = 42) -> list[EvaluationExample]:
    rng = random.Random(seed)  # noqa: S311 - deterministic dataset generation
    dataset: list[EvaluationExample] = []
    for course, topics in COURSE_TOPICS.items():
        for idx in range(samples_per_course):
            topic = topics[idx % len(topics)]
            dataset.append(_build_example(course, topic, idx, rng))
    # Ensure at least 100 entries
    while len(dataset) < 100:
        topic = rng.choice(COURSE_TOPICS["16"])
        dataset.append(_build_example("16", topic, len(dataset), rng))
    return dataset


class EvaluationDataset:
    def __init__(self, examples: list[EvaluationExample]) -> None:
        self.examples = examples

    @classmethod
    def load(cls, path: Path) -> EvaluationDataset:
        payload = json.loads(path.read_text(encoding="utf-8"))
        examples = [EvaluationExample(**item) for item in payload]
        return cls(examples)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(example) for example in self.examples]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def filter_by_course(self, course: CourseId) -> EvaluationDataset:
        return EvaluationDataset([example for example in self.examples if example.course == course])

    def summary(self) -> dict[str, int]:
        stats: dict[str, int] = dict.fromkeys(COURSE_TOPICS, 0)
        for example in self.examples:
            stats[example.course] += 1
        return stats


def ensure_default_dataset(path: Path, samples_per_course: int = 6) -> EvaluationDataset:
    if path.exists():
        return EvaluationDataset.load(path)
    dataset = EvaluationDataset(generate_synthetic_dataset(samples_per_course=samples_per_course))
    dataset.save(path)
    return dataset

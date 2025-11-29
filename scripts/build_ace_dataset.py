"""Build an ACE-compatible dataset from curated GenAI questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_QUESTIONS = Path("data/ace/questions_genai.json")
DEFAULT_OUTPUT_DIR = Path("outputs/ace_datasets")


class DatasetBuildError(RuntimeError):
    """Raised when curated question data is missing or invalid."""


class QuestionsFileNotFound(DatasetBuildError):
    def __init__(self, path: Path) -> None:
        super().__init__(f"Questions file not found: {path}")
        self.path = path


class EmptyQuestionsFile(DatasetBuildError):
    def __init__(self) -> None:
        super().__init__("Questions file must be a non-empty JSON array.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ACE dataset from curated questions.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help=f"Path to curated questions JSON (default: {DEFAULT_QUESTIONS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dataset path (default: outputs/ace_datasets/ace_dataset_<timestamp>.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions included (default: all)",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Seed for shuffling questions before limiting (default: none, original order)",
    )
    return parser.parse_args()


def load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise QuestionsFileNotFound(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise EmptyQuestionsFile()
    return data


def synthesize_answer(question: dict[str, Any]) -> str:
    module = question.get("module", "the topic")
    key_points = question.get("key_points") or []
    lines = [
        f"Here is a concise explanation about {module.lower()}:",
        "",
    ]
    if key_points:
        lines.append("Key takeaways:")
        for point in key_points:
            lines.append(f"- {point}")
        lines.append("")
    lines.append(
        "This summary is grounded in the IBM GenAI Professional Certificate materials. "
        "Always cite the relevant course and module when using it in an assistant response."
    )
    return textwrap.dedent("\n".join(lines)).strip()


def build_dataset(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    for entry in questions:
        session_id = f"ace-gen-{uuid4().hex}"
        answer = synthesize_answer(entry)
        dataset.append({
            "session_id": session_id,
            "input": entry["question"],
            "output": answer,
            "reference_output": answer,
            "sources": [
                {
                    "source": entry.get("course", "IBM GenAI Course"),
                    "module": entry.get("module", ""),
                }
            ],
            "metadata": {
                "course": entry.get("course", ""),
                "module": entry.get("module", ""),
                "question_id": entry.get("id", ""),
            },
        })
    return dataset


def resolve_output_path(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"ace_dataset_{timestamp}.json"


def main() -> None:
    args = parse_args()
    try:
        questions = load_questions(args.questions)
    except DatasetBuildError as exc:
        print(f"Failed to load questions: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.shuffle_seed is not None:
        seed_bytes = str(args.shuffle_seed).encode("utf-8")
        keyed = [
            (hashlib.sha256(seed_bytes + str(idx).encode("utf-8")).digest(), question)
            for idx, question in enumerate(questions)
        ]
        keyed.sort(key=lambda item: item[0])
        questions = [entry for _, entry in keyed]

    if args.limit is not None:
        questions = questions[: args.limit]

    dataset = build_dataset(questions)

    output_path = resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Generated dataset with {len(dataset)} entries -> {output_path}")


if __name__ == "__main__":
    main()

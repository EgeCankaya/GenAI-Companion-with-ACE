"""Validate curated GenAI question files used for ACE dataset generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = ("id", "question", "course", "module")


class QuestionValidationError(RuntimeError):
    """Raised when the curated question set fails validation."""


def _validate_entry(entry: dict[str, Any], seen_ids: set[str], index: int) -> None:
    if not isinstance(entry, dict):
        raise TypeError(f"Entry #{index} is not an object")

    for field in REQUIRED_FIELDS:
        if field not in entry:
            raise QuestionValidationError(f"Entry #{index} missing required field '{field}'")
        value = entry[field]
        if not isinstance(value, str) or not value.strip():
            raise QuestionValidationError(f"Field '{field}' in entry #{index} must be a non-empty string")

    entry_id = entry["id"]
    if entry_id in seen_ids:
        raise QuestionValidationError(f"Duplicate id '{entry_id}' found (entry #{index})")
    seen_ids.add(entry_id)

    key_points = entry.get("key_points", []) or []
    if not isinstance(key_points, list):
        raise TypeError(f"'key_points' in entry #{index} must be a list")
    for point in key_points:
        if not isinstance(point, str) or not point.strip():
            raise QuestionValidationError(
                f"Each key point in entry #{index} must be a non-empty string (found: {point!r})"
            )


def validate_file(path: Path) -> int:
    if not path.exists():
        raise QuestionValidationError(f"Questions file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise QuestionValidationError("Questions file must be a non-empty JSON array")

    seen_ids: set[str] = set()
    for index, entry in enumerate(raw, start=1):
        _validate_entry(entry, seen_ids, index)

    return len(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate curated GenAI question set.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/ace/questions_genai.json"),
        help="Path to questions JSON (default: data/ace/questions_genai.json)",
    )
    args = parser.parse_args()

    try:
        count = validate_file(args.questions)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Validation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Validation succeeded: {count} questions in {args.questions}")


if __name__ == "__main__":
    main()


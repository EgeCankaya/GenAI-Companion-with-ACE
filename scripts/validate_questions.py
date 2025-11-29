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

    @classmethod
    def invalid_entry(cls, index: int) -> QuestionValidationError:
        return cls(f"Entry #{index} is not an object")

    @classmethod
    def missing_field(cls, index: int, field: str) -> QuestionValidationError:
        return cls(f"Entry #{index} missing required field '{field}'")

    @classmethod
    def empty_field(cls, index: int, field: str) -> QuestionValidationError:
        return cls(f"Field '{field}' in entry #{index} must be a non-empty string")

    @classmethod
    def duplicate_id(cls, entry_id: str, index: int) -> QuestionValidationError:
        return cls(f"Duplicate id '{entry_id}' found (entry #{index})")

    @classmethod
    def invalid_key_points(cls, index: int) -> QuestionValidationError:
        return cls(f"'key_points' in entry #{index} must be a list")

    @classmethod
    def invalid_key_point_value(cls, index: int, value: Any) -> QuestionValidationError:
        return cls(f"Each key point in entry #{index} must be a non-empty string (found: {value!r})")

    @classmethod
    def file_not_found(cls, path: Path) -> QuestionValidationError:
        return cls(f"Questions file not found: {path}")

    @classmethod
    def invalid_file(cls) -> QuestionValidationError:
        return cls("Questions file must be a non-empty JSON array")


def _validate_entry(entry: dict[str, Any], seen_ids: set[str], index: int) -> None:
    if not isinstance(entry, dict):
        raise QuestionValidationError.invalid_entry(index)

    for field in REQUIRED_FIELDS:
        if field not in entry:
            raise QuestionValidationError.missing_field(index, field)
        value = entry[field]
        if not isinstance(value, str) or not value.strip():
            raise QuestionValidationError.empty_field(index, field)

    entry_id = entry["id"]
    if entry_id in seen_ids:
        raise QuestionValidationError.duplicate_id(entry_id, index)
    seen_ids.add(entry_id)

    key_points = entry.get("key_points", []) or []
    if not isinstance(key_points, list):
        raise QuestionValidationError.invalid_key_points(index)
    for point in key_points:
        if not isinstance(point, str) or not point.strip():
            raise QuestionValidationError.invalid_key_point_value(index, point)


def validate_file(path: Path) -> int:
    if not path.exists():
        raise QuestionValidationError.file_not_found(path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise QuestionValidationError.invalid_file()

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

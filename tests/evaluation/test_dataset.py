from __future__ import annotations

from pathlib import Path

from genai_companion_with_ace.evaluation.dataset import (
    EvaluationDataset,
    ensure_default_dataset,
    generate_synthetic_dataset,
)


def test_generate_synthetic_dataset_has_minimum_questions() -> None:
    dataset = generate_synthetic_dataset(samples_per_course=6)
    assert len(dataset) >= 100
    courses = {example.course for example in dataset}
    assert courses == {str(i) for i in range(1, 17)}


def test_ensure_default_dataset_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "eval" / "dataset.json"
    dataset = ensure_default_dataset(path)
    assert path.exists()
    reloaded = EvaluationDataset.load(path)
    assert len(reloaded.examples) == len(dataset.examples)


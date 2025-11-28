from __future__ import annotations

from pathlib import Path

from genai_companion_with_ace.evaluation import (
    EvaluationDataset,
    EvaluationEngine,
    EvaluationExample,
    EvaluationRubric,
    save_metrics_report,
)


def test_save_metrics_report_creates_json_and_markdown(tmp_path: Path) -> None:
    example = EvaluationExample(
        id="ex",
        question="What is AI?",
        course="1",
        module="AI",
        difficulty="basic",
        question_type="conceptual",
        golden_answer="AI is the study of intelligent agents.",
    )
    dataset = EvaluationDataset([example])
    engine = EvaluationEngine(EvaluationRubric())

    def answer_provider(ex: EvaluationExample):
        return {"answer": "AI is the study of intelligent agents.", "contexts": []}

    results = engine.evaluate(dataset, answer_provider)
    report_path = save_metrics_report(engine, results, tmp_path, report_name="test_report")

    assert report_path.exists()
    markdown = tmp_path / "test_report.md"
    assert markdown.exists()


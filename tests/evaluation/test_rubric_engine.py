from __future__ import annotations

from genai_companion_with_ace.evaluation import (
    EvaluationDataset,
    EvaluationEngine,
    EvaluationExample,
    EvaluationRubric,
)


def test_evaluation_engine_scores_answers_without_contexts() -> None:
    example = EvaluationExample(
        id="sample",
        question="Explain AI history.",
        course="1",
        module="AI history",
        difficulty="basic",
        question_type="conceptual",
        golden_answer="AI history covers symbolic AI and modern machine learning.",
        evaluation_criteria={"must_include": ["history"], "should_cite": ["Course 1"]},
    )
    dataset = EvaluationDataset([example])
    engine = EvaluationEngine(EvaluationRubric())

    def answer_provider(ex: EvaluationExample):
        return {
            "answer": "AI history spans symbolic approaches and machine learning progress.",
            "contexts": [],
        }

    results = engine.evaluate(dataset, answer_provider)
    assert results
    aggregate = engine.aggregate(results)
    assert "answer_relevancy" in aggregate
    assert 0.0 <= aggregate["answer_relevancy"] <= 1.0

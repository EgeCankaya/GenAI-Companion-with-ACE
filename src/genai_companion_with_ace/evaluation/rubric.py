"""Evaluation rubric leveraging RAGAS where available."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

try:  # pragma: no cover - optional heavy dependency
    from datasets import Dataset  # type: ignore[import-untyped]
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import AnswerCorrectness, AnswerRelevancy, ContextRelevance, Faithfulness  # ContextRelevance not ContextRelevancy

    _RAGAS_AVAILABLE = True
except Exception:  # pragma: no cover
    Dataset = None  # type: ignore[assignment,misc]
    ragas_evaluate = None  # type: ignore[assignment,misc]
    AnswerCorrectness = None  # type: ignore[assignment,misc]
    AnswerRelevancy = None  # type: ignore[assignment,misc]
    ContextRelevance = None  # type: ignore[assignment,misc]
    Faithfulness = None  # type: ignore[assignment,misc]
    _RAGAS_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EvaluationRubric:
    """Scores answers against a golden reference."""

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "answer_relevancy": 0.3,
            "answer_correctness": 0.3,
            "context_relevancy": 0.2,
            "faithfulness": 0.2,
        }
    )

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str] | None,
        golden_answer: str,
    ) -> dict[str, float]:
        if _RAGAS_AVAILABLE and contexts:
            return self._evaluate_with_ragas(question, answer, contexts, golden_answer)
        return self._simple_overlap(question, answer, golden_answer)

    def _evaluate_with_ragas(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        golden_answer: str,
    ) -> dict[str, float]:
        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [golden_answer],
        })
        metrics = [
            AnswerRelevancy(),
            AnswerCorrectness(),
            ContextRelevance(),  # Fixed: ContextRelevance not ContextRelevancy
            Faithfulness(),
        ]
        result = ragas_evaluate(dataset, metrics=metrics)
        # Extract scores from result - result is an EvaluationResult with a results dict
        scores: dict[str, float] = {}
        for metric in metrics:
            metric_name = metric.__class__.__name__.lower()
            # Handle both dict-like and object-like result structures
            if isinstance(result, dict) and metric_name in result:
                metric_scores = result[metric_name]
                if metric_scores and len(metric_scores) > 0:
                    scores[metric_name] = float(metric_scores[0])
            elif hasattr(result, "results") and metric_name in result.results:  # type: ignore[attr-defined]
                metric_scores = result.results[metric_name]  # type: ignore[attr-defined,index]
                if metric_scores and len(metric_scores) > 0:
                    scores[metric_name] = float(metric_scores[0])
        return scores

    def _simple_overlap(self, question: str, answer: str, golden_answer: str) -> dict[str, float]:
        """Fallback scoring based on keyword overlap."""

        def _score(reference: str, candidate: str) -> float:
            ref_tokens = set(reference.lower().split())
            cand_tokens = set(candidate.lower().split())
            if not ref_tokens:
                return 0.0
            return len(ref_tokens & cand_tokens) / len(ref_tokens)

        relevancy = _score(question, answer)
        correctness = _score(golden_answer, answer)
        return {
            "answer_relevancy": relevancy,
            "answer_correctness": correctness,
            "context_relevancy": relevancy,
            "faithfulness": correctness,
        }

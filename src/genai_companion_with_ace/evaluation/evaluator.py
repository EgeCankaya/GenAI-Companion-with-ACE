"""Evaluation engine to apply rubric scoring over datasets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable

from genai_companion_with_ace.evaluation.dataset import EvaluationDataset, EvaluationExample
from genai_companion_with_ace.evaluation.rubric import EvaluationRubric

GenerateAnswerFn = Callable[[EvaluationExample], dict[str, object]]


@dataclass(slots=True)
class EvaluationResult:
    example_id: str
    scores: dict[str, float]
    raw_answer: str


class EvaluationEngine:
    def __init__(self, rubric: EvaluationRubric) -> None:
        self.rubric = rubric

    def evaluate(
        self,
        dataset: EvaluationDataset,
        answer_provider: GenerateAnswerFn,
    ) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        for example in dataset.examples:
            payload = answer_provider(example)
            answer = str(payload.get("answer", ""))
            contexts = payload.get("contexts") or []
            if not isinstance(contexts, list):
                contexts = [str(contexts)]
            scores = self.rubric.evaluate(
                question=example.question,
                answer=answer,
                contexts=[str(ctx) for ctx in contexts],
                golden_answer=example.golden_answer,
            )
            results.append(EvaluationResult(example_id=example.id, scores=scores, raw_answer=answer))
        return results

    @staticmethod
    def aggregate(results: Iterable[EvaluationResult]) -> dict[str, float]:
        totals: dict[str, float] = {}
        count = 0
        for result in results:
            count += 1
            for metric, score in result.scores.items():
                totals.setdefault(metric, 0.0)
                totals[metric] += score
        if count == 0:
            return dict.fromkeys(totals, 0.0)
        return {metric: value / count for metric, value in totals.items()}

    def summarize(self, results: Iterable[EvaluationResult]) -> dict[str, float]:
        aggregated = self.aggregate(results)
        weighted = 0.0
        total_weight = 0.0
        for metric, weight in self.rubric.weights.items():
            score = aggregated.get(metric, 0.0)
            weighted += score * weight
            total_weight += weight
        if total_weight > 0:
            aggregated["weighted_score"] = weighted / total_weight
        else:
            aggregated["weighted_score"] = 0.0
        return aggregated


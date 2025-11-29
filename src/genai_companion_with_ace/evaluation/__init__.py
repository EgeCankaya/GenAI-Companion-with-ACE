"""Evaluation utilities for the IBM Gen AI Companion."""

from .dataset import EvaluationDataset, EvaluationExample, ensure_default_dataset, generate_synthetic_dataset
from .evaluator import EvaluationEngine, EvaluationResult
from .reporting import save_metrics_report
from .rubric import EvaluationRubric

__all__ = [
    "EvaluationDataset",
    "EvaluationEngine",
    "EvaluationExample",
    "EvaluationResult",
    "EvaluationRubric",
    "ensure_default_dataset",
    "generate_synthetic_dataset",
    "save_metrics_report",
]

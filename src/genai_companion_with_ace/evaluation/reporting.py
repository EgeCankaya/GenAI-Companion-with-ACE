"""Utilities to persist evaluation metrics."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from genai_companion_with_ace.evaluation.evaluator import EvaluationEngine, EvaluationResult


def save_metrics_report(
    engine: EvaluationEngine,
    results: Iterable[EvaluationResult],
    output_dir: Path,
    report_name: str = "evaluation",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = engine.summarize(results)
    json_path = output_dir / f"{report_name}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown_path = output_dir / f"{report_name}.md"
    lines = ["# Evaluation Summary", ""]
    for metric, value in summary.items():
        lines.append(f"- **{metric.replace('_', ' ').title()}**: {value:.3f}")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path


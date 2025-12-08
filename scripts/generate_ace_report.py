"""Generate comprehensive evaluation reports for ACE runs."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_REPORT_DIR = PROJECT_ROOT / "outputs" / "ace_runs"
PLAYBOOK_DIR = PROJECT_ROOT / "outputs" / "ace_playbooks"


def parse_run_report(report_path: Path) -> dict[str, Any]:
    """Parse an ACE run report markdown file."""
    content = report_path.read_text(encoding="utf-8")
    data: dict[str, Any] = {
        "report_path": report_path,
        "timestamp": report_path.stem.replace("run_", ""),
    }

    # Extract basic info
    dataset_match = re.search(r"\*\*Dataset:\*\* (.+)", content)
    if dataset_match:
        data["dataset"] = dataset_match.group(1).strip()

    entries_match = re.search(r"\*\*Dataset entries:\*\* (\d+)", content)
    if entries_match:
        data["dataset_entries"] = int(entries_match.group(1))

    iterations_match = re.search(r"\*\*Iterations:\*\* (\d+)", content)
    if iterations_match:
        data["iterations"] = int(iterations_match.group(1))

    status_match = re.search(r"\*\*Status:\*\* (.+)", content)
    if status_match:
        data["status"] = status_match.group(1).strip()

    playbook_match = re.search(r"\*\*Playbook:\*\* (.+)", content)
    if playbook_match:
        data["playbook_path"] = playbook_match.group(1).strip()

    error_match = re.search(r"\*\*Error:\*\* (.+)", content, re.DOTALL)
    if error_match:
        data["error"] = error_match.group(1).strip()

    # Extract performance metrics
    metrics_section = re.search(r"## Performance Metrics\n(.*?)(?=\n\n|\Z)", content, re.DOTALL)
    if metrics_section:
        metrics_text = metrics_section.group(1)
        metrics: dict[str, Any] = {}
        for line in metrics_text.split("\n"):
            match = re.search(r"\*\*(\w+)\*\*: (.+)", line)
            if match:
                key = match.group(1)
                value_str = match.group(2).strip()
                # Try to parse as number
                try:
                    if "." in value_str:
                        metrics[key] = float(value_str)
                    else:
                        metrics[key] = int(value_str)
                except ValueError:
                    metrics[key] = value_str
        data["metrics"] = metrics

    # Extract metadata section
    metadata_section = re.search(r"## Playbook Metadata\n(.*?)(?=\n##|\Z)", content, re.DOTALL)
    if metadata_section:
        metadata_text = metadata_section.group(1)
        metadata: dict[str, Any] = {}
        for line in metadata_text.split("\n"):
            match = re.search(r"\*\*(\w+)\*\*: (.+)", line)
            if match:
                key = match.group(1)
                value_str = match.group(2).strip()
                # Try to parse as dict (for performance_metrics)
                if value_str.startswith("{") and value_str.endswith("}"):
                    try:
                        metadata[key] = eval(value_str)  # noqa: S307
                    except Exception:
                        metadata[key] = value_str
                else:
                    metadata[key] = value_str
        data["metadata"] = metadata

    return data


def get_latest_run_report() -> Path | None:
    """Get the most recent run report file."""
    reports = sorted(RUN_REPORT_DIR.glob("run_*.md"))
    return reports[-1] if reports else None


def get_previous_reports(limit: int = 5) -> list[dict[str, Any]]:
    """Get previous run reports for comparison."""
    reports = sorted(RUN_REPORT_DIR.glob("run_*.md"))
    previous: list[dict[str, Any]] = []
    for report_path in reports[-limit:]:
        try:
            data = parse_run_report(report_path)
            if data.get("status") == "success" and "metrics" in data:
                previous.append(data)
        except Exception:
            continue
    return previous


def validate_playbook_structure(playbook_path: Path) -> list[str]:
    """Quick validation of playbook structure issues."""
    issues: list[str] = []
    try:
        with playbook_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return ["Failed to parse playbook YAML"]

    context = data.get("context", {})
    heuristics = context.get("heuristics", [])

    # Check for duplicate IDs
    heuristic_ids: set[str] = set()
    for heuristic in heuristics:
        if isinstance(heuristic, dict):
            h_id = heuristic.get("id")
            if h_id:
                if h_id in heuristic_ids:
                    issues.append(f"Duplicate heuristic ID: {h_id}")
                else:
                    heuristic_ids.add(h_id)

    # Check for repetitive instructions
    system_instructions = context.get("system_instructions", "")
    if isinstance(system_instructions, str):
        lines = [line.strip() for line in system_instructions.split("\n") if line.strip()]
        seen: dict[str, int] = {}
        for line in lines:
            seen[line] = seen.get(line, 0) + 1
        for line, count in seen.items():
            if count > 2:
                issues.append(f"Instruction line repeated {count} times: '{line[:50]}...'")

    return issues


def generate_report(run_report_path: Path | None = None) -> str:
    """Generate a comprehensive evaluation report."""
    if run_report_path is None:
        run_report_path = get_latest_run_report()
        if run_report_path is None:
            return "No run reports found."

    current_run = parse_run_report(run_report_path)
    previous_runs = get_previous_reports(limit=5)

    # Filter out current run from previous
    previous_runs = [r for r in previous_runs if r["report_path"] != run_report_path]

    lines: list[str] = []
    lines.append("# ACE Run Evaluation Report")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Run Report:** `{run_report_path.name}`")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    status = current_run.get("status", "unknown")
    if status == "success":
        lines.append("✅ **Status: SUCCESS** - The ACE iteration cycle completed successfully.")
    elif status == "error":
        lines.append("❌ **Status: ERROR** - The ACE iteration cycle failed.")
        if "error" in current_run:
            lines.append(f"**Error:** {current_run['error']}")
    else:
        lines.append(f"⚠️ **Status: {status.upper()}**")

    if "playbook_path" in current_run:
        playbook_path = PROJECT_ROOT / current_run["playbook_path"]
        if playbook_path.exists():
            issues = validate_playbook_structure(playbook_path)
            if issues:
                lines.append("")
                lines.append("⚠️ **Playbook Issues Detected:**")
                for issue in issues:
                    lines.append(f"- {issue}")

    lines.append("")

    # Run Details
    lines.append("## Run Details")
    lines.append("")
    if "dataset_entries" in current_run:
        lines.append(f"- **Dataset:** {current_run.get('dataset_entries', 'N/A')} entries")
    if "iterations" in current_run:
        lines.append(f"- **Iterations:** {current_run.get('iterations', 'N/A')}")
    if "playbook_path" in current_run:
        lines.append(f"- **Playbook:** {current_run['playbook_path']}")
    if "metadata" in current_run:
        metadata = current_run["metadata"]
        if "parent_version" in metadata:
            lines.append(f"- **Parent Version:** {metadata['parent_version']}")
        if "iteration" in metadata:
            lines.append(f"- **Iteration Completed:** {metadata['iteration']}")
    lines.append("")

    # Performance Metrics
    if "metrics" in current_run:
        lines.append("## Performance Metrics Analysis")
        lines.append("")
        metrics = current_run["metrics"]

        # Primary metrics (semantic similarity, token counts, inference time)
        lines.append("### Primary Metrics (Educational Content Focus)")
        lines.append("")
        lines.append("| Metric | Value | Status |")
        lines.append("|--------|-------|--------|")

        semantic_sim = metrics.get("semantic_similarity", 0)
        if semantic_sim >= 0.5:
            status_icon = "✅"
        elif semantic_sim >= 0.4:
            status_icon = "⚠️"
        else:
            status_icon = "❌"
        lines.append(f"| **Semantic Similarity** | {semantic_sim:.3f} | {status_icon} |")

        avg_tokens = metrics.get("avg_tokens", 0)
        if 100 <= avg_tokens <= 200:
            status_icon = "✅"
        else:
            status_icon = "⚠️"
        lines.append(f"| **Avg Tokens** | {avg_tokens} | {status_icon} |")

        inference_time = metrics.get("inference_time_sec", 0)
        if inference_time <= 10:
            status_icon = "✅"
        elif inference_time <= 20:
            status_icon = "⚠️"
        else:
            status_icon = "❌"
        lines.append(f"| **Inference Time** | {inference_time:.1f}s | {status_icon} |")

        accuracy = metrics.get("accuracy", 0)
        if accuracy >= 0.9:
            status_icon = "✅"
        else:
            status_icon = "⚠️"
        lines.append(f"| **Accuracy** | {accuracy:.1%} | {status_icon} |")

        lines.append("")

        # Secondary metrics (BLEU/ROUGE - low signal for educational content)
        lines.append("### Secondary Metrics (Low Signal for Educational Content)")
        lines.append("")
        lines.append("> **Note:** BLEU and ROUGE scores are typically low for instructional/creative content")
        lines.append("> where exact matches are rare. These metrics are provided for reference but should")
        lines.append("> not be the primary evaluation criteria.")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| **BLEU Score** | {metrics.get('bleu_score', 0):.6f} |")
        lines.append(f"| **ROUGE Score** | {metrics.get('rouge_score', 0):.6f} |")
        lines.append(f"| **Exact Match** | {metrics.get('exact_match', 0):.1%} |")
        lines.append("")

        # Trend Analysis
        if previous_runs and len(previous_runs) > 0:
            lines.append("### Trend Analysis")
            lines.append("")
            prev_metrics = previous_runs[-1].get("metrics", {})
            lines.append("| Metric | Current | Previous | Change |")
            lines.append("|--------|---------|----------|--------|")

            for metric_name in ["semantic_similarity", "avg_tokens", "inference_time_sec", "accuracy"]:
                current_val = metrics.get(metric_name, 0)
                prev_val = prev_metrics.get(metric_name, 0)
                if prev_val != 0:
                    change = ((current_val - prev_val) / prev_val) * 100
                    change_str = f"{change:+.1f}%"
                    if abs(change) < 1:
                        change_icon = "➡️"
                    elif change > 0:
                        change_icon = "📈"
                    else:
                        change_icon = "📉"
                    lines.append(
                        f"| {metric_name.replace('_', ' ').title()} | {current_val:.3f} | {prev_val:.3f} | {change_icon} {change_str} |"
                    )
                else:
                    lines.append(f"| {metric_name.replace('_', ' ').title()} | {current_val:.3f} | N/A | - |")
            lines.append("")

    # Convergence Status
    if "metadata" in current_run:
        metadata = current_run["metadata"]
        convergence = metadata.get("convergence_status", "")
        if convergence:
            lines.append("## Convergence Status")
            lines.append("")
            if convergence == "degraded":
                lines.append("⚠️ **Status: DEGRADED**")
                lines.append("")
                lines.append("The playbook convergence is marked as degraded. This may indicate:")
                lines.append("- The playbook has reached a local optimum")
                lines.append("- Evaluation metrics may not be appropriate for this content type")
                lines.append("- The dataset may need more diversity")
                lines.append("- Consider adjusting ACE iteration parameters")
            else:
                lines.append(f"✅ **Status: {convergence.upper()}**")
            lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    if "metrics" in current_run:
        metrics = current_run["metrics"]
        semantic_sim = metrics.get("semantic_similarity", 0)
        if semantic_sim < 0.4:
            lines.append("- ⚠️ **Low semantic similarity** - Consider reviewing playbook heuristics and examples")
        if semantic_sim < 0.5 and previous_runs:
            prev_sim = previous_runs[-1].get("metrics", {}).get("semantic_similarity", 0)
            if semantic_sim < prev_sim:
                lines.append("- 📉 **Declining semantic similarity** - Review recent playbook changes")

    if "playbook_path" in current_run:
        playbook_path = PROJECT_ROOT / current_run["playbook_path"]
        if playbook_path.exists():
            issues = validate_playbook_structure(playbook_path)
            if issues:
                lines.append(
                    "- 🔧 **Fix playbook structure issues** - Run `python scripts/validate_playbook.py` to identify and fix issues"
                )

    if "metadata" in current_run and current_run["metadata"].get("convergence_status") == "degraded":
        lines.append("- 🔄 **Review ACE parameters** - Consider adjusting iteration count or evaluation criteria")

    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comprehensive ACE evaluation report")
    parser.add_argument(
        "--run-report",
        type=Path,
        help="Path to specific run report (default: latest)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: outputs/ace_runs/ACE_EVALUATION_REPORT_<timestamp>.md)",
    )

    args = parser.parse_args()

    run_report_path = args.run_report
    if run_report_path is None:
        run_report_path = get_latest_run_report()
        if run_report_path is None:
            print("❌ No run reports found.", file=sys.stderr)
            sys.exit(1)

    if not run_report_path.exists():
        print(f"❌ Run report not found: {run_report_path}", file=sys.stderr)
        sys.exit(1)

    report_content = generate_report(run_report_path)

    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RUN_REPORT_DIR / f"ACE_EVALUATION_REPORT_{timestamp}.md"

    output_path.write_text(report_content, encoding="utf-8")
    print(f"✅ Evaluation report generated: {output_path}")


if __name__ == "__main__":
    main()

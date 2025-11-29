"""Run ACE improvement cycles using a prepared dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from genai_companion_with_ace.ace_integration import ACETriggerConfig, PlaybookLoader  # noqa: E402
from genai_companion_with_ace.ace_integration.ace_trigger import ACERunnerAdapter  # noqa: E402
from genai_companion_with_ace.config import CompanionConfig  # noqa: E402

DEFAULT_CONFIG = Path("configs/companion_config.yaml")
RUN_REPORT_DIR = Path("outputs/ace_runs")


class DatasetLoadError(RuntimeError):
    """Raised when the ACE dataset file is missing or malformed."""


class DatasetFileNotFound(DatasetLoadError):
    def __init__(self, path: Path) -> None:
        super().__init__(f"Dataset file not found: {path}")
        self.path = path


class InvalidDatasetError(DatasetLoadError):
    def __init__(self) -> None:
        super().__init__("Dataset must be a non-empty JSON array.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACE iterations using a prepared dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to ACE dataset JSON (output of build_ace_dataset.py)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to companion config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of ACE iterations to run (default: 1)",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DatasetFileNotFound(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise InvalidDatasetError()
    return data


def load_trigger_config(config_path: Path, iterations: int) -> tuple[ACETriggerConfig, PlaybookLoader]:
    companion_config = CompanionConfig.from_file(config_path)
    ace_cfg = companion_config.ace_config()
    repo_path = Path(ace_cfg.get("repository_path", "../Agentic-Context-Engineering")).resolve()
    playbook_dir = companion_config.outputs.ace_playbooks
    config_path_value = ace_cfg.get("config_path")
    resolved_config_path: Path | None = None
    if config_path_value:
        raw_path = Path(config_path_value)
        if not raw_path.is_absolute():
            raw_path = (config_path.parent / raw_path).resolve()
        resolved_config_path = raw_path

    trigger_config = ACETriggerConfig(
        repo_path=repo_path,
        playbook_output_dir=playbook_dir,
        iterations=iterations,
        trigger_threshold=ace_cfg.get("trigger_threshold", 50),
        config_path=resolved_config_path,
    )
    loader = PlaybookLoader(playbook_dir)
    return trigger_config, loader


def rename_playbook(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target = path.parent / f"playbook_auto_{timestamp}.yaml"
    if path != target:
        shutil.move(path, target)
        return target
    return path


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_playbook_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # pylint: disable=broad-except
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload.get("metadata", {})


def write_report(
    *,
    dataset_path: Path,
    dataset_count: int,
    dataset_hash: str,
    iterations: int,
    status: str,
    playbook_path: Path | None = None,
    playbook_metadata: dict[str, Any] | None = None,
    error: str | None = None,
) -> Path:
    RUN_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = RUN_REPORT_DIR / f"run_{timestamp}.md"

    lines = [
        f"# ACE Run Report - {timestamp}",
        "",
        f"- **Dataset:** {dataset_path}",
        f"- **Dataset entries:** {dataset_count}",
        f"- **Dataset SHA256:** `{dataset_hash}`",
        f"- **Iterations:** {iterations}",
        f"- **Status:** {status}",
    ]
    if playbook_path:
        lines.append(f"- **Playbook:** {playbook_path}")
    if error:
        lines.append(f"- **Error:** {error}")

    if playbook_metadata:
        lines.extend(["", "## Playbook Metadata"])
        for key, value in playbook_metadata.items():
            lines.append(f"- **{key}**: {value}")

        perf = playbook_metadata.get("performance_metrics", {})
        if perf:
            lines.extend(["", "## Performance Metrics"])
            for metric, value in perf.items():
                lines.append(f"- **{metric}**: {value}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    try:
        dataset = load_dataset(args.dataset)
    except DatasetLoadError as exc:
        print(f"Invalid dataset: {exc}", file=sys.stderr)
        sys.exit(1)
    dataset_count = len(dataset)
    dataset_hash = dataset_sha256(args.dataset)
    trigger_config, loader = load_trigger_config(args.config, args.iterations)

    try:
        runner = ACERunnerAdapter(trigger_config)
    except Exception as exc:  # pylint: disable=broad-except
        report = write_report(
            dataset_path=args.dataset,
            dataset_count=dataset_count,
            dataset_hash=dataset_hash,
            iterations=args.iterations,
            status="error",
            error=f"Failed to initialize ACE runner: {exc}",
        )
        print(f"Failed to initialize ACE runner: {exc}", file=sys.stderr)
        print(f"Run report written to {report}", file=sys.stderr)
        sys.exit(1)
    try:
        new_playbook = runner.run(
            dataset,
            iterations=args.iterations,
            playbook_output_dir=trigger_config.playbook_output_dir,
        )
    except Exception as exc:  # pylint: disable=broad-except
        report = write_report(
            dataset_path=args.dataset,
            dataset_count=dataset_count,
            dataset_hash=dataset_hash,
            iterations=args.iterations,
            status="error",
            error=str(exc),
        )
        print(f"ACE run failed: {exc}", file=sys.stderr)
        print(f"Run report written to {report}", file=sys.stderr)
        sys.exit(1)

    if not new_playbook:
        report = write_report(
            dataset_path=args.dataset,
            dataset_count=dataset_count,
            dataset_hash=dataset_hash,
            iterations=args.iterations,
            status="no_playbook",
            error="ACE runner did not return a playbook",
        )
        print("ACE runner did not produce a new playbook. Existing playbook retained.")
        print(f"Run report written to {report}")
        sys.exit(0)

    final_path = rename_playbook(new_playbook)
    latest_context = loader.load_latest()
    playbook_metadata = load_playbook_metadata(final_path)

    report = write_report(
        dataset_path=args.dataset,
        dataset_count=dataset_count,
        dataset_hash=dataset_hash,
        iterations=args.iterations,
        status="success",
        playbook_path=final_path,
        playbook_metadata=playbook_metadata,
    )

    print("ACE improvement cycle completed.")
    print(f"- Dataset: {args.dataset} ({dataset_count} entries)")
    print(f"- Iterations: {args.iterations}")
    print(f"- New playbook: {final_path}")
    print(f"- Loaded version: {latest_context.version}")
    print(f"- Run report: {report}")


if __name__ == "__main__":
    main()

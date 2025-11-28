"""Trigger ACE improvement cycles using logged conversations."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@contextlib.contextmanager
def _in_directory(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


from genai_companion_with_ace.ace_integration.conversation_logger import ConversationLogger
from genai_companion_with_ace.ace_integration.playbook_loader import PlaybookContext, PlaybookLoader

LOGGER = logging.getLogger(__name__)


class RunnerProtocol(Protocol):
    def run(self, dataset: list[dict], iterations: int, playbook_output_dir: Path) -> Path | None:
        ...


@dataclass(slots=True)
class ACETriggerConfig:
    repo_path: Path
    playbook_output_dir: Path
    config_path: Path | None = None
    iterations: int = 1
    trigger_threshold: int = 50


class ACERunnerAdapter:
    """Thin wrapper around the ACE runner to enable dependency injection in tests."""

    def __init__(self, config: ACETriggerConfig) -> None:
        try:
            from agentic_context_engineering.runners.ace_runner import ACERunner  # Lazy import
            from agentic_context_engineering.utils import llm_client  # type: ignore
        except ImportError as e:
            error_msg = (
                f"ACE framework not available. Import error: {e}\n"
                "Make sure 'agentic-context-engineering' is installed or the repository path is correct."
            )
            raise ImportError(error_msg) from e

        # Patch the ACE LLM client to allow CPU-only environments by skipping CUDA checks.
        if not hasattr(llm_client.LLMClient, "_genai_cpu_patch"):
            def _skip_gpu_check(self):  # type: ignore
                return None

            llm_client.LLMClient._verify_gpu = _skip_gpu_check  # type: ignore[attr-defined]
            llm_client.LLMClient._genai_cpu_patch = True  # type: ignore[attr-defined]

        config_path = str(config.config_path) if config.config_path else None
        self._repo_path = Path(config.repo_path).resolve()
        try:
            with _in_directory(self._repo_path):
                self._runner = ACERunner(config_path=config_path)
        except Exception as e:
            error_msg = (
                f"Failed to initialize ACE runner: {e}\n"
                f"Check that the ACE repository path is correct: {config.repo_path}"
            )
            raise RuntimeError(error_msg) from e
        self._iterations = config.iterations

    def run(self, dataset: list[dict], iterations: int, playbook_output_dir: Path) -> Path | None:
        """Run ACE cycles and ensure playbook is in the project's output directory."""
        if not dataset:
            message = "ACE dataset must contain at least one conversation turn"
            raise ValueError(message)
        try:
            tasks = [item["input"] for item in dataset]
            with _in_directory(self._repo_path):
                results = self._runner.run_iterations(
                    num_iterations=iterations, tasks=tasks, evaluation_dataset=dataset
                )
            raw_playbook = results.get("final_playbook")

            if raw_playbook is None:
                return None

            from agentic_context_engineering.playbook_schema import Playbook  # type: ignore

            if isinstance(raw_playbook, str):
                ace_playbook_path = Path(raw_playbook)
            elif isinstance(raw_playbook, Playbook):
                ace_playbook_path = (
                    self._repo_path / "outputs" / f"playbook_v{raw_playbook.version}.yaml"
                )
                ace_playbook_path.parent.mkdir(parents=True, exist_ok=True)
                raw_playbook.to_yaml(str(ace_playbook_path))
            else:
                raise RuntimeError(f"Unsupported playbook result type: {type(raw_playbook)!r}")

            if not ace_playbook_path.is_absolute():
                ace_playbook_path = self._repo_path / ace_playbook_path
            
            # Ensure output directory exists
            playbook_output_dir.mkdir(parents=True, exist_ok=True)
            
            # If ACE generated playbook is not in our output directory, copy it there
            if ace_playbook_path.parent != playbook_output_dir:
                target_path = playbook_output_dir / ace_playbook_path.name
                LOGGER.info("Copying ACE-generated playbook from %s to %s", ace_playbook_path, target_path)
                shutil.copy2(ace_playbook_path, target_path)
                return target_path
            
            return ace_playbook_path
        except Exception as e:
            error_msg = (
                f"Error during ACE cycle execution: {e}\n"
                "This may indicate issues with the ACE framework configuration or conversation data format."
            )
            raise RuntimeError(error_msg) from e


def run_ace_cycles(
    *,
    conversation_logger: ConversationLogger,
    playbook_loader: PlaybookLoader,
    trigger_config: ACETriggerConfig,
    runner: RunnerProtocol | None = None,
) -> PlaybookContext:
    """Execute ACE improvement cycles and reload the resulting playbook.
    
    Generated playbooks are stored in the project's outputs/ace_playbooks directory,
    making them self-contained within this project.
    """
    dataset = conversation_logger.export_for_ace()
    if not dataset:
        message = "No conversations available to trigger ACE"
        raise ValueError(message)

    LOGGER.info("Running ACE cycles with %s conversation turns", len(dataset))
    runner = runner or ACERunnerAdapter(trigger_config)
    
    # Pass the playbook output directory to ensure playbooks are stored locally
    new_playbook = runner.run(
        dataset, 
        iterations=trigger_config.iterations,
        playbook_output_dir=trigger_config.playbook_output_dir
    )

    if new_playbook:
        LOGGER.info("ACE generated playbook at %s (stored in project directory)", new_playbook)
    else:
        LOGGER.warning("ACE did not produce a new playbook; reusing previous version.")

    return playbook_loader.load_latest()


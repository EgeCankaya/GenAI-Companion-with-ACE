from __future__ import annotations

from pathlib import Path

import yaml
from langchain_core.documents import Document

from genai_companion_with_ace.ace_integration import ACETriggerConfig, PlaybookLoader, run_ace_cycles
from genai_companion_with_ace.ace_integration.conversation_logger import ConversationLogger
from genai_companion_with_ace.rag.retrieval import RetrievalResult, RetrievedChunk


def write_playbook(path: Path, version: str, instructions: str) -> None:
    payload = {
        "version": version,
        "context": {
            "system_instructions": instructions,
            "heuristics": ["Always cite sources."],
            "examples": [],
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def build_retrieval_result() -> RetrievalResult:
    doc = Document(page_content="Sample content", metadata={"source": "Course 1", "course": "1", "module": "Intro"})
    chunk = RetrievedChunk(document=doc, score=1.0, source="dense")
    return RetrievalResult(combined=[chunk], dense_results=[chunk], keyword_results=[], attachment_chunks=[])


class FakeRunner:
    def __init__(self, target_dir: Path) -> None:
        self._target_dir = target_dir

    def run(self, dataset, iterations: int, playbook_output_dir: Path) -> Path:
        new_playbook = playbook_output_dir / "playbook_v99.yaml"
        write_playbook(new_playbook, "9.9.9", "Improved instructions")
        return new_playbook


def test_run_ace_cycles_with_fake_runner(tmp_path: Path) -> None:
    conversations_dir = tmp_path / "logs"
    playbook_dir = tmp_path / "playbooks"
    playbook_dir.mkdir()

    write_playbook(playbook_dir / "playbook_v1.yaml", "1.0.0", "Base instructions")

    logger = ConversationLogger(conversations_dir)
    logger.log_turn(
        session_id="session-1",
        question="What is AI?",
        answer="Artificial Intelligence is the field of building intelligent systems.",
        retrieval_result=build_retrieval_result(),
        metadata={"course": "1"},
    )

    loader = PlaybookLoader(playbook_dir=playbook_dir)
    config = ACETriggerConfig(repo_path=tmp_path, playbook_output_dir=playbook_dir, iterations=1)
    runner = FakeRunner(playbook_dir)

    context = run_ace_cycles(
        conversation_logger=logger,
        playbook_loader=loader,
        trigger_config=config,
        runner=runner,
    )

    assert context.version == "9.9.9"

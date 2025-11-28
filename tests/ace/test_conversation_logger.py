from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from genai_companion_with_ace.ace_integration.conversation_logger import ConversationLogger
from genai_companion_with_ace.rag.retrieval import RetrievalResult, RetrievedChunk


def build_retrieval_result() -> RetrievalResult:
    doc = Document(page_content="Sample content", metadata={"source": "Course 1", "course": "1", "module": "Intro"})
    chunk = RetrievedChunk(document=doc, score=1.0, source="dense")
    return RetrievalResult(combined=[chunk], dense_results=[chunk], keyword_results=[], attachment_chunks=[])


def test_conversation_logger_round_trip(tmp_path: Path) -> None:
    logger = ConversationLogger(tmp_path)
    retrieval_result = build_retrieval_result()

    logger.log_turn(
        session_id="session-1",
        question="What is AI?",
        answer="Artificial Intelligence is the field of building intelligent systems.",
        retrieval_result=retrieval_result,
        metadata={"course": "1"},
    )

    exported = logger.export_for_ace()
    assert exported
    assert exported[0]["input"].startswith("What is AI?")
    assert exported[0]["sources"][0]["course"] == "1"


def test_count_logged_turns(tmp_path: Path) -> None:
    """Test that count_logged_turns correctly counts all logged conversation turns."""
    logger = ConversationLogger(tmp_path / "logs")

    # Initially should be 0
    assert logger.count_logged_turns() == 0

    # Log a few turns
    for i in range(5):
        logger.log_turn(
            session_id=f"session-{i % 2}",  # Alternate between 2 sessions
            question=f"Question {i}",
            answer=f"Answer {i}",
            retrieval_result=build_retrieval_result(),
            metadata={"course": "1"},
        )

    # Should count all 5 turns
    assert logger.count_logged_turns() == 5

    # Log a few more
    for i in range(3):
        logger.log_turn(
            session_id="session-2",
            question=f"Question {i + 5}",
            answer=f"Answer {i + 5}",
            retrieval_result=build_retrieval_result(),
            metadata={"course": "2"},
        )

    # Should now count 8 total turns
    assert logger.count_logged_turns() == 8


def test_turn_counter_persistence(tmp_path: Path) -> None:
    logger = ConversationLogger(tmp_path / "logs")
    for i in range(3):
        logger.log_turn(
            session_id="session-cache",
            question=f"Question {i}",
            answer=f"Answer {i}",
            retrieval_result=build_retrieval_result(),
            metadata={"course": "cache"},
        )
    assert logger.count_logged_turns() == 3

    # Recreate logger to ensure counter cache is reused
    new_logger = ConversationLogger(tmp_path / "logs")
    assert new_logger.count_logged_turns() == 3


def test_log_rotation(tmp_path: Path) -> None:
    logger = ConversationLogger(tmp_path, rotate_after_bytes=200)
    for i in range(20):
        logger.log_turn(
            session_id="session-rotate",
            question=f"Question {i}",
            answer=f"Answer {i}",
            retrieval_result=build_retrieval_result(),
            metadata={"course": "rotate"},
        )

    rotated_files = [path for path in tmp_path.glob("session-rotate*.jsonl") if path.name != "session-rotate.jsonl"]
    assert rotated_files, "Expected rotation to create archived log files."


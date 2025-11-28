from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from genai_companion_with_ace.chat.conversation import ConversationManager, SessionHistoryStore
from genai_companion_with_ace.chat.modes import ConversationMode
from genai_companion_with_ace.chat.query_processor import AttachmentInput, QueryProcessor


def build_manager(tmp_path: Path) -> ConversationManager:
    store = SessionHistoryStore(tmp_path / "history.db")
    return ConversationManager(store, max_context_turns=3)


def test_query_processor_creates_session_and_classifies_course(tmp_path: Path) -> None:
    manager = build_manager(tmp_path)
    retrieval = MagicMock()
    processor = QueryProcessor(manager, retrieval, default_mode=ConversationMode.STUDY)

    processed = processor.process(
        session_id="session-1",
        user_input="How does backpropagation work in deep learning?",
    )

    assert processed.course == "9"
    assert processed.mode == ConversationMode.STUDY
    assert processed.context_window[-1].content.startswith("How does backpropagation")

    # Ensure attachments are converted and history is reused
    second = processor.process(
        session_id="session-1",
        user_input="Explain how gradients flow in the network.",
        attachments=[AttachmentInput(name="notes.txt", content="Gradient notes")],
        mode=ConversationMode.QUICK,
    )

    assert len(second.context_window) >= 2
    assert second.attachments and second.attachments[0].name == "notes.txt"
    assert second.mode == ConversationMode.QUICK


def test_query_processor_sets_detail_metadata(tmp_path: Path) -> None:
    manager = build_manager(tmp_path)
    retrieval = MagicMock()
    processor = QueryProcessor(manager, retrieval, default_mode=ConversationMode.STUDY)

    processed = processor.process(
        session_id="detail-session",
        user_input="Please explain parameter-efficient fine-tuning in detail with a deep dive.",
    )

    assert processed.metadata.get("detail_level") == "deep"

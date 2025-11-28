from __future__ import annotations

from pathlib import Path

from genai_companion_with_ace.chat.conversation import ConversationManager, SessionHistoryStore


def test_persistent_conversation_history(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    store = SessionHistoryStore(db_path)
    manager = ConversationManager(store, max_context_turns=2)

    session = manager.start_session(session_id="session-1", user_id="student-123", course="5", mode="study")
    assert session.session_id == "session-1"

    manager.append_user_message("What is backpropagation?")
    manager.append_assistant_message("Backpropagation computes gradients.")
    manager.append_user_message("Can you give more detail?")
    manager.append_assistant_message("Sure, it propagates errors backwards.")

    recent_context = manager.get_recent_context()
    assert len(recent_context) == 4  # two turns retained
    assert recent_context[0].content == "What is backpropagation?"

    history = store.get_messages("session-1")
    assert len(history) == 4  # persisted with truncation applied

    # Ensure persistence by creating a new manager
    new_store = SessionHistoryStore(db_path)
    new_manager = ConversationManager(new_store, max_context_turns=2)
    loaded = new_manager.resume_session("session-1")
    assert len(loaded) == 4
    assert loaded[-1].content.startswith("Sure")

    sessions = new_manager.list_sessions()
    assert sessions and sessions[0].session_id == "session-1"

    new_manager.delete_session("session-1")
    assert not new_manager.list_sessions()


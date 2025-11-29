"""Conversation management and persistent history for the AI companion."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from genai_companion_with_ace.utils.time import utcnow_isoformat

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Message:
    """Represents a single conversational turn."""

    role: str
    content: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConversationSession:
    """Metadata for a persisted conversation session."""

    session_id: str
    created_at: str
    user_id: str | None = None
    course: str | None = None
    mode: str | None = None
    summary: str | None = None


class SessionHistoryStore:
    """SQLite-backed persistence for conversation history."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    user_id TEXT,
                    course TEXT,
                    mode TEXT,
                    summary TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    session_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT,
                    PRIMARY KEY (session_id, idx),
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            connection.commit()

    def create_session(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        course: str | None = None,
        mode: str | None = None,
    ) -> ConversationSession:
        created_at = utcnow_isoformat()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO sessions(session_id, created_at, user_id, course, mode)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, created_at, user_id, course, mode),
            )
            connection.commit()
        LOGGER.info("Created conversation session %s", session_id)
        return ConversationSession(
            session_id=session_id, created_at=created_at, user_id=user_id, course=course, mode=mode
        )

    def append_message(self, session_id: str, message: Message) -> None:
        with self._connection() as connection:
            max_idx = connection.execute(
                "SELECT COALESCE(MAX(idx), -1) FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            new_idx = max_idx + 1
            connection.execute(
                """
                INSERT INTO messages(session_id, idx, role, content, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    new_idx,
                    message.role,
                    message.content,
                    message.created_at,
                    json.dumps(message.metadata, ensure_ascii=False),
                ),
            )
            connection.commit()

    def get_messages(self, session_id: str) -> list[Message]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT role, content, created_at, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY idx ASC
                """,
                (session_id,),
            ).fetchall()
        messages: list[Message] = []
        for row in rows:
            metadata = json.loads(row[3]) if row[3] else {}
            messages.append(Message(role=row[0], content=row[1], created_at=row[2], metadata=metadata))
        return messages

    def truncate_session(self, session_id: str, max_turns: int) -> None:
        if max_turns <= 0:
            return
        with self._connection() as connection:
            indices = [
                row[0]
                for row in connection.execute(
                    "SELECT idx FROM messages WHERE session_id = ? ORDER BY idx ASC",
                    (session_id,),
                ).fetchall()
            ]
            if len(indices) <= max_turns:
                return
            cutoff_value = indices[-max_turns]
            connection.execute(
                "DELETE FROM messages WHERE session_id = ? AND idx < ?",
                (session_id, cutoff_value),
            )
            connection.commit()

    def list_sessions(self, limit: int = 20) -> list[ConversationSession]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT session_id, created_at, user_id, course, mode, summary
                FROM sessions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            ConversationSession(
                session_id=row[0],
                created_at=row[1],
                user_id=row[2],
                course=row[3],
                mode=row[4],
                summary=row[5],
            )
            for row in rows
        ]

    def get_session(self, session_id: str) -> ConversationSession | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT session_id, created_at, user_id, course, mode, summary
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return ConversationSession(
            session_id=row[0],
            created_at=row[1],
            user_id=row[2],
            course=row[3],
            mode=row[4],
            summary=row[5],
        )

    def delete_session(self, session_id: str) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            connection.commit()

    def upsert_summary(self, session_id: str, summary: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE sessions SET summary = ? WHERE session_id = ?",
                (summary, session_id),
            )
            connection.commit()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self._db_path))
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()


class ConversationManager:
    """High-level conversation manager coordinating memory and retrieval context."""

    def __init__(self, store: SessionHistoryStore, *, max_context_turns: int = 12) -> None:
        self._store = store
        self._max_context_turns = max_context_turns
        self._active_session: str | None = None

    def start_session(
        self,
        *,
        session_id: str,
        user_id: str | None = None,
        course: str | None = None,
        mode: str | None = None,
    ) -> ConversationSession:
        session = self._store.create_session(session_id, user_id=user_id, course=course, mode=mode)
        self._active_session = session.session_id
        return session

    def resume_session(self, session_id: str) -> list[Message]:
        self._active_session = session_id
        return self._store.get_messages(session_id)

    def append_user_message(self, content: str, metadata: dict[str, Any] | None = None) -> Message:
        return self._append_message("user", content, metadata)

    def append_assistant_message(self, content: str, metadata: dict[str, Any] | None = None) -> Message:
        return self._append_message("assistant", content, metadata)

    def get_recent_context(self, session_id: str | None = None) -> list[Message]:
        active = session_id or self._active_session
        if not active:
            return []
        messages = self._store.get_messages(active)
        if self._max_context_turns <= 0:
            return messages
        return messages[-self._max_context_turns * 2 :]

    def get_session(self, session_id: str) -> ConversationSession | None:
        return self._store.get_session(session_id)

    def list_sessions(self, limit: int = 20) -> list[ConversationSession]:
        return self._store.list_sessions(limit=limit)

    def delete_session(self, session_id: str) -> None:
        self._store.delete_session(session_id)
        if self._active_session == session_id:
            self._active_session = None

    def summarize_session(self, session_id: str, summary: str) -> None:
        self._store.upsert_summary(session_id, summary)

    def _append_message(self, role: str, content: str, metadata: dict[str, Any] | None) -> Message:
        if not self._active_session:
            message = "No active session; call start_session() first"
            raise RuntimeError(message)
        message = Message(role=role, content=content, created_at=utcnow_isoformat(), metadata=metadata or {})
        self._store.append_message(self._active_session, message)
        self._store.truncate_session(self._active_session, self._max_context_turns * 2)
        return message

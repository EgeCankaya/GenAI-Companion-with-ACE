"""Conversation logging utilities for ACE integration."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from genai_companion_with_ace.utils.time import utcnow_isoformat

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from genai_companion_with_ace.rag.retrieval import RetrievalResult


@dataclass(slots=True)
class LoggedTurn:
    session_id: str
    question: str
    answer: str
    retrieved_sources: list[dict[str, str]]
    metadata: dict[str, str]
    timestamp: str


class ConversationLogger:
    """Persist conversation turns and prepare datasets for ACE improvement cycles."""

    def __init__(self, output_dir: Path, *, rotate_after_bytes: int = 5 * 1024 * 1024) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._counter_path = self._output_dir / ".turn_counts.json"
        self._turn_count: int | None = None
        self._rotate_after_bytes = rotate_after_bytes
        self._load_counter()

    def log_turn(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        retrieval_result: RetrievalResult | None = None,
        metadata: dict[str, str] | None = None,
    ) -> LoggedTurn:
        timestamp = utcnow_isoformat()
        sources = self._serialize_sources(retrieval_result)
        record = LoggedTurn(
            session_id=session_id,
            question=question,
            answer=answer,
            retrieved_sources=sources,
            metadata=dict(metadata or {}),
            timestamp=timestamp,
        )
        session_file = self._append_record(record)
        self._increment_counter()
        self._rotate_if_needed(session_file)
        return record

    def count_logged_turns(self) -> int:
        """Count the total number of logged conversation turns."""
        if self._turn_count is not None:
            return self._turn_count
        count = sum(1 for _ in self._iter_records())
        self._turn_count = count
        self._persist_counter()
        return count

    def export_for_ace(self, limit: int | None = None) -> list[dict[str, str]]:
        """Export a list of conversation turns formatted for ACE training."""
        exports: list[dict[str, str]] = []
        for record in self._iter_records():
            exports.append({
                "session_id": record["session_id"],
                "input": record["question"],
                "output": record["answer"],
                "sources": record.get("retrieved_sources", []),
                "metadata": record.get("metadata", {}),
            })
            if limit is not None and len(exports) >= limit:
                break
        return exports

    def _append_record(self, record: LoggedTurn) -> Path:
        session_file = self._output_dir / f"{record.session_id}.jsonl"
        payload = {
            "session_id": record.session_id,
            "question": record.question,
            "answer": record.answer,
            "retrieved_sources": record.retrieved_sources,
            "metadata": record.metadata,
            "timestamp": record.timestamp,
        }
        with session_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False))
            file.write("\n")
        return session_file

    def _iter_records(self) -> Iterable[dict[str, Any]]:
        for path in sorted(self._output_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        LOGGER.warning("Skipping malformed log line in %s", path)

    def _rotate_if_needed(self, session_file: Path) -> None:
        if self._rotate_after_bytes <= 0 or not session_file.exists():
            return
        if session_file.stat().st_size < self._rotate_after_bytes:
            return
        timestamp = utcnow_isoformat().replace(":", "").replace("-", "")
        archived = session_file.with_name(f"{session_file.stem}.{timestamp}{session_file.suffix}")
        try:
            session_file.rename(archived)
            LOGGER.info("Rotated conversation log %s -> %s", session_file.name, archived.name)
        except Exception as exc:  # pragma: no cover - filesystem dependent
            LOGGER.warning("Failed to rotate log %s: %s", session_file, exc)

    def _load_counter(self) -> None:
        if not self._counter_path.exists():
            return
        try:
            data = json.loads(self._counter_path.read_text(encoding="utf-8"))
            self._turn_count = int(data.get("total", 0))
        except Exception:  # pragma: no cover - corrupt cache
            self._turn_count = None

    def _increment_counter(self) -> None:
        self._turn_count = (self._turn_count or 0) + 1
        self._persist_counter()

    def _persist_counter(self) -> None:
        if self._turn_count is None:
            return
        payload = {"total": self._turn_count}
        try:
            self._counter_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:  # pragma: no cover - filesystem dependent
            LOGGER.warning("Failed to persist conversation turn counter to %s", self._counter_path)

    @staticmethod
    def _serialize_sources(retrieval_result: RetrievalResult | None) -> list[dict[str, str]]:
        if retrieval_result is None:
            return []
        sources: list[dict[str, str]] = []
        for chunk in retrieval_result.combined:
            metadata = chunk.document.metadata
            sources.append({
                "source": metadata.get("source") or metadata.get("document_id", "unknown"),
                "course": metadata.get("course", ""),
                "module": metadata.get("module", ""),
            })
        return sources

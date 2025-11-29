"""Query processing utilities for the AI companion."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from genai_companion_with_ace.chat.conversation import ConversationManager, ConversationSession, Message
from genai_companion_with_ace.chat.modes import ConversationMode
from genai_companion_with_ace.rag import RetrievalAttachment, RetrievalOrchestrator

COURSE_KEYWORDS: dict[str, Sequence[str]] = {
    "1": ("artificial intelligence", "ai basics", "history of ai"),
    "2": ("generative ai", "gen ai applications"),
    "3": ("prompt engineering", "prompt design"),
    "4": ("python for data science", "jupyter", "pandas basics"),
    "5": ("flask", "api development", "rest api"),
    "6": ("ai applications", "generative ai python"),
    "7": ("data analysis", "matplotlib", "data visualization"),
    "8": ("machine learning", "scikit-learn", "supervised learning"),
    "9": ("deep learning", "keras", "neural network", "backpropagation"),
    "10": ("llm architecture", "data preparation"),
    "11": ("foundational models", "language understanding"),
    "12": ("transformer", "self-attention"),
    "13": ("fine-tuning transformers", "parameter efficient"),
    "14": ("advanced fine-tuning", "lora", "qlora"),
    "15": ("rag", "langchain", "ai agents"),
    "16": ("capstone", "project", "rag pipeline"),
}

DETAIL_KEYWORDS: tuple[str, ...] = (
    "in detail",
    "deep dive",
    "comprehensive",
    "explain thoroughly",
    "long answer",
    "detailed explanation",
    "full explanation",
)


@dataclass(slots=True)
class AttachmentInput:
    """User-provided attachment destined for retrieval."""

    name: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ProcessedQuery:
    session: ConversationSession
    user_message: Message
    context_window: list[Message]
    course: str | None
    mode: ConversationMode
    attachments: list[RetrievalAttachment]
    metadata: dict[str, str]


class CourseClassifier:
    """Lightweight heuristic course classifier."""

    pattern_cache: ClassVar[dict[str, re.Pattern[str]]] = {}

    @classmethod
    def classify(cls, text: str) -> str | None:
        normalized = text.lower()
        for course_id, keywords in COURSE_KEYWORDS.items():
            for keyword in keywords:
                pattern = cls.pattern_cache.get(keyword)
                if pattern is None:
                    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                    cls.pattern_cache[keyword] = pattern
                if pattern.search(normalized):
                    return course_id
        return None


class QueryProcessor:
    """Prepares user questions for retrieval and generation."""

    def __init__(
        self,
        conversation_manager: ConversationManager,
        retrieval: RetrievalOrchestrator,
        *,
        default_mode: ConversationMode = ConversationMode.STUDY,
    ) -> None:
        self._conversation_manager = conversation_manager
        self._retrieval = retrieval
        self._default_mode = default_mode

    def process(
        self,
        *,
        session_id: str,
        user_input: str,
        attachments: Iterable[AttachmentInput] | None = None,
        mode: ConversationMode | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ProcessedQuery:
        metadata = dict(metadata or {})
        mode = mode or self._default_mode
        course = metadata.get("course") or CourseClassifier.classify(user_input)
        metadata.setdefault("course", course or "unknown")
        metadata.setdefault("mode", mode.value)

        normalized_input = user_input.lower()
        if "detail_level" not in metadata and any(keyword in normalized_input for keyword in DETAIL_KEYWORDS):
            metadata["detail_level"] = "deep"

        session_meta = self._conversation_manager.get_session(session_id)
        if session_meta is None:
            session_meta = self._conversation_manager.start_session(
                session_id=session_id,
                course=course,
                mode=mode.value,
            )
        else:
            self._conversation_manager.resume_session(session_id)

        user_message = self._conversation_manager.append_user_message(user_input, metadata=metadata)
        processed_attachments = self._to_retrieval_attachments(attachments)
        context_window = self._conversation_manager.get_recent_context(session_id)

        return ProcessedQuery(
            session=session_meta,
            user_message=user_message,
            context_window=context_window,
            course=course,
            mode=mode,
            attachments=processed_attachments,
            metadata=metadata,
        )

    def _to_retrieval_attachments(
        self,
        attachments: Iterable[AttachmentInput] | None,
    ) -> list[RetrievalAttachment]:
        if not attachments:
            return []
        converted: list[RetrievalAttachment] = []
        for item in attachments:
            converted.append(
                RetrievalAttachment(
                    name=item.name,
                    content=item.content,
                    metadata=item.metadata,
                )
            )
        return converted

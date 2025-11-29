"""Response formatting utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from genai_companion_with_ace.chat.modes import ConversationMode, get_mode_settings


@dataclass(slots=True, frozen=True)
class Citation:
    source: str
    snippet: str | None = None
    metadata: dict | None = None


@dataclass(slots=True, frozen=True)
class FormattedResponse:
    mode: ConversationMode
    answer: str
    citations: Sequence[Citation]
    follow_ups: Sequence[str]
    disclaimer: str | None = None

    def render(self, include_citations: bool = False, include_followups: bool = True) -> str:
        """
        Render the formatted response.

        Args:
            include_citations: Whether to include the Sources section at the end.
                              Defaults to False since inline citations are already in the answer.
            include_followups: Whether to include suggested follow-up questions.
        """
        lines: list[str] = []
        if self.disclaimer:
            lines.append(self.disclaimer)
        lines.append(self.answer.strip())

        # Only include Sources section if explicitly requested
        # Inline citations are already present in the answer text
        if include_citations and self.citations:
            lines.append("\nSources:")
            for idx, citation in enumerate(self.citations, start=1):
                entry = f"[{idx}] {citation.source}"
                if citation.snippet:
                    entry += f" — {citation.snippet.strip()}"
                lines.append(entry)

        if include_followups and self.follow_ups:
            lines.append("\nSuggested follow-ups:")
            lines.extend(f"- {item}" for item in self.follow_ups)

        return "\n".join(lines).strip()


class ResponseFormatter:
    """Formats model responses according to the current conversation mode."""

    def format_answer(
        self,
        answer: str,
        *,
        citations: Iterable[Citation] = (),
        mode: ConversationMode = ConversationMode.STUDY,
        follow_ups: Sequence[str] | None = None,
        disclaimer: str | None = None,
    ) -> FormattedResponse:
        settings = get_mode_settings(mode)
        processed_answer = answer.strip()

        if settings.hint_only:
            processed_answer = self._truncate_to_hint(processed_answer)
        elif settings.concise:
            processed_answer = self._ensure_concise(processed_answer)

        if settings.include_objectives and not processed_answer.lower().startswith("learning objectives"):
            processed_answer = self._prepend_objectives(processed_answer)

        citation_list = list(citations)
        return FormattedResponse(
            mode=mode,
            answer=processed_answer,
            citations=citation_list,
            follow_ups=list(follow_ups or []),
            disclaimer=disclaimer,
        )

    def format_fallback(
        self,
        *,
        mode: ConversationMode,
        reason: str,
        follow_up_suggestion: str | None = None,
    ) -> FormattedResponse:
        base_message = "I do not have enough information in the indexed IBM course materials to answer that question."
        if reason:
            base_message += f" Reason: {reason}."
        follow_ups = [follow_up_suggestion] if follow_up_suggestion else []
        disclaimer = "This assistant relies on IBM Generative AI Professional Certificate content."
        return FormattedResponse(
            mode=mode,
            answer=base_message,
            citations=[],
            follow_ups=follow_ups,
            disclaimer=disclaimer,
        )

    @staticmethod
    def _truncate_to_hint(answer: str) -> str:
        sentences = answer.split(". ")
        return ". ".join(sentences[:2]).strip() + ("..." if len(sentences) > 2 else "")

    @staticmethod
    def _ensure_concise(answer: str) -> str:
        # For quick mode, keep it concise but don't truncate mid-sentence
        # Just return the answer as-is - the LLM will generate concise responses
        return answer

    @staticmethod
    def _prepend_objectives(answer: str) -> str:
        objectives_header = "Learning objectives:\n- Understand the concept.\n- Review related examples.\n\n"
        return objectives_header + answer

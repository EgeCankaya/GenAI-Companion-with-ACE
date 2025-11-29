"""Answer generation pipeline that combines RAG context with ACE playbook guidance."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import requests

from genai_companion_with_ace.ace_integration import ACETriggerConfig, run_ace_cycles
from genai_companion_with_ace.ace_integration.conversation_logger import ConversationLogger
from genai_companion_with_ace.ace_integration.playbook_loader import PlaybookContext, PlaybookLoader
from genai_companion_with_ace.chat.conversation import ConversationManager
from genai_companion_with_ace.chat.response_formatter import Citation, FormattedResponse, ResponseFormatter
from genai_companion_with_ace.rag.retrieval import RetrievalOrchestrator, RetrievalResult

LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":  # pragma: no cover
    from genai_companion_with_ace.chat.query_processor import ProcessedQuery


class GenerationError(Exception):
    """Raised when the LLM generation fails."""


@dataclass(slots=True)
class GenerationConfig:
    temperature: float = 0.2
    max_tokens: int = 4096  # Increased to allow longer, more thorough responses without truncation
    citation_top_k: int = 5
    include_history: bool = True
    default_deep_dive_sections: int = 8
    default_target_words: int = 600


@dataclass(slots=True)
class OutlineSection:
    title: str
    bullets: list[str]


@dataclass(slots=True)
class OutlineContext:
    sections: list[OutlineSection]
    outline_text: str
    key_points: list[str]
    target_words: int


@dataclass(slots=True)
class GenerationParams:
    max_tokens: int
    temperature: float


class AnswerGenerator:
    """Generates grounded answers using retrieved context and ACE playbooks."""

    def __init__(
        self,
        *,
        llm_client: Any,
        retrieval: RetrievalOrchestrator,
        conversation_manager: ConversationManager,
        formatter: ResponseFormatter,
        playbook: PlaybookContext,
        conversation_logger: ConversationLogger | None = None,
        playbook_loader: PlaybookLoader | None = None,
        trigger_config: ACETriggerConfig | None = None,
        config: GenerationConfig | None = None,
        deep_dive_llm_client=None,
    ) -> None:
        self._llm_client = llm_client
        self._deep_llm_client = deep_dive_llm_client
        self._retrieval = retrieval
        self._conversation_manager = conversation_manager
        self._formatter = formatter
        self._playbook = playbook
        self._config = config or GenerationConfig()
        self._logger = conversation_logger
        self._playbook_loader = playbook_loader
        self._trigger_config = trigger_config
        self._trigger_state_path = (
            trigger_config.playbook_output_dir / ".ace_trigger_state.json" if trigger_config else None
        )
        self._last_trigger_count = self._load_trigger_state()
        self._last_retrieval: RetrievalResult | None = None

    def generate(self, processed_query: ProcessedQuery) -> FormattedResponse:
        retrieval_result = self._perform_retrieval(processed_query)
        detail_requested = self._is_detail_requested(processed_query)
        outline_context = self._build_outline_context(
            processed_query,
            retrieval_result,
            detail_requested,
        )
        prompt = self._build_prompt(
            processed_query,
            retrieval_result,
            detail_requested=detail_requested,
            outline_text=outline_context.outline_text,
            key_points=outline_context.key_points,
            target_words=outline_context.target_words if detail_requested else None,
        )
        params = self._resolve_generation_params(detail_requested)
        self._log_prompt_stats(processed_query, detail_requested, params, prompt)
        answer = self._invoke_llm(
            prompt,
            processed_query,
            outline_context,
            params,
            detail_requested,
        )

        citations = self._build_citations(retrieval_result)
        formatted = self._formatter.format_answer(
            answer,
            citations=citations,
            mode=processed_query.mode,
        )

        rendered_answer = formatted.render()
        self._conversation_manager.append_assistant_message(
            rendered_answer,
            metadata={
                "citations": [citation.source for citation in citations],
                "course": processed_query.course or "unknown",
                "mode": processed_query.mode.value,
            },
        )
        if detail_requested:
            word_count = len(rendered_answer.split())
            LOGGER.debug(
                "Deep-dive response stats: words=%d chars=%d sections=%d citations=%d",
                word_count,
                len(rendered_answer),
                rendered_answer.count("\n\n"),
                len(citations),
            )

        if self._logger:
            # Track which heuristics were used in this generation
            heuristic_ids = self._playbook.get_heuristic_ids()
            metadata = {
                "course": processed_query.course or "unknown",
                "mode": processed_query.mode.value,
                "playbook_version": self._playbook.version,
                "heuristic_ids": ",".join(heuristic_ids) if heuristic_ids else "",
            }
            self._logger.log_turn(
                session_id=processed_query.session.session_id,
                question=processed_query.user_message.content,
                answer=rendered_answer,
                retrieval_result=retrieval_result,
                metadata=metadata,
            )
            # Check if we should trigger ACE cycles automatically
            self._check_and_trigger_ace_cycles()

        return formatted

    def _perform_retrieval(self, processed_query: ProcessedQuery) -> RetrievalResult:
        retrieval_result = self._retrieval.retrieve(
            processed_query.user_message.content,
            attachments=processed_query.attachments,
        )
        self._last_retrieval = retrieval_result
        return retrieval_result

    @staticmethod
    def _is_detail_requested(processed_query: ProcessedQuery) -> bool:
        return processed_query.metadata.get("detail_level") == "deep"

    def _build_outline_context(
        self,
        processed_query: ProcessedQuery,
        retrieval_result: RetrievalResult,
        detail_requested: bool,
    ) -> OutlineContext:
        target_words = self._resolve_target_words(processed_query)
        if not detail_requested:
            return OutlineContext([], "", [], target_words)
        key_points = self._summarize_retrieval_for_outline(retrieval_result)
        outline_sections = self._generate_outline(processed_query, key_points)
        outline_text = self._format_outline_for_prompt(outline_sections)
        return OutlineContext(outline_sections, outline_text, key_points, target_words)

    def _resolve_generation_params(self, detail_requested: bool) -> GenerationParams:
        max_tokens = self._config.max_tokens
        if detail_requested:
            max_tokens = int(max_tokens * 1.5)
        temperature = 0.25 if detail_requested else self._config.temperature
        return GenerationParams(max_tokens=max_tokens, temperature=temperature)

    def _log_prompt_stats(
        self,
        processed_query: ProcessedQuery,
        detail_requested: bool,
        params: GenerationParams,
        prompt: str,
    ) -> None:
        LOGGER.debug(
            "Deep-dive=%s session=%s mode=%s max_tokens=%s temperature=%s prompt_len=%d chars",
            detail_requested,
            processed_query.session.session_id,
            processed_query.mode.value,
            params.max_tokens,
            params.temperature,
            len(prompt),
        )

    def _invoke_llm(
        self,
        prompt: str,
        processed_query: ProcessedQuery,
        outline_context: OutlineContext,
        params: GenerationParams,
        detail_requested: bool,
    ) -> str:
        llm_client = self._select_llm(detail_requested)
        try:
            answer = llm_client.generate(
                prompt,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
            )
        except requests.exceptions.ConnectionError as exc:
            raise GenerationError(self._connection_error_message(exc)) from exc
        except Exception as exc:  # pragma: no cover - LLM errors
            raise GenerationError(self._llm_error_message(exc)) from exc
        return self._continue_outline_if_needed(
            answer,
            processed_query,
            outline_context,
            params,
            detail_requested,
        )

    def _check_and_trigger_ace_cycles(self) -> None:
        """Check if threshold is reached and automatically trigger ACE cycles if needed."""
        if not self._logger or not self._playbook_loader or not self._trigger_config:
            return

        current_count = self._logger.count_logged_turns()
        threshold = self._trigger_config.trigger_threshold

        # Calculate the next trigger point (e.g., if threshold is 50, trigger at 50, 100, 150, etc.)
        next_trigger_point = ((self._last_trigger_count // threshold) + 1) * threshold

        # Only trigger if we've crossed the next trigger point
        if current_count >= next_trigger_point:
            LOGGER.info(
                "Conversation count (%d) reached threshold (%d). Triggering ACE improvement cycles...",
                current_count,
                threshold,
            )
            try:
                new_playbook = run_ace_cycles(
                    conversation_logger=self._logger,
                    playbook_loader=self._playbook_loader,
                    trigger_config=self._trigger_config,
                )
                self._playbook = new_playbook
                self._last_trigger_count = current_count
                self._persist_trigger_state()
                LOGGER.info(
                    "ACE cycles completed. Playbook updated to version %s. Next trigger at %d conversations.",
                    new_playbook.version,
                    next_trigger_point + threshold,
                )
            except Exception as exc:  # pragma: no cover - ACE errors
                LOGGER.error("Failed to run ACE cycles: %s", exc, exc_info=True)
                # Don't update last_trigger_count so we can retry on next turn

    def _load_trigger_state(self) -> int:
        if not self._trigger_state_path or not self._trigger_state_path.exists():
            return 0
        try:
            data = json.loads(self._trigger_state_path.read_text(encoding="utf-8"))
            return int(data.get("last_trigger_count", 0))
        except Exception:  # pragma: no cover - corrupt cache
            return 0

    def _persist_trigger_state(self) -> None:
        if not self._trigger_state_path:
            return
        payload = {"last_trigger_count": self._last_trigger_count}
        try:
            self._trigger_state_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:  # pragma: no cover - filesystem dependent
            LOGGER.warning("Failed to persist ACE trigger state to %s", self._trigger_state_path)

    def retrieval_debug_lines(self, top_k: int = 5) -> list[str]:
        if not self._last_retrieval:
            return []
        lines: list[str] = []
        for idx, chunk in enumerate(self._last_retrieval.combined[:top_k], start=1):
            metadata = chunk.document.metadata
            source = metadata.get("source") or metadata.get("document_id") or "unknown"
            course = metadata.get("course")
            module = metadata.get("module")
            qualifier = f" (Course {course} - {module})" if course and module else ""
            preview = chunk.document.page_content.replace("\n", " ")[:160]
            lines.append(f"{idx}. [{chunk.source}] score={chunk.score:.2f} source={source}{qualifier}")
            lines.append(f"    {preview}...")
        return lines

    def last_retrieval_contexts(self, top_k: int | None = None) -> list[str]:
        if not self._last_retrieval:
            return []
        chunks = self._last_retrieval.combined
        if top_k is not None:
            chunks = chunks[:top_k]
        return [chunk.document.page_content for chunk in chunks]

    def _build_prompt(
        self,
        processed_query: ProcessedQuery,
        retrieval_result: RetrievalResult,
        *,
        detail_requested: bool = False,
        outline_text: str = "",
        key_points: list[str] | None = None,
        target_words: int | None = None,
    ) -> str:
        sections: list[str] = []
        sections.append(self._playbook.to_prompt_block())
        sections.append(self._format_context(processed_query))
        sections.append(self._format_retrieval_chunks(retrieval_result, detail_requested=detail_requested))
        if outline_text:
            sections.append("Planned outline:\n" + outline_text)
        if key_points:
            sections.append("Key points synthesized from sources:\n" + "\n".join(f"- {point}" for point in key_points))
        sections.append(f"User question:\n{processed_query.user_message.content}")
        instructions = [
            "- Provide a comprehensive, well-structured answer that is related to and grounded in the retrieved sources.",
            "- You may use your own knowledge to enhance explanations, provide context, or clarify concepts, but keep the answer focused on the topics in the sources.",
            "- Cite sources inline using the format [Source: Course X - Module Y] when referencing specific information from the retrieved context.",
            "- If information is missing from the sources, you may provide additional context from your knowledge, but clearly distinguish between what's in the sources and what you're adding.",
        ]
        if detail_requested:
            instructions.append(
                "- Produce a deep dive: include an overview, key techniques, step-by-step reasoning, trade-offs, pitfalls, and concise code or pseudo-code where it clarifies the explanation. Conclude with a brief summary and suggested next steps."
            )
            instructions.append(
                "- Cite only retrieved course sources inline. If you add outside knowledge, label it as 'Additional context' and do not invent external papers or references."
            )
            sections.append(
                f"Target length: approximately {target_words or self._config.default_target_words} words. Ensure every outline section is fully developed."
            )
        else:
            instructions.append("- Ensure your answer thoroughly covers the topics mentioned in the sources.")
        sections.append("Instructions:\n" + "\n".join(instructions))
        return "\n\n".join(section for section in sections if section.strip())

    def _format_context(self, processed_query: ProcessedQuery) -> str:
        if not self._config.include_history or not processed_query.context_window:
            return ""
        formatted_turns = []
        for message in processed_query.context_window[-8:]:
            role = message.role.capitalize()
            formatted_turns.append(f"{role}: {message.content}")
        return "Conversation history:\n" + "\n".join(formatted_turns)

    def _format_retrieval_chunks(
        self,
        retrieval_result: RetrievalResult,
        *,
        detail_requested: bool = False,
    ) -> str:
        if not retrieval_result.combined:
            return "No supporting documents were retrieved. Respond based on known course materials."

        lines: list[str] = ["Retrieved context:"]
        max_chunks = self._config.citation_top_k * (2 if detail_requested else 1)
        for index, chunk in enumerate(retrieval_result.combined[:max_chunks], start=1):
            metadata = chunk.document.metadata
            source = metadata.get("source") or metadata.get("course") or "Unknown Source"
            lines.append(f"[{index}] Source: {source}\nContent: {chunk.document.page_content}\nMetadata: {metadata}")
        return "\n".join(lines)

    def _build_citations(self, retrieval_result: RetrievalResult) -> list[Citation]:
        citations: list[Citation] = []
        for chunk in retrieval_result.combined[: self._config.citation_top_k]:
            metadata = chunk.document.metadata
            course = metadata.get("course")
            module = metadata.get("module")
            source_label = metadata.get("source") or metadata.get("document_id") or "Course Material"
            if course and module:
                source_label = f"Course {course} - {module}"
            citations.append(
                Citation(
                    source=source_label,
                    snippet=chunk.document.page_content[:200]
                    + ("..." if len(chunk.document.page_content) > 200 else ""),
                    metadata=metadata,
                )
            )
        return citations

    @staticmethod
    def _missing_outline_sections(answer: str, sections: list[OutlineSection]) -> list[OutlineSection]:
        lowered = answer.lower()
        missing: list[OutlineSection] = []
        for section in sections:
            if section.title.lower() not in lowered:
                missing.append(section)
        return missing

    def _continue_outline_sections(
        self,
        answer: str,
        processed_query: ProcessedQuery,
        missing_sections: list[OutlineSection],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        continuation_prompt = (
            "Continue the deep-dive answer. Expand ONLY the remaining sections listed below. "
            "Maintain the established tone, cite sources inline, and avoid repeating completed sections.\n\n"
            f"Question: {processed_query.user_message.content}\n\n"
            "Remaining sections:\n"
            + "\n".join(
                f"- {section.title}: {', '.join(section.bullets) or 'Use best judgment'}"
                for section in missing_sections
            )
            + "\n\nDo not restate the introduction. Append the new sections so they flow naturally after the existing answer."
        )
        llm_client = self._select_llm(True)
        continuation = llm_client.generate(
            continuation_prompt,
            temperature=temperature,
            max_tokens=max(256, max_tokens),
        )
        result = answer.rstrip() + "\n\n" + continuation.strip()
        return str(result)  # Ensure we return str, not Any

    def _continue_outline_if_needed(
        self,
        answer: str,
        processed_query: ProcessedQuery,
        outline_context: OutlineContext,
        params: GenerationParams,
        detail_requested: bool,
    ) -> str:
        if not detail_requested or not outline_context.sections:
            return answer
        missing_sections = self._missing_outline_sections(answer, outline_context.sections)
        if not missing_sections:
            return answer
        return self._continue_outline_sections(
            answer,
            processed_query,
            missing_sections,
            temperature=params.temperature,
            max_tokens=max(256, params.max_tokens // 2),
        )

    def _select_llm(self, detail_requested: bool) -> Any:
        if detail_requested and self._deep_llm_client:
            return self._deep_llm_client
        return self._llm_client

    def _resolve_target_words(self, processed_query: ProcessedQuery) -> int:
        metadata_target = processed_query.metadata.get("target_words")
        if metadata_target and str(metadata_target).isdigit():
            return max(300, int(metadata_target))
        return self._config.default_target_words

    def _summarize_retrieval_for_outline(self, retrieval_result: RetrievalResult, max_points: int = 8) -> list[str]:
        points: list[str] = []
        for chunk in retrieval_result.combined[:max_points]:
            metadata = chunk.document.metadata
            source = metadata.get("source") or metadata.get("module") or "Course material"
            snippet = chunk.document.page_content.strip().replace("\n", " ")
            snippet = snippet[:220] + ("..." if len(snippet) > 220 else "")
            points.append(f"{source}: {snippet}")
        return points

    def _generate_outline(
        self,
        processed_query: ProcessedQuery,
        key_points: list[str],
    ) -> list[OutlineSection]:
        target_sections = self._config.default_deep_dive_sections
        prompt = (
            "Create a structured JSON outline for a deep-dive answer.\n"
            "Requirements:\n"
            f"- Provide {target_sections} sections (overview, key techniques, comparisons, code walkthrough, pitfalls, FAQs, summary, next steps, etc.).\n"
            "- Each section must include a concise title and 2-3 bullet highlights referencing the key points.\n"
            '- Respond ONLY with JSON matching: {"sections":[{"title":"...","bullets":["..."]}, ...]}.\n'
            "- Do not include markdown or explanations outside the JSON.\n\n"
            f"User question: {processed_query.user_message.content}\n\n"
            "Key points from retrieved sources:\n" + "\n".join(f"- {kp}" for kp in key_points)
        )
        llm = self._select_llm(True)
        response = llm.generate(prompt, temperature=0.35, max_tokens=1024)
        try:
            data: dict[str, Any] = json.loads(response)
            sections_raw = data.get("sections", [])
            outline: list[OutlineSection] = []
            for item in sections_raw[:target_sections]:
                title = str(item.get("title") or "").strip()
                bullets = [str(b).strip() for b in item.get("bullets", []) if str(b).strip()]
                if title:
                    outline.append(OutlineSection(title=title, bullets=bullets[:3]))
            if outline:
                return outline
        except Exception as exc:  # pragma: no cover - fallback parse
            LOGGER.debug("Failed to parse outline JSON; falling back to plain-text outline: %s", exc)
        # Fallback simple outline by splitting lines
        fallback_outline: list[OutlineSection] = []
        for _idx, line in enumerate(response.splitlines()):
            line = line.strip("-* \t")
            if not line:
                continue
            fallback_outline.append(OutlineSection(title=line, bullets=[]))
            if len(fallback_outline) >= target_sections:
                break
        if not fallback_outline:
            fallback_outline.append(OutlineSection(title="Overview", bullets=key_points[:3]))
        return fallback_outline

    @staticmethod
    def _format_outline_for_prompt(sections: list[OutlineSection]) -> str:
        lines: list[str] = []
        for idx, section in enumerate(sections, start=1):
            lines.append(f"{idx}. {section.title}")
            for bullet in section.bullets:
                lines.append(f"   - {bullet}")
        return "\n".join(lines)

    @staticmethod
    def _connection_error_message(exc: Exception) -> str:
        error_text = str(exc)
        if "localhost:11434" in error_text or "11434" in error_text:
            return (
                "❌ Ollama is not running!\n\n"
                "Please start Ollama before using the companion:\n"
                "  1. Open a terminal and run: ollama serve\n"
                "  2. Or start Ollama from your applications\n"
                "  3. Verify it's running: ollama list\n\n"
                "For more information, visit: https://ollama.com/"
            )
        return f"Failed to connect to LLM service: {exc}"

    def _llm_error_message(self, exc: Exception) -> str:
        error_str = str(exc)
        if "404" in error_str or "not found" in error_str.lower():
            model_hint = self._model_hint_from_error(error_str)
            return (
                "❌ Model not found!\n\n"
                "The required model is not installed in Ollama.\n\n"
                "To install the model:\n"
                f"  Run: ollama pull {model_hint}\n\n"
                "This will download the model (may take several minutes).\n"
                "After installation, try your question again.\n\n"
                "To see installed models: ollama list"
            )
        return f"Failed to generate answer: {exc}"

    @staticmethod
    def _model_hint_from_error(error_str: str) -> str:
        if "pull" in error_str:
            match = re.search(r"pull\s+([^\s`]+)", error_str)
            if match:
                return match.group(1)
        return "llama3.1:8b"

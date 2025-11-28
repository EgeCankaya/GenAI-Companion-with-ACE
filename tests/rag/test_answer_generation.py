from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document

from genai_companion_with_ace.ace_integration import ACETriggerConfig
from genai_companion_with_ace.ace_integration.conversation_logger import ConversationLogger
from genai_companion_with_ace.ace_integration.playbook_loader import PlaybookContext, PlaybookLoader
from genai_companion_with_ace.chat.conversation import ConversationManager, SessionHistoryStore
from genai_companion_with_ace.chat.modes import ConversationMode
from genai_companion_with_ace.chat.query_processor import ProcessedQuery
from genai_companion_with_ace.chat.response_formatter import ResponseFormatter
from genai_companion_with_ace.rag import AnswerGenerator, GenerationConfig
from genai_companion_with_ace.rag.retrieval import RetrievalResult, RetrievedChunk


class DummyLLM:
    def __init__(self) -> None:
        self.last_prompt: str | None = None
        self.last_kwargs: dict | None = None

    def generate(self, prompt: str, **kwargs) -> str:
        self.last_prompt = prompt
        self.last_kwargs = kwargs
        return (
            "Transformers use self-attention to weight tokens based on relevance. "
            "Source: Course 12 - Transformer Architecture."
        )


class ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, str | dict]] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        index = len(self.calls) - 1
        return self.responses[min(index, len(self.responses) - 1)]


def build_processed_query(
    manager: ConversationManager,
    *,
    metadata: dict[str, str] | None = None,
) -> ProcessedQuery:
    session = manager.start_session(session_id="session-1", course="12", mode=ConversationMode.STUDY.value)
    message = manager.append_user_message("Explain self-attention in transformers.")
    context = manager.get_recent_context()
    base_metadata = {"course": "12", "mode": "study"}
    if metadata:
        base_metadata.update(metadata)
    return ProcessedQuery(
        session=session,
        user_message=message,
        context_window=context,
        course="12",
        mode=ConversationMode.STUDY,
        attachments=[],
        metadata=base_metadata,
    )


def build_retrieval_result() -> RetrievalResult:
    document = Document(
        page_content="Self-attention computes weighted sums of values using softmax-normalized scores.",
        metadata={
            "course": "12",
            "module": "Transformer Architecture",
            "source": "Course 12 - Module 2",
            "document_id": "transformer-doc",
            "chunk_id": 0,
        },
    )
    chunk = RetrievedChunk(document=document, score=1.0, source="dense")
    return RetrievalResult(
        combined=[chunk],
        dense_results=[chunk],
        keyword_results=[],
        attachment_chunks=[],
    )


def test_answer_generator_formats_response(tmp_path: Path) -> None:
    store = SessionHistoryStore(tmp_path / "history.db")
    manager = ConversationManager(store, max_context_turns=3)

    playbook = PlaybookContext(
        version="test",
        system_instructions="You are a helpful study assistant.",
        heuristics=["Provide detailed answers.", "Always cite sources."],
        examples=[],
    )

    retrieval = MagicMock()
    retrieval.retrieve.return_value = build_retrieval_result()

    logger = ConversationLogger(tmp_path / "logs")

    generator = AnswerGenerator(
        llm_client=DummyLLM(),
        retrieval=retrieval,
        conversation_manager=manager,
        formatter=ResponseFormatter(),
        playbook=playbook,
        conversation_logger=logger,
        config=GenerationConfig(),
    )

    processed_query = build_processed_query(manager)
    formatted = generator.generate(processed_query)

    assert formatted.answer
    assert formatted.citations
    assert formatted.citations[0].source.startswith("Course 12")

    messages = store.get_messages("session-1")
    assert messages[-1].role == "assistant"
    assert "citations" in messages[-1].metadata

    exported = logger.export_for_ace()
    assert exported
    assert exported[0]["session_id"] == "session-1"


def test_answer_generator_expands_for_detail_requests(tmp_path: Path) -> None:
    store = SessionHistoryStore(tmp_path / "history.db")
    manager = ConversationManager(store, max_context_turns=3)

    playbook = PlaybookContext(
        version="test",
        system_instructions="You are a helpful study assistant.",
        heuristics=[],
        examples=[],
    )

    def build_detail_chunks() -> RetrievalResult:
        chunks = []
        for idx in range(6):
            content = f"Detail chunk {idx} with unique content."
            doc = Document(
                page_content=content,
                metadata={
                    "course": "12",
                    "module": f"Module {idx}",
                    "source": f"Source {idx}",
                },
            )
            chunks.append(RetrievedChunk(document=doc, score=1.0 - idx * 0.01, source="dense"))
        return RetrievalResult(
            combined=chunks,
            dense_results=chunks,
            keyword_results=[],
            attachment_chunks=[],
        )

    retrieval = MagicMock()
    retrieval.retrieve.return_value = build_detail_chunks()

    llm = DummyLLM()
    generator = AnswerGenerator(
        llm_client=llm,
        retrieval=retrieval,
        conversation_manager=manager,
        formatter=ResponseFormatter(),
        playbook=playbook,
        config=GenerationConfig(),
    )

    processed_query = build_processed_query(
        manager,
        metadata={"course": "12", "mode": "study", "detail_level": "deep"},
    )
    generator.generate(processed_query)

    assert llm.last_kwargs is not None
    assert llm.last_kwargs.get("max_tokens") == int(GenerationConfig().max_tokens * 1.5)
    assert "deep dive" in (llm.last_prompt or "").lower()
    assert "Detail chunk 5" in (llm.last_prompt or "")


def test_deep_dive_outline_and_target_words(tmp_path: Path) -> None:
    store = SessionHistoryStore(tmp_path / "history.db")
    manager = ConversationManager(store, max_context_turns=3)
    playbook = PlaybookContext(
        version="test",
        system_instructions="You are a helpful study assistant.",
        heuristics=[],
        examples=[],
    )

    retrieval = MagicMock()
    retrieval.retrieve.return_value = build_retrieval_result()

    scripted_llm = ScriptedLLM(
        [
            '{"sections":[{"title":"Overview","bullets":["context","goal"]},{"title":"Techniques","bullets":["Adapters","LoRA"]}]}',
            "Overview section content.\n\nTechniques section content.",
        ]
    )

    generator = AnswerGenerator(
        llm_client=scripted_llm,
        deep_dive_llm_client=scripted_llm,
        retrieval=retrieval,
        conversation_manager=manager,
        formatter=ResponseFormatter(),
        playbook=playbook,
        config=GenerationConfig(),
    )

    processed_query = build_processed_query(
        manager,
        metadata={"course": "12", "mode": "study", "detail_level": "deep", "target_words": "900"},
    )
    generator.generate(processed_query)

    assert len(scripted_llm.calls) >= 2
    assert "structured JSON outline" in scripted_llm.calls[0]["prompt"]
    assert "Target length" in scripted_llm.calls[1]["prompt"]


def test_automatic_ace_triggering(tmp_path: Path) -> None:
    """Test that ACE cycles are automatically triggered when threshold is reached."""
    store = SessionHistoryStore(tmp_path / "history.db")
    manager = ConversationManager(store, max_context_turns=3)

    playbook = PlaybookContext(
        version="1.0.0",
        system_instructions="You are a helpful study assistant.",
        heuristics=["Provide detailed answers."],
        examples=[],
    )

    retrieval = MagicMock()
    retrieval.retrieve.return_value = build_retrieval_result()

    logger = ConversationLogger(tmp_path / "logs")
    playbook_loader = MagicMock(spec=PlaybookLoader)
    playbook_loader.load_latest.return_value = playbook

    # Create a mock runner that simulates ACE cycles
    class MockACERunner:
        def __init__(self) -> None:
            self.call_count = 0

        def run(self, dataset: list[dict], iterations: int) -> Path | None:
            self.call_count += 1
            # Simulate creating a new playbook
            new_playbook_path = tmp_path / f"playbook_v{1.0 + self.call_count * 0.1}.yaml"
            new_playbook_path.touch()
            return new_playbook_path

    mock_runner = MockACERunner()

    trigger_config = ACETriggerConfig(
        repo_path=tmp_path,
        playbook_output_dir=tmp_path / "playbooks",
        trigger_threshold=3,  # Low threshold for testing
        iterations=1,
    )

    generator = AnswerGenerator(
        llm_client=DummyLLM(),
        retrieval=retrieval,
        conversation_manager=manager,
        formatter=ResponseFormatter(),
        playbook=playbook,
        conversation_logger=logger,
        playbook_loader=playbook_loader,
        trigger_config=trigger_config,
        config=GenerationConfig(),
    )

    # Generate 2 conversations - should not trigger yet
    for i in range(2):
        session_id = f"session-{i}"
        session = manager.start_session(session_id=session_id, course="12", mode=ConversationMode.STUDY.value)
        message = manager.append_user_message(f"Question {i}")
        context = manager.get_recent_context()
        processed_query = ProcessedQuery(
            session=session,
            user_message=message,
            context_window=context,
            course="12",
            mode=ConversationMode.STUDY,
            attachments=[],
            metadata={"course": "12"},
        )
        generator.generate(processed_query)

    # Should not have triggered yet (only 2 conversations)
    assert mock_runner.call_count == 0

    # Generate one more conversation to reach threshold of 3
    session = manager.start_session(session_id="session-2", course="12", mode=ConversationMode.STUDY.value)
    message = manager.append_user_message("Question 2")
    context = manager.get_recent_context()
    processed_query = ProcessedQuery(
        session=session,
        user_message=message,
        context_window=context,
        course="12",
        mode=ConversationMode.STUDY,
        attachments=[],
        metadata={"course": "12"},
    )

    # Mock the run_ace_cycles function
    from unittest.mock import patch

    with patch("genai_companion_with_ace.rag.answer_generation.run_ace_cycles") as mock_run_ace:
        updated_playbook = PlaybookContext(
            version="1.1.0",
            system_instructions="Updated instructions",
            heuristics=["Provide detailed answers.", "Cite sources."],
            examples=[],
        )
        mock_run_ace.return_value = updated_playbook
        playbook_loader.load_latest.return_value = updated_playbook

        generator.generate(processed_query)

        # Should have triggered ACE cycles
        mock_run_ace.assert_called_once()
        assert generator._playbook.version == "1.1.0"
        assert generator._last_trigger_count == 3


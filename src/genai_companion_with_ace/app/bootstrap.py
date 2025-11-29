"""Runtime bootstrap helpers for the CLI and UI front-ends."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genai_companion_with_ace.ace_integration import ACETriggerConfig, ConversationLogger, PlaybookLoader
from genai_companion_with_ace.chat import ConversationManager, ConversationMode, ResponseFormatter, SessionHistoryStore
from genai_companion_with_ace.chat.query_processor import QueryProcessor
from genai_companion_with_ace.config import CompanionConfig
from genai_companion_with_ace.llm import LangChainLLMAdapter
from genai_companion_with_ace.rag import (
    AnswerGenerator,
    DocumentIngestionPipeline,
    EmbeddingFactory,
    GenerationConfig,
    RetrievalOrchestrator,
    VectorStoreManager,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeComponents:
    """Bundled runtime components for reuse across entry points."""

    config: CompanionConfig
    llm_settings: dict[str, Any]
    ingestion: DocumentIngestionPipeline
    vector_store: VectorStoreManager
    retrieval: RetrievalOrchestrator
    conversation_manager: ConversationManager
    playbook_loader: PlaybookLoader
    answer_generator: AnswerGenerator
    query_processor: QueryProcessor
    conversation_logger: ConversationLogger
    trigger_config: ACETriggerConfig
    ui_settings: dict[str, Any]


def build_runtime_components(config: CompanionConfig) -> RuntimeComponents:
    ingestion = DocumentIngestionPipeline(config.ingestion_config())
    embeddings = EmbeddingFactory(config.embedding_settings()).build()
    vector_store = VectorStoreManager(embeddings, config.vector_store_config())
    retrieval = RetrievalOrchestrator(vector_store, ingestion, config.retrieval_config())

    history_store = SessionHistoryStore(config.conversation_history_path())
    conversation_manager = ConversationManager(history_store, max_context_turns=config.conversation_max_turns())

    playbook_loader = PlaybookLoader.from_companion_config(config)
    playbook = playbook_loader.load_latest()
    logger = ConversationLogger(config.outputs.conversations)

    llm_settings = config.llm_settings()
    generation_config = GenerationConfig(
        temperature=llm_settings.get("temperature", 0.2),
        max_tokens=llm_settings.get("max_tokens", 4096),
    )
    llm_client = LangChainLLMAdapter(llm_settings)
    deep_llm_client = None
    llm_cfg = config.raw.get("llm", {})
    deep_provider_name = llm_cfg.get("deep_dive_provider")
    if deep_provider_name:
        try:
            deep_settings = config.llm_settings(provider=deep_provider_name)
            deep_llm_client = LangChainLLMAdapter(deep_settings)
        except Exception as exc:  # pragma: no cover - config dependent
            LOGGER.warning("Failed to initialize deep-dive provider %s: %s", deep_provider_name, exc)
    formatter = ResponseFormatter()

    ace_cfg = config.ace_config()
    trigger_config = ACETriggerConfig(
        repo_path=Path(ace_cfg.get("repository_path", "../Agentic-Context-Engineering")),
        playbook_output_dir=config.outputs.ace_playbooks,
        iterations=ace_cfg.get("iterations", 1),
        trigger_threshold=ace_cfg.get("trigger_threshold", 50),
        config_path=Path(ace_cfg.get("config_path")) if ace_cfg.get("config_path") else None,
    )

    answer_generator = AnswerGenerator(
        llm_client=llm_client,
        retrieval=retrieval,
        conversation_manager=conversation_manager,
        formatter=formatter,
        playbook=playbook,
        conversation_logger=logger,
        playbook_loader=playbook_loader,
        trigger_config=trigger_config,
        config=generation_config,
        deep_dive_llm_client=deep_llm_client,
    )
    query_processor = QueryProcessor(conversation_manager, retrieval, default_mode=ConversationMode.STUDY)
    ui_settings = config.ui_cli_settings()
    return RuntimeComponents(
        config=config,
        llm_settings=llm_settings,
        ingestion=ingestion,
        vector_store=vector_store,
        retrieval=retrieval,
        conversation_manager=conversation_manager,
        playbook_loader=playbook_loader,
        answer_generator=answer_generator,
        query_processor=query_processor,
        conversation_logger=logger,
        trigger_config=trigger_config,
        ui_settings=ui_settings,
    )


def load_runtime_from_path(config_path: Path) -> RuntimeComponents:
    """Load configuration and bootstrap all runtime services."""
    companion_config = CompanionConfig.from_file(config_path)
    return build_runtime_components(companion_config)

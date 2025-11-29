"""Configuration helpers for the GenAI Companion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from genai_companion_with_ace.rag.embeddings import EmbeddingConfig
from genai_companion_with_ace.rag.ingestion import IngestionConfig
from genai_companion_with_ace.rag.retrieval import RetrievalConfig
from genai_companion_with_ace.rag.vector_store import VectorStoreConfig

LLMProvider = Literal["ollama", "openai"]
EmbeddingProvider = Literal["sentence-transformers", "openai"]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(slots=True)
class OutputsConfig:
    conversations: Path
    metrics: Path
    ace_playbooks: Path
    logs: Path

    def ensure_all(self) -> None:
        for path in (self.conversations, self.metrics, self.ace_playbooks, self.logs):
            _ensure_dir(path)


@dataclass(slots=True)
class CompanionConfig:
    source: Path
    raw: dict[str, Any]
    outputs: OutputsConfig

    @classmethod
    def from_file(cls, path: Path) -> CompanionConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        outputs = raw.get("outputs", {})
        conversations = Path(outputs.get("conversations", "outputs/conversations"))
        metrics = Path(outputs.get("metrics", "outputs/metrics"))
        ace_playbooks = Path(outputs.get("ace_playbooks", "outputs/ace_playbooks"))
        logs = Path(outputs.get("logs", "outputs/logs"))
        instance = cls(
            source=path,
            raw=raw,
            outputs=OutputsConfig(
                conversations=conversations,
                metrics=metrics,
                ace_playbooks=ace_playbooks,
                logs=logs,
            ),
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        """Validate and materialize directories declared in the config."""
        self.outputs.ensure_all()
        vector_cfg = self.raw.get("vector_store", {})
        persist_dir = Path(vector_cfg.get("persist_directory", "data/chroma"))
        _ensure_dir(persist_dir)
        ingestion_cfg = self.raw.get("ingestion", {})
        processed_dir = Path(ingestion_cfg.get("processed_dir", "data/processed"))
        _ensure_dir(processed_dir)
        history_cfg = self.raw.get("conversation_history", {})
        history_path = Path(history_cfg.get("path", self.outputs.conversations / "history.db"))
        history_path.parent.mkdir(parents=True, exist_ok=True)

    def llm_settings(self, provider: LLMProvider | None = None) -> dict[str, Any]:
        llm_cfg = self.raw.get("llm", {})
        provider_name: str = provider or llm_cfg.get("provider", "ollama")
        providers = llm_cfg.get("providers", {})
        provider_settings = providers.get(provider_name)
        if not provider_settings:
            available = ", ".join(sorted(providers.keys()))
            message = f"Unsupported LLM provider '{provider_name}'. Available: {available or 'none'}."
            raise ValueError(message)
        merged = dict(provider_settings)
        merged["provider"] = provider_name
        # Apply global overrides
        for key, value in llm_cfg.items():
            if key not in {"provider", "providers"} and key not in merged:
                merged[key] = value
        return merged

    def embedding_settings(self, provider: EmbeddingProvider | None = None) -> EmbeddingConfig:
        emb_cfg = self.raw.get("embedding", {})
        provider_name: str = provider or emb_cfg.get("provider", "sentence-transformers")
        providers = emb_cfg.get("providers", {})
        provider_settings = providers.get(provider_name)
        if not provider_settings:
            available = ", ".join(sorted(providers.keys()))
            message = f"Unsupported embedding provider '{provider_name}'. Available: {available or 'none'}."
            raise ValueError(message)
        if provider_name == "sentence-transformers":
            return EmbeddingConfig(
                provider="sentence-transformers",
                model_name=provider_settings.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                device=provider_settings.get("device", "cpu"),
                normalize_embeddings=provider_settings.get("normalize_embeddings", True),
            )
        if provider_name == "openai":
            return EmbeddingConfig(
                provider="openai",
                model_name=provider_settings.get("model_name", "text-embedding-3-small"),
                api_base=provider_settings.get("api_base"),
                api_key_env=provider_settings.get("api_key_env", "OPENAI_API_KEY"),
                dimensions=provider_settings.get("dimensions"),
            )
        message = f"Embedding provider '{provider_name}' is not supported."
        raise ValueError(message)

    def vector_store_config(self) -> VectorStoreConfig:
        cfg = self.raw.get("vector_store", {})
        defaults = VectorStoreConfig()
        metadata_fields = cfg.get("metadata_fields", defaults.metadata_fields)
        return VectorStoreConfig(
            collection_name=cfg.get("collection_name", defaults.collection_name),
            persist_directory=Path(cfg.get("persist_directory", defaults.persist_directory)),
            reset_on_start=cfg.get("reset_on_start", defaults.reset_on_start),
            tenant=cfg.get("tenant", defaults.tenant),
            metadata_fields=tuple(metadata_fields),
        )

    def ingestion_config(self) -> IngestionConfig:
        cfg = self.raw.get("ingestion", {})
        defaults = IngestionConfig()
        max_bytes = cfg.get("max_file_size_bytes")
        if max_bytes is None and "max_file_size_mb" in cfg:
            max_bytes = int(cfg.get("max_file_size_mb", defaults.max_file_size_bytes // (1024 * 1024))) * 1024 * 1024
        if max_bytes is None:
            max_bytes = defaults.max_file_size_bytes
        return IngestionConfig(
            processed_dir=Path(cfg.get("processed_dir", defaults.processed_dir)),
            chunk_size=cfg.get("chunk_size", defaults.chunk_size),
            chunk_overlap=cfg.get("chunk_overlap", defaults.chunk_overlap),
            allowed_metadata_keys=tuple(cfg.get("metadata_keys", defaults.allowed_metadata_keys)),
            max_file_size_bytes=int(max_bytes),
        )

    def retrieval_config(self) -> RetrievalConfig:
        cfg = self.raw.get("retrieval", {})
        defaults = RetrievalConfig()
        return RetrievalConfig(
            dense_top_k=cfg.get("dense_top_k", defaults.dense_top_k),
            keyword_top_k=cfg.get("keyword_top_k", defaults.keyword_top_k),
            hybrid_top_k=cfg.get("hybrid_top_k", defaults.hybrid_top_k),
            deduplicate=cfg.get("deduplicate", defaults.deduplicate),
            min_score_threshold=cfg.get("min_score_threshold", defaults.min_score_threshold),
        )

    def conversation_history_path(self) -> Path:
        cfg = self.raw.get("conversation_history", {})
        path = Path(cfg.get("path", self.outputs.conversations / "history.db"))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def conversation_max_turns(self) -> int:
        cfg = self.raw.get("conversation_history", {})
        result = cfg.get("max_turns", 20)
        return int(result)  # Ensure we return int, not Any

    def ace_config(self) -> dict[str, Any]:
        cfg = self.raw.get("ace", {})
        resolved = dict(cfg)
        resolved.setdefault("playbook_output_dir", str(self.outputs.ace_playbooks))
        return resolved

    def ui_cli_settings(self) -> dict[str, Any]:
        defaults = {
            "boxed_answers": True,
            "reflow_on_resize": True,
            "box_style": "simple",
        }
        ui_cfg = self.raw.get("ui", {})
        cli_cfg = ui_cfg.get("cli", {})
        resolved = dict(defaults)
        if isinstance(cli_cfg, dict):
            resolved.update(cli_cfg)
        return resolved

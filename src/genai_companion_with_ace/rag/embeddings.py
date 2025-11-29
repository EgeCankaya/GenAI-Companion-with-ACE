"""Embedding factory utilities for the IBM Gen AI Companion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from langchain_core.embeddings import Embeddings

LOGGER = logging.getLogger(__name__)


def _resolve_device(device: str | None) -> str:
    """Resolve device string to a valid PyTorch device.

    Args:
        device: Device string, can be 'auto', 'cpu', 'cuda', etc.

    Returns:
        Valid PyTorch device string (e.g., 'cpu' or 'cuda')
    """
    if device is None or device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


ProviderLiteral = Literal["sentence-transformers", "openai"]


@dataclass(slots=True, frozen=True)
class EmbeddingConfig:
    """Configuration parameters for creating embedding functions."""

    provider: ProviderLiteral = "sentence-transformers"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = "cpu"
    normalize_embeddings: bool = True
    # OpenAI specific
    api_base: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    dimensions: int | None = None


class EmbeddingFactory:
    """Factory wrapper that instantiates embedding clients on demand."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self._config = config or EmbeddingConfig()
        self._embedding: Embeddings | None = None

    @property
    def config(self) -> EmbeddingConfig:
        return self._config

    def build(self) -> Embeddings:
        """Instantiate and memoize the embedding client."""
        if self._embedding is None:
            provider = self._config.provider
            if provider == "sentence-transformers":
                self._embedding = self._build_sentence_transformers()
            elif provider == "openai":
                self._embedding = self._build_openai()
            else:
                message = f"Unsupported embedding provider: {provider}"
                raise ValueError(message)
        return self._embedding

    def _build_sentence_transformers(self) -> Embeddings:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover - dependency should be present
            message = "Install 'langchain-huggingface' to enable sentence transformer embeddings"
            raise RuntimeError(message) from exc

        device = _resolve_device(self._config.device)
        LOGGER.info(
            "Loading sentence-transformer embeddings model '%s' (device=%s)",
            self._config.model_name,
            device,
        )
        return HuggingFaceEmbeddings(
            model_name=self._config.model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": self._config.normalize_embeddings},
        )

    def _build_openai(self) -> Embeddings:
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover
            message = "Install 'langchain-openai' to enable OpenAI embeddings"
            raise RuntimeError(message) from exc

        LOGGER.info("Loading OpenAI embedding model '%s'", self._config.model_name)
        from typing import Any

        options: dict[str, Any] = {
            "model": self._config.model_name,
            "api_key": None,
            "openai_api_base": self._config.api_base,
            "tiktoken_model_name": self._config.model_name,
        }
        if self._config.dimensions:
            options["dimensions"] = self._config.dimensions

        return OpenAIEmbeddings(**options)

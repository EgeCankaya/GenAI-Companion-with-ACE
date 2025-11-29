"""Diagnostics utilities for the GenAI Companion."""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_core.embeddings import Embeddings

from genai_companion_with_ace.config import CompanionConfig
from genai_companion_with_ace.llm import check_ollama_connection, check_ollama_model
from genai_companion_with_ace.rag.vector_store import VectorStoreManager


@dataclass(slots=True)
class DiagnosticResult:
    status: str
    details: str

    def as_dict(self) -> dict[str, str]:
        return {"status": self.status, "details": self.details}


class _DiagnosticsEmbeddings(Embeddings):
    """Lightweight stub embeddings used only for health checks."""

    def __init__(self, dimension: int = 3) -> None:
        self._dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self._dimension

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - async rarely used
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:  # pragma: no cover
        return self.embed_query(text)


def _check_python_version() -> DiagnosticResult:
    major, minor = sys.version_info.major, sys.version_info.minor
    if major >= 3 and 9 <= minor <= 13:
        return DiagnosticResult("ok", f"Python {platform.python_version()} detected.")
    return DiagnosticResult(
        "warn",
        f"Python {platform.python_version()} detected; project recommends 3.9-3.13.",
    )


def _check_optional_dependency(module_name: str, friendly_name: str) -> DiagnosticResult:
    try:
        importlib.import_module(module_name)
        return DiagnosticResult("ok", f"{friendly_name} available.")
    except Exception as exc:  # pragma: no cover - best effort
        return DiagnosticResult("warn", f"{friendly_name} missing: {exc}")


def _check_vector_store(config: CompanionConfig) -> DiagnosticResult:
    try:
        manager = VectorStoreManager(_DiagnosticsEmbeddings(), config.vector_store_config())
        healthy, message = manager.check_health()
        if healthy:
            return DiagnosticResult("ok", f"Chroma collection '{manager.collection_name}' accessible.")
        return DiagnosticResult("error", message or "Vector store health check failed.")
    except Exception as exc:  # pragma: no cover
        return DiagnosticResult("error", f"Vector store check failed: {exc}")


def _check_ollama(config: CompanionConfig) -> DiagnosticResult:
    llm_settings = config.llm_settings()
    if llm_settings.get("provider") != "ollama":
        return DiagnosticResult("not_applicable", "LLM provider is not Ollama.")
    base_url = llm_settings.get("base_url", "http://localhost:11434")
    if not check_ollama_connection(base_url):
        return DiagnosticResult("error", f"Ollama not reachable at {base_url}.")
    model_name = str(llm_settings.get("model", "llama3.1:8b"))
    installed, models = check_ollama_model(base_url, model_name)
    if not installed:
        available = ", ".join(models) if models else "none"
        return DiagnosticResult("warn", f"Model '{model_name}' missing. Available: {available}.")
    return DiagnosticResult("ok", f"Ollama reachable and model '{model_name}' installed.")


def _check_ace_repository(config: CompanionConfig) -> DiagnosticResult:
    ace_cfg = config.ace_config()
    repo_path = Path(ace_cfg.get("repository_path", "../Agentic-Context-Engineering")).expanduser()
    if repo_path.exists():
        return DiagnosticResult("ok", f"ACE repo found at {repo_path}.")
    return DiagnosticResult("warn", f"ACE repository not found at {repo_path}.")


def run_diagnostics(config: CompanionConfig) -> dict[str, dict[str, str]]:
    """Run a suite of health checks and return structured results."""
    results: dict[str, dict[str, str]] = {}
    results["python"] = _check_python_version().as_dict()
    results["ollama"] = _check_ollama(config).as_dict()
    results["vector_store"] = _check_vector_store(config).as_dict()
    results["ace_repository"] = _check_ace_repository(config).as_dict()

    optional_dependencies = {
        "pypdf": "PDF ingestion (pypdf)",
        "docx": "DOCX ingestion (python-docx)",
        "langchain_huggingface": "Sentence-transformer embeddings",
        "langchain_openai": "OpenAI LLM/embeddings integration",
    }
    for module_name, friendly in optional_dependencies.items():
        key = f"dep:{module_name}"
        results[key] = _check_optional_dependency(module_name, friendly).as_dict()

    return results


__all__ = ["DiagnosticResult", "run_diagnostics"]

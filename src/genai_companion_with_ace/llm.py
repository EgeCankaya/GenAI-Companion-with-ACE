"""LLM provider helpers."""

from __future__ import annotations

from typing import Any

import requests


def check_ollama_connection(base_url: str = "http://localhost:11434", timeout: int = 2) -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return response.status_code == 200
    except (OSError, requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


def check_ollama_model(base_url: str, model_name: str, timeout: int = 2) -> tuple[bool, list[str]]:
    """Check if a specific Ollama model is installed. Returns (is_installed, available_models)."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        data = response.json()
    except (OSError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, ValueError):
        return False, []
    if response.status_code != 200:
        return False, []

    models = [model.get("name", "") for model in data.get("models", [])]
    model_installed = any(
        model_name == model or model_name.startswith(model.split(":")[0]) or model.startswith(model_name.split(":")[0])
        for model in models
    )
    return model_installed, models


class LangChainLLMAdapter:
    """Adapter that hydrates a LangChain LLM client based on provider config."""

    def __init__(self, settings: dict[str, Any]) -> None:
        provider = settings.get("provider", "ollama")
        self.provider = str(provider)
        self.settings = settings
        self._current_max_tokens = settings.get("max_tokens", 4096)
        self.llm = self._initialize_llm()

    def _initialize_llm(self, max_tokens: int | None = None) -> Any:  # pragma: no cover - depends on runtime environment
        max_tokens = max_tokens or self._current_max_tokens
        if self.provider == "ollama":
            from langchain_community.llms import Ollama

            return Ollama(
                model=self.settings.get("model", "llama3.1:8b"),
                base_url=self.settings.get("base_url", "http://localhost:11434"),
                temperature=self.settings.get("temperature", 0.2),
                num_ctx=self.settings.get("num_ctx", 8192),
                num_predict=max_tokens,
            )
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.settings.get("model", "gpt-4o-mini"),
                temperature=self.settings.get("temperature", 0.2),
                max_tokens=max_tokens,  # type: ignore[call-arg]  # ChatOpenAI accepts max_tokens but type stubs may be outdated
                base_url=self.settings.get("base_url"),
                timeout=self.settings.get("timeout"),
            )
        message = f"Unsupported LLM provider: {self.provider}"
        raise ValueError(message)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        # Support dynamic max_tokens per call
        max_tokens = kwargs.get("max_tokens")
        if max_tokens and max_tokens != self._current_max_tokens:
            self._current_max_tokens = max_tokens
            self.llm = self._initialize_llm(max_tokens=max_tokens)

        response = self.llm.invoke(prompt)
        if isinstance(response, str):
            return response
        return getattr(response, "content", str(response))

from __future__ import annotations

import pytest
from langchain_core.embeddings import Embeddings

from genai_companion_with_ace.rag.embeddings import EmbeddingConfig, EmbeddingFactory


class DummyEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.last_query: str | None = None
        self.last_documents: list[str] | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.last_documents = texts
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.last_query = text
        return [float(len(text))]


def test_sentence_transformer_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyEmbeddings()

    def fake_ctor(**kwargs):
        fake_ctor.kwargs = kwargs
        return dummy

    monkeypatch.setattr("langchain_huggingface.HuggingFaceEmbeddings", fake_ctor)

    factory = EmbeddingFactory(EmbeddingConfig(model_name="sentence-transformers/test-model"))
    embeddings = factory.build()

    assert embeddings is dummy
    assert fake_ctor.kwargs["model_name"] == "sentence-transformers/test-model"
    assert fake_ctor.kwargs["model_kwargs"]["device"] == "cpu"
    embeddings.embed_query("hello")
    assert dummy.last_query == "hello"


def test_sentence_transformer_auto_device(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyEmbeddings()

    def fake_ctor(**kwargs):
        fake_ctor.kwargs = kwargs
        return dummy

    monkeypatch.setattr("langchain_huggingface.HuggingFaceEmbeddings", fake_ctor)

    # Mock torch.cuda.is_available to return False for predictable test
    def mock_cuda_available():
        return False

    monkeypatch.setattr("torch.cuda.is_available", mock_cuda_available)

    factory = EmbeddingFactory(EmbeddingConfig(model_name="sentence-transformers/test-model", device="auto"))
    embeddings = factory.build()

    assert embeddings is dummy
    # Device "auto" should be resolved to "cpu" when CUDA is not available
    assert fake_ctor.kwargs["model_kwargs"]["device"] == "cpu"


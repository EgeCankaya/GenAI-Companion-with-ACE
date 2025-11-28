from __future__ import annotations

from pathlib import Path

from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document

from genai_companion_with_ace.rag import VectorStoreConfig, VectorStoreManager


def test_vector_store_add_and_reset(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings(size=4)
    config = VectorStoreConfig(
        collection_name="test_collection",
        persist_directory=tmp_path / "chroma",
    )
    manager = VectorStoreManager(embeddings, config)

    documents = [
        Document(page_content="Transformers use attention", metadata={"document_id": "doc-1", "course": "12"}),
        Document(page_content="Flask routes use decorators", metadata={"document_id": "doc-2", "course": "5"}),
    ]
    ids = manager.add_documents(documents)
    assert len(ids) == 2

    retriever = manager.as_retriever(search_kwargs={"k": 1})
    results = retriever.invoke("attention")
    assert results

    manager.delete_documents([documents[0].metadata["document_id"]])
    remaining = manager.as_retriever(search_kwargs={"k": 5}).invoke("attention")
    assert all(doc.metadata.get("document_id") != "doc-1" for doc in remaining)

    manager.reset()
    post_reset_results = manager.as_retriever(search_kwargs={"k": 1}).invoke("attention")
    assert post_reset_results == []


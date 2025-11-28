from __future__ import annotations

from pathlib import Path

from langchain_community.embeddings import FakeEmbeddings

from genai_companion_with_ace.rag import (
    DocumentIngestionPipeline,
    IngestionConfig,
    RetrievalAttachment,
    RetrievalConfig,
    RetrievalOrchestrator,
    VectorStoreConfig,
    VectorStoreManager,
)


def build_orchestrator(tmp_path: Path) -> tuple[RetrievalOrchestrator, DocumentIngestionPipeline]:
    ingestion = DocumentIngestionPipeline(
        IngestionConfig(processed_dir=tmp_path / "processed", chunk_size=64, chunk_overlap=10)
    )
    vector_manager = VectorStoreManager(
        FakeEmbeddings(size=8),
        VectorStoreConfig(collection_name="test", persist_directory=tmp_path / "chroma"),
    )
    orchestrator = RetrievalOrchestrator(
        vector_store=vector_manager,
        ingestion_pipeline=ingestion,
        config=RetrievalConfig(dense_top_k=3, keyword_top_k=3, hybrid_top_k=5),
    )
    return orchestrator, ingestion


def test_retrieval_combines_sources(tmp_path: Path) -> None:
    orchestrator, ingestion = build_orchestrator(tmp_path)

    docs = ingestion.ingest_raw_content(
        content="Transformers rely on self-attention and positional encoding.",
        source_name="course12_transformers.md",
        metadata={"course": "12"},
        persist=False,
    )
    orchestrator.index_documents(docs)

    attachments = [
        RetrievalAttachment(
            name="notes.txt",
            content="Remember that attention weights are normalized with softmax.",
            metadata={"course": "12", "topic": "attention"},
        )
    ]

    result = orchestrator.retrieve("How do transformers use attention?", attachments=attachments)

    assert result.combined
    assert any(chunk.source.startswith("attachment") for chunk in result.attachment_chunks)
    assert any(chunk.source.startswith("dense") for chunk in result.dense_results)
    assert len(result.combined) <= orchestrator.config.hybrid_top_k


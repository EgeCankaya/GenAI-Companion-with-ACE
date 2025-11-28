from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from genai_companion_with_ace.rag import DocumentIngestionPipeline, IngestionConfig, load_processed_document


def create_pipeline(tmp_path: Path) -> DocumentIngestionPipeline:
    config = IngestionConfig(
        processed_dir=tmp_path / "processed",
        chunk_size=50,
        chunk_overlap=10,
    )
    return DocumentIngestionPipeline(config)


def test_ingest_raw_content_persists_chunks(tmp_path: Path) -> None:
    pipeline = create_pipeline(tmp_path)
    content = "Backpropagation computes gradients.\n" * 5

    documents = pipeline.ingest_raw_content(
        content=content,
        source_name="course09_backprop.md",
        metadata={"course": "9", "module": "Backpropagation"},
        persist=True,
    )

    assert len(documents) >= 1
    first_doc = documents[0]
    assert isinstance(first_doc, Document)
    assert first_doc.metadata["course"] == "9"
    assert first_doc.metadata["module"] == "Backpropagation"
    assert first_doc.metadata["document_id"].startswith("course09_backprop.md")

    doc_dir = pipeline.config.processed_dir / first_doc.metadata["document_id"]
    rehydrated = load_processed_document(doc_dir)
    assert len(rehydrated) == len(documents)
    assert rehydrated[0].page_content == first_doc.page_content


def test_ingest_path_rejects_unsupported_extension(tmp_path: Path) -> None:
    pipeline = create_pipeline(tmp_path)
    unsupported = tmp_path / "notes.unsupported"
    unsupported.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError):
        pipeline.ingest_path(unsupported)


def test_ingest_path_enforces_size_limit(tmp_path: Path) -> None:
    config = IngestionConfig(
        processed_dir=tmp_path / "processed",
        chunk_size=50,
        chunk_overlap=10,
        max_file_size_bytes=1024,
    )
    pipeline = DocumentIngestionPipeline(config)
    oversized = tmp_path / "oversized.txt"
    oversized.write_text("x" * 2048, encoding="utf-8")

    with pytest.raises(ValueError):
        pipeline.ingest_path(oversized)


def test_list_ingested_documents_returns_manifest(tmp_path: Path) -> None:
    pipeline = create_pipeline(tmp_path)
    sample = tmp_path / "lecture.md"
    sample.write_text("# Title\n\nContent", encoding="utf-8")

    pipeline.ingest_path(sample, metadata={"course": "1"}, persist=True)
    summaries = pipeline.list_ingested_documents()

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.metadata["course"] == "1"
    assert summary.chunk_count > 0


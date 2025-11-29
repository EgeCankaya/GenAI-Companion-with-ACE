"""Utility script to bootstrap the vector store with evaluation questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from genai_companion_with_ace.config import CompanionConfig
from genai_companion_with_ace.rag import (
    DocumentIngestionPipeline,
    EmbeddingFactory,
    RetrievalOrchestrator,
    VectorStoreManager,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest evaluation questions into the vector store.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/eval_questions_100.json"),
        help="Path to the evaluation questions JSON file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/companion_config.yaml"),
        help="Path to the companion configuration file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on how many questions to ingest.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of chunks to index per batch.",
    )
    return parser.parse_args()


def load_dataset(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        return entries[:limit]
    return entries


def build_ingestion_components(config_path: Path) -> tuple[DocumentIngestionPipeline, RetrievalOrchestrator]:
    companion_config = CompanionConfig.from_file(config_path)
    ingestion = DocumentIngestionPipeline(companion_config.ingestion_config())
    embeddings = EmbeddingFactory(companion_config.embedding_settings()).build()
    vector_store = VectorStoreManager(embeddings, companion_config.vector_store_config())
    retrieval = RetrievalOrchestrator(vector_store, ingestion, companion_config.retrieval_config())
    return ingestion, retrieval


def to_document_text(example: dict[str, Any]) -> str:
    question = example.get("question", "").strip()
    golden = example.get("golden_answer", "").strip()
    criteria = example.get("evaluation_criteria", {}) or {}
    must_include = ", ".join(criteria.get("must_include", []))
    should_cite = ", ".join(criteria.get("should_cite", []))

    lines = [
        f"# Evaluation Question: {example.get('id', 'unknown')}",
        "",
        "## Prompt",
        question or "N/A",
        "",
        "## Golden Answer",
        golden or "N/A",
    ]
    if must_include or should_cite:
        lines.extend([
            "",
            "## Evaluation Criteria",
        ])
        if must_include:
            lines.append(f"- Must include: {must_include}")
        if should_cite:
            lines.append(f"- Should cite: {should_cite}")
    return "\n".join(lines)


def ingest_dataset(
    *,
    dataset: list[dict[str, Any]],
    ingestion: DocumentIngestionPipeline,
    retrieval: RetrievalOrchestrator,
    batch_size: int,
) -> tuple[int, int]:
    chunk_buffer: list[Any] = []
    chunk_total = 0

    for entry in dataset:
        metadata = {
            "course": entry.get("course", "unknown"),
            "module": entry.get("module", "unknown"),
            "topic": entry.get("module", "unknown"),
            "difficulty": entry.get("difficulty", "unknown"),
            "content_type": "evaluation_question",
        }
        docs = ingestion.ingest_raw_content(
            content=to_document_text(entry),
            source_name=f"eval_{entry.get('id', 'unknown')}.md",
            metadata=metadata,
            persist=False,
        )
        chunk_buffer.extend(docs)
        chunk_total += len(docs)

        if len(chunk_buffer) >= batch_size:
            retrieval.index_documents(chunk_buffer)
            chunk_buffer.clear()

    if chunk_buffer:
        retrieval.index_documents(chunk_buffer)

    return len(dataset), chunk_total


def main() -> None:
    args = parse_args()

    dataset = load_dataset(args.dataset, args.limit)
    if not dataset:
        print("No evaluation questions found; nothing to ingest.")
        return

    ingestion, retrieval = build_ingestion_components(args.config)
    doc_count, chunk_count = ingest_dataset(
        dataset=dataset,
        ingestion=ingestion,
        retrieval=retrieval,
        batch_size=max(1, args.batch_size),
    )

    print(
        f"Ingested {doc_count} evaluation questions "
        f"({chunk_count} chunks) into collection '{retrieval._vector_store.collection_name}'.",
    )


if __name__ == "__main__":
    main()

"""Ingest transcript files into the vector store."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from genai_companion_with_ace.config import CompanionConfig
from genai_companion_with_ace.rag import (
    DocumentIngestionPipeline,
    EmbeddingFactory,
    RetrievalOrchestrator,
    VectorStoreManager,
)
from genai_companion_with_ace.rag.ingestion import SUPPORTED_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest course transcripts into the vector store.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/transcripts"),
        help="Directory containing transcript files (txt, md, pdf, srt, vtt, etc.).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/companion_config.yaml"),
        help="Path to companion configuration.",
    )
    parser.add_argument(
        "--course",
        type=str,
        default="unknown",
        help="Course identifier to attach as metadata.",
    )
    parser.add_argument(
        "--module",
        type=str,
        default="transcripts",
        help="Module/topic identifier for metadata.",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="all",
        help="Difficulty tag for metadata.",
    )
    parser.add_argument(
        "--content-type",
        type=str,
        default="transcript",
        help="Content type metadata label.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of chunks to index per batch.",
    )
    return parser.parse_args()


def build_components(config: CompanionConfig) -> tuple[DocumentIngestionPipeline, RetrievalOrchestrator]:
    ingestion = DocumentIngestionPipeline(config.ingestion_config())
    embeddings = EmbeddingFactory(config.embedding_settings()).build()
    vector_store = VectorStoreManager(embeddings, config.vector_store_config())
    retrieval = RetrievalOrchestrator(vector_store, ingestion, config.retrieval_config())
    return ingestion, retrieval


def discover_files(root: Path) -> Sequence[Path]:
    if not root.exists():
        return []
    allowed = {ext.lower() for ext in SUPPORTED_EXTENSIONS}
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed]
    return sorted(files)


def ingest_files(
    *,
    files: Iterable[Path],
    ingestion: DocumentIngestionPipeline,
    retrieval: RetrievalOrchestrator,
    metadata: dict[str, str],
    batch_size: int,
) -> tuple[int, int]:
    all_chunks = []
    chunk_counter = 0
    file_counter = 0

    for path in files:
        docs = ingestion.ingest_path(path, metadata=metadata, persist=True)
        if not docs:
            continue
        all_chunks.extend(docs)
        chunk_counter += len(docs)
        file_counter += 1

        if len(all_chunks) >= batch_size:
            retrieval.index_documents(all_chunks)
            all_chunks.clear()

    if all_chunks:
        retrieval.index_documents(all_chunks)

    return file_counter, chunk_counter


def main() -> None:
    args = parse_args()
    files = discover_files(args.input_dir)
    if not files:
        print(f"No transcript files found in {args.input_dir}.")
        return

    companion_config = CompanionConfig.from_file(args.config)
    ingestion, retrieval = build_components(companion_config)

    metadata = {
        "course": args.course,
        "module": args.module,
        "topic": args.module,
        "difficulty": args.difficulty,
        "content_type": args.content_type,
    }

    file_count, chunk_count = ingest_files(
        files=files,
        ingestion=ingestion,
        retrieval=retrieval,
        metadata=metadata,
        batch_size=max(1, args.batch_size),
    )

    print(
        f"Ingested {file_count} transcript files ({chunk_count} chunks) into collection "
        f"'{retrieval._vector_store.collection_name}'.",  # noqa: SLF001
    )


if __name__ == "__main__":
    main()



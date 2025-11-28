"""Hybrid retrieval orchestration for the IBM Gen AI Companion."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from genai_companion_with_ace.rag.ingestion import DocumentIngestionPipeline
from genai_companion_with_ace.rag.vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class RetrievalConfig:
    """Customizable retrieval parameters."""

    dense_top_k: int = 5
    keyword_top_k: int = 5
    hybrid_top_k: int = 8
    deduplicate: bool = True
    min_score_threshold: float = 0.0


@dataclass(slots=True)
class RetrievalAttachment:
    """Runtime attachment supplied alongside a user query."""

    name: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    """Metadata wrapper for retrieved chunks."""

    document: Document
    score: float
    source: str  # dense | keyword | attachment


@dataclass(slots=True)
class RetrievalResult:
    """Container holding retrieval outputs for downstream prompting."""

    combined: list[RetrievedChunk]
    dense_results: list[RetrievedChunk]
    keyword_results: list[RetrievedChunk]
    attachment_chunks: list[RetrievedChunk]


class RetrievalOrchestrator:
    """Coordinates dense, keyword, and attachment retrieval."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        ingestion_pipeline: DocumentIngestionPipeline,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._ingestion_pipeline = ingestion_pipeline
        self._config = config or RetrievalConfig()
        self._keyword_docs: list[Document] = []
        self._bm25: BM25Retriever | None = None

    @property
    def config(self) -> RetrievalConfig:
        return self._config

    def index_documents(self, documents: Iterable[Document]) -> list[str]:
        """Index a new batch of documents in both vector store and keyword retriever."""
        docs = [self._vector_store.metadata_filter(doc) for doc in documents]
        ids = self._vector_store.add_documents(docs)
        self._keyword_docs.extend(docs)
        self._refresh_keyword_retriever()
        return ids

    def retrieve(
        self,
        query: str,
        *,
        attachments: Sequence[RetrievalAttachment] | None = None,
    ) -> RetrievalResult:
        dense_chunks = self._dense_search(query)
        keyword_chunks = self._keyword_search(query)
        attachment_chunks = self._attachment_chunks(attachments or [])
        combined = self._merge_results(dense_chunks, keyword_chunks, attachment_chunks)
        return RetrievalResult(
            combined=combined,
            dense_results=dense_chunks,
            keyword_results=keyword_chunks,
            attachment_chunks=attachment_chunks,
        )

    def _dense_search(self, query: str) -> list[RetrievedChunk]:
        retriever = self._vector_store.as_retriever(search_kwargs={"k": self._config.dense_top_k})
        documents = retriever.invoke(query)
        results: list[RetrievedChunk] = []
        for index, document in enumerate(documents):
            score = 1.0 / (index + 1)
            results.append(RetrievedChunk(document=document, score=score, source="dense"))
        return results

    def _keyword_search(self, query: str) -> list[RetrievedChunk]:
        if not self._bm25:
            return []
        documents = self._bm25.invoke(query)[: self._config.keyword_top_k]
        results: list[RetrievedChunk] = []
        for index, document in enumerate(documents):
            score = 1.0 / (index + 1.5)
            results.append(RetrievedChunk(document=document, score=score, source="keyword"))
        return results

    def _attachment_chunks(self, attachments: Sequence[RetrievalAttachment]) -> list[RetrievedChunk]:
        if not attachments:
            return []
        chunks: list[RetrievedChunk] = []
        for attachment in attachments:
            metadata = {"source": f"attachment:{attachment.name}"}
            metadata.update(attachment.metadata)
            docs = self._ingestion_pipeline.ingest_raw_content(
                content=attachment.content,
                source_name=attachment.name,
                metadata=metadata,
                persist=False,
            )
            for index, doc in enumerate(docs):
                chunks.append(
                    RetrievedChunk(
                        document=doc,
                        score=1.0 - (index * 0.05),
                        source="attachment",
                    )
                )
        return chunks

    def _merge_results(
        self,
        dense_results: Sequence[RetrievedChunk],
        keyword_results: Sequence[RetrievedChunk],
        attachment_results: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        combined: list[RetrievedChunk] = []
        seen_ids: dict[str, RetrievedChunk] = {}

        for bucket in (dense_results, keyword_results, attachment_results):
            for chunk in bucket:
                key = self._derive_chunk_key(chunk.document)
                existing = seen_ids.get(key)
                if existing is None:
                    seen_ids[key] = chunk
                    combined.append(chunk)
                else:
                    # Keep the higher score and annotate provenance
                    if chunk.score > existing.score:
                        existing.score = chunk.score
                    existing.source = f"{existing.source}+{chunk.source}"

        combined.sort(key=lambda chunk: chunk.score, reverse=True)
        if self._config.hybrid_top_k:
            combined = combined[: self._config.hybrid_top_k]

        if self._config.min_score_threshold > 0.0:
            combined = [chunk for chunk in combined if chunk.score >= self._config.min_score_threshold]

        return combined

    def _refresh_keyword_retriever(self) -> None:
        if not self._keyword_docs:
            self._bm25 = None
            return
        self._bm25 = BM25Retriever.from_documents(self._keyword_docs)

    @staticmethod
    def _derive_chunk_key(document: Document) -> str:
        metadata = document.metadata
        chunk_id = metadata.get("chunk_id")
        document_id = metadata.get("document_id") or metadata.get("source")
        return f"{document_id}:{chunk_id}"


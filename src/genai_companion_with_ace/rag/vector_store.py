"""Vector store management utilities for the IBM Gen AI Companion."""

from __future__ import annotations

import logging
import shutil
import uuid
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore as VectorStoreBase

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class VectorStoreConfig:
    """Configuration for connecting to a vector store backend."""

    collection_name: str = "ibm_genai_companion"
    persist_directory: Path = Path("data/chroma")
    reset_on_start: bool = False
    tenant: str | None = None
    metadata_fields: Sequence[str] = ("course", "module", "topic", "difficulty")


class VectorStoreManager:
    """Facilitates CRUD operations against the underlying vector store."""

    def __init__(self, embeddings: Embeddings, config: VectorStoreConfig | None = None) -> None:
        self._embeddings = embeddings
        self._config = config or VectorStoreConfig()
        self._config.persist_directory.mkdir(parents=True, exist_ok=True)
        self._store: VectorStoreBase | None = None
        if self._config.reset_on_start:
            self.reset()

    @property
    def collection_name(self) -> str:
        return self._config.collection_name

    @property
    def persist_directory(self) -> Path:
        return self._config.persist_directory

    def as_retriever(self, **kwargs):  # type: ignore[no-untyped-def]
        """Return a LangChain retriever wrapper."""
        store = self._ensure_store()
        return store.as_retriever(**kwargs)

    def add_documents(self, documents: Iterable[Document]) -> list[str]:
        """Add a batch of documents to the vector store and return their IDs."""
        store = self._ensure_store()
        doc_list = list(documents)
        if not doc_list:
            return []
        ids: list[str] = []
        for doc in doc_list:
            doc_id = doc.metadata.get("document_id")
            if not doc_id:
                doc_id = f"chunk-{uuid.uuid4().hex}"
                doc.metadata["document_id"] = doc_id
            ids.append(str(doc_id))

        LOGGER.info("Adding %s documents to vector store '%s'", len(doc_list), self._config.collection_name)
        store.add_documents(doc_list, ids=ids)
        return ids

    def update_document(self, document_id: str, document: Document) -> None:
        """Replace a vector entry by deleting and re-adding it."""
        self.delete_documents([document_id])
        document.metadata.setdefault("document_id", document_id)
        self.add_documents([document])

    def delete_documents(self, document_ids: Sequence[str]) -> None:
        """Delete the specified document IDs from the vector store."""
        store = self._ensure_store()
        if not document_ids:
            return
        LOGGER.info("Deleting %s documents from vector store '%s'", len(document_ids), self._config.collection_name)
        store.delete(ids=list(document_ids))

    def reset(self) -> None:
        """Drop the existing vector store collection and start fresh.

        Ensures Chroma releases file handles before deleting on-disk artifacts to avoid
        DuckDB \"readonly database\" errors observed in CI.
        """
        self._dispose_store()
        self._clean_persist_directory()
        LOGGER.info("Vector store reset complete. Ready for re-initialization.")

    def check_health(self) -> tuple[bool, str | None]:
        """Check if the vector store is healthy and accessible.

        Returns:
            Tuple of (is_healthy, error_message). If healthy, error_message is None.
        """
        try:
            store = self._ensure_store()
            # Try to access the collection to verify it's healthy
            _ = store._collection  # type: ignore[attr-defined]  # Access internal collection to verify health
        except Exception as e:
            error_msg = str(e)
            if "panic" in error_msg.lower() or "out of range" in error_msg.lower():
                return False, f"Vector store appears corrupted: {error_msg[:200]}"
            return False, f"Vector store health check failed: {error_msg[:200]}"
        else:
            return True, None

    def metadata_filter(self, document: Document) -> Document:
        """Retain only relevant metadata fields."""
        filtered = {
            key: document.metadata[key]
            for key in document.metadata
            if key in self._config.metadata_fields or key.startswith("course")
        }
        filtered["document_id"] = document.metadata.get("document_id")
        return Document(page_content=document.page_content, metadata=filtered)

    def _ensure_store(self) -> VectorStoreBase:
        if self._store is None:
            self._config.persist_directory.mkdir(parents=True, exist_ok=True)
            LOGGER.debug(
                "Initializing Chroma vector store(collection=%s, persist_dir=%s)",
                self._config.collection_name,
                self._config.persist_directory,
            )
            client_settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                persist_directory=str(self._config.persist_directory),
            )
            self._store = Chroma(
                collection_name=self._config.collection_name,
                persist_directory=str(self._config.persist_directory),
                embedding_function=self._embeddings,
                client_settings=client_settings,
            )
        return self._store

    def _dispose_store(self) -> None:
        """Release Chroma resources before deleting files on disk."""
        store = self._store
        if store is None:
            return
        delete_collection = getattr(store, "delete_collection", None)
        if callable(delete_collection):
            try:
                delete_collection()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                LOGGER.warning("Chroma collection deletion failed (may be corrupted): %s", exc)
        client = getattr(store, "_client", None)
        if client is not None:
            for method_name in ("persist", "reset", "close"):
                method = getattr(client, method_name, None)
                if callable(method):
                    with suppress(Exception):  # pragma: no cover - best effort cleanup
                        method()
            system = getattr(client, "_system", None)
            stop = getattr(system, "stop", None)
            if callable(stop):
                with suppress(Exception):
                    stop()
        self._store = None

    def _clean_persist_directory(self) -> None:
        """Delete and recreate the persistence directory with robust logging."""
        persist_path = self._config.persist_directory
        if persist_path.exists():
            LOGGER.info("Removing persisted vector store at %s", persist_path)
            try:
                shutil.rmtree(persist_path)
            except Exception as exc:
                LOGGER.warning("Error removing vector store directory (may be corrupted): %s", exc)
                if persist_path.exists():
                    for item in persist_path.iterdir():
                        try:
                            if item.is_dir():
                                shutil.rmtree(item)
                            else:
                                item.unlink()
                        except Exception as inner_exc:
                            LOGGER.debug("Failed removing %s: %s", item, inner_exc)
        persist_path.mkdir(parents=True, exist_ok=True)

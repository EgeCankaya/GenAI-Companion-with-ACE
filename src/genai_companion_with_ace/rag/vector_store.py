"""Vector store management utilities for the IBM Gen AI Companion."""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore as VectorStoreBase

LOGGER = logging.getLogger(__name__)

# Workaround for ChromaDB SegmentAPI compatibility with hnswlib
# ChromaDB's SegmentAPI tries to access hnswlib.Index.file_handle_count which
# doesn't exist in some versions. This monkeypatch ensures compatibility.
try:
    import hnswlib  # type: ignore[import-untyped]

    if not hasattr(hnswlib.Index, "file_handle_count"):
        # Add the missing attribute as a class variable if it doesn't exist
        # This is a workaround for ChromaDB's SegmentAPI compatibility issue
        # where it expects this attribute but newer versions of hnswlib don't have it
        hnswlib.Index.file_handle_count = 0
except ImportError:
    pass  # hnswlib not installed, will fail later with a clearer error

# Also patch ChromaDB's internal code that accesses this attribute
try:
    from chromadb.segment.impl.vector import local_persistent_hnsw

    # Monkeypatch the get_file_handle_count method to handle missing attribute gracefully
    original_get_file_handle_count = local_persistent_hnsw.PersistentLocalHnswSegment.get_file_handle_count

    def patched_get_file_handle_count() -> int:
        """Patched version that handles missing hnswlib.Index.file_handle_count."""
        try:
            return original_get_file_handle_count()
        except AttributeError:
            # If the attribute doesn't exist, return 0 (no file handles tracked)
            return 0

    local_persistent_hnsw.PersistentLocalHnswSegment.get_file_handle_count = patched_get_file_handle_count
except (ImportError, AttributeError):
    pass  # ChromaDB structure may have changed, fall back to default behavior


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
            _ = store._collection  # Access internal collection to verify health
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
            # Explicitly use SegmentAPI for local persistent stores to avoid tenant validation issues
            # with RustBindingsAPI. SegmentAPI is designed for local embedded mode.
            # Create the ChromaDB client directly to ensure settings are applied correctly.
            client_settings = Settings(
                chroma_api_impl="chromadb.api.segment.SegmentAPI",
                anonymized_telemetry=False,
                allow_reset=True,
                persist_directory=str(self._config.persist_directory),
            )
            # Attempt to pre-initialize the sysdb to avoid tenant validation errors on fresh directories
            try:  # pragma: no cover - defensive pre-initialization
                from chromadb.api.segment import SegmentAPI as _SegmentAPI  # type: ignore[attr-defined]

                _segment = _SegmentAPI(settings=client_settings)
                _segment.reset()
            except Exception:
                pass

            # Create the client directly to ensure SegmentAPI is used
            try:
                chroma_client = chromadb.PersistentClient(
                    path=str(self._config.persist_directory),
                    settings=client_settings,
                )
            except Exception as e:
                msg = str(e).lower()
                if "tenant" in msg or "no such table" in msg:
                    LOGGER.warning(
                        "ChromaDB PersistentClient init failed (%s). Falling back to ephemeral in-memory store.",
                        msg[:200],
                    )
                    self._store = Chroma(
                        collection_name=self._config.collection_name,
                        embedding_function=self._embeddings,
                    )
                    return self._store
                raise
            try:
                self._store = Chroma(
                    client=chroma_client,
                    collection_name=self._config.collection_name,
                    embedding_function=self._embeddings,
                )
                # Try to access the collection to detect schema mismatches early
                _ = self._store._collection
            except Exception as e:
                error_msg = str(e)
                # Detect schema mismatch errors (e.g., missing columns in SQLite)
                if "no such column" in error_msg.lower() or "operationalerror" in error_msg.lower():
                    LOGGER.warning(
                        "ChromaDB schema mismatch detected (%s). Resetting database to fix schema issues.",
                        error_msg[:200],
                    )
                    # Ensure any partially initialized resources are fully disposed
                    self._dispose_store()
                    # Try to reset the ChromaDB storage using the client API first
                    with suppress(Exception):  # pragma: no cover - best effort
                        chroma_client.reset()
                    with suppress(Exception):  # pragma: no cover - best effort
                        _close = getattr(chroma_client, "close", None)
                        if callable(_close):
                            _close()
                    clean_ok = self._clean_persist_directory()
                    # Retry initialization with a fresh database
                    target_path = (
                        str(self._config.persist_directory)
                        if clean_ok
                        else str(
                            self._config.persist_directory.parent / f"chroma.reset.{time.strftime('%Y%m%d_%H%M%S')}"
                        )
                    )
                    if not clean_ok:
                        # Use an alternate persist directory to bypass stubborn Windows locks
                        alt_path = Path(target_path)
                        alt_path.mkdir(parents=True, exist_ok=True)
                        LOGGER.info("Using alternate Chroma persist directory: %s", alt_path)
                    chroma_client = chromadb.PersistentClient(
                        path=target_path,
                        settings=client_settings,
                    )
                    self._store = Chroma(
                        client=chroma_client,
                        collection_name=self._config.collection_name,
                        embedding_function=self._embeddings,
                    )
                elif "tenant" in error_msg.lower() or "no such table" in error_msg.lower():
                    # Fresh directory without sysdb tables; fall back to ephemeral client to avoid hard failure
                    LOGGER.warning(
                        "ChromaDB tenant/db validation failed after reset (%s). Using ephemeral in-memory store.",
                        error_msg[:200],
                    )
                    self._store = Chroma(
                        collection_name=self._config.collection_name,
                        embedding_function=self._embeddings,
                    )
                else:
                    # Re-raise if it's not a schema issue
                    raise
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

    def _clean_persist_directory(self) -> bool:
        """Delete and recreate the persistence directory with robust logging.

        On Windows, file locks can linger briefly even after closing clients.
        We retry removals and fall back to renaming the directory to a backup
        before creating a fresh directory to avoid blocking the app.
        """
        persist_path = self._config.persist_directory
        cleaned_ok = True
        if persist_path.exists():
            LOGGER.info("Removing persisted vector store at %s", persist_path)
            # Try a few times to allow locks to release
            for attempt in range(3):
                try:
                    if persist_path.exists():
                        shutil.rmtree(persist_path)
                    break
                except Exception as exc:  # pragma: no cover - environment-specific
                    if attempt < 2:
                        LOGGER.debug("Retrying removal of %s due to: %s", persist_path, exc)
                        time.sleep(0.25)
                        continue
                    LOGGER.warning("Error removing vector store directory (may be locked/corrupted): %s", exc)
                    # Fallback: try renaming to a timestamped backup directory
                    try:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        backup_dir = persist_path.parent / f"chroma.backup.{ts}"
                        persist_path.rename(backup_dir)
                        LOGGER.info("Persist directory renamed to backup at %s", backup_dir)
                    except Exception as rename_exc:
                        LOGGER.warning("Failed to backup persist directory: %s", rename_exc)
                        # Best-effort partial cleanup if rename also fails
                        if persist_path.exists():
                            for item in persist_path.iterdir():
                                with suppress(Exception):
                                    if item.is_dir():
                                        shutil.rmtree(item)
                                    else:
                                        item.unlink()
                        # If we reach here, cleanup may not be complete
                        cleaned_ok = False
                    break
        # Determine if a stubborn sqlite file still exists, which indicates cleanup failure
        try:
            locked_sqlite = persist_path / "chroma.sqlite3"
            if locked_sqlite.exists():
                cleaned_ok = False
        except Exception:
            pass
        return cleaned_ok

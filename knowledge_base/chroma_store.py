"""
knowledge_base/chroma_store.py
───────────────────────────────
ChromaDB vector store manager.
Handles creation, loading, and semantic retrieval of document chunks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import logfire
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings

logger = logging.getLogger(__name__)


class ChromaStore:
    """
    Thin wrapper around LangChain's Chroma integration.

    Usage
    -----
    store = ChromaStore()
    store.add_documents(docs)
    results = store.similarity_search("what is RAG?", k=4)
    """

    def __init__(self) -> None:
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.embeddings_model,
        )
        self._persist_dir = str(settings.chroma_path)
        self._collection_name = settings.chroma_collection_name

        # Ensure persistence directory exists
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        self._db: Chroma = self._load_or_create()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_or_create(self) -> Chroma:
        """Load existing index or create a fresh one."""
        with logfire.span("chroma.load_or_create"):
            db = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embeddings,
                persist_directory=self._persist_dir,
            )
            count = db._collection.count()
            logfire.info(
                "ChromaDB ready",
                collection=self._collection_name,
                doc_count=count,
            )
            logger.info(
                "ChromaDB loaded — collection=%s, documents=%d",
                self._collection_name,
                count,
            )
            return db

    # ── Public API ────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add pre-chunked documents to the vector store.
        Duplicate detection is handled by ChromaDB's internal ID system.
        """
        with logfire.span("chroma.add_documents", count=len(documents)):
            self._db.add_documents(documents)
            logfire.info("Documents indexed", count=len(documents))
            logger.info("Indexed %d document chunks into ChromaDB", len(documents))

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: float = 0.0,
    ) -> List[Document]:
        """
        Perform semantic similarity search.

        Parameters
        ----------
        query : str
            The search query.
        k : int, optional
            Number of results to return. Defaults to settings.top_k_docs.
        score_threshold : float
            Minimum cosine similarity score (0–1). Docs below are dropped.

        Returns
        -------
        List[Document]
            Ranked list of relevant document chunks.
        """
        k = k or settings.top_k_docs
        with logfire.span("chroma.similarity_search", query=query, k=k):
            results = self._db.similarity_search_with_relevance_scores(
                query=query, k=k
            )
            # Filter by score threshold
            filtered = [doc for doc, score in results if score >= score_threshold]
            logfire.info(
                "Similarity search complete",
                query=query,
                total_results=len(results),
                after_filter=len(filtered),
            )
            return filtered

    def as_retriever(self, k: Optional[int] = None):
        """Return a LangChain-compatible retriever interface."""
        k = k or settings.top_k_docs
        return self._db.as_retriever(search_kwargs={"k": k})

    @property
    def document_count(self) -> int:
        """Total number of chunks in the collection."""
        return self._db._collection.count()

    def reset(self) -> None:
        """Delete all documents from the collection (useful for tests)."""
        self._db.delete_collection()
        self._db = self._load_or_create()
        logger.warning("ChromaDB collection reset — all documents deleted.")


# ── Singleton ──────────────────────────────────────────────────────────────────

_store_instance: Optional[ChromaStore] = None


def get_chroma_store() -> ChromaStore:
    """Return (or lazily create) the global ChromaStore singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ChromaStore()
    return _store_instance

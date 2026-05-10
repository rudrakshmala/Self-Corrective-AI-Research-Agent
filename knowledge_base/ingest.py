"""
knowledge_base/ingest.py
─────────────────────────
Document ingestion pipeline.

Supports: PDF, DOCX, TXT, web URLs, and raw text strings.
All documents are chunked and pushed into ChromaDB.

Usage (CLI)
-----------
python -m knowledge_base.ingest --source ./docs/my_paper.pdf
python -m knowledge_base.ingest --url https://example.com/article
python -m knowledge_base.ingest --text "Insert any raw text here..."
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import logfire
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from knowledge_base.chroma_store import get_chroma_store

logger = logging.getLogger(__name__)


# ── Text Splitter ─────────────────────────────────────────────────────────────

def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_file(file_path: str) -> List[Document]:
    """Auto-detect file type and load documents."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in (".docx", ".doc"):
        loader = Docx2txtLoader(str(path))
    elif suffix in (".txt", ".md"):
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()
    logger.info("Loaded %d pages from %s", len(docs), file_path)
    return docs


def _load_url(url: str) -> List[Document]:
    """Load content from a web URL."""
    loader = WebBaseLoader(web_paths=[url])
    docs = loader.load()
    logger.info("Loaded %d pages from URL: %s", len(docs), url)
    return docs


def _load_text(text: str, metadata: Optional[dict] = None) -> List[Document]:
    """Wrap raw text in a Document."""
    return [Document(page_content=text, metadata=metadata or {"source": "raw_text"})]


# ── Core Pipeline ─────────────────────────────────────────────────────────────

def ingest_documents(documents: List[Document]) -> int:
    """
    Chunk and index a list of Documents into ChromaDB.

    Returns
    -------
    int
        Number of chunks indexed.
    """
    with logfire.span("ingest.pipeline", doc_count=len(documents)):
        splitter = _get_splitter()
        chunks = splitter.split_documents(documents)

        logfire.info(
            "Chunking complete",
            original_docs=len(documents),
            chunks=len(chunks),
        )
        logger.info(
            "Split %d documents into %d chunks (size=%d, overlap=%d)",
            len(documents),
            len(chunks),
            settings.chunk_size,
            settings.chunk_overlap,
        )

        store = get_chroma_store()
        store.add_documents(chunks)
        return len(chunks)


def ingest_file(file_path: str) -> int:
    """Convenience wrapper: load file → chunk → index."""
    docs = _load_file(file_path)
    return ingest_documents(docs)


def ingest_url(url: str) -> int:
    """Convenience wrapper: load URL → chunk → index."""
    docs = _load_url(url)
    return ingest_documents(docs)


def ingest_text(text: str, metadata: Optional[dict] = None) -> int:
    """Convenience wrapper: raw text → chunk → index."""
    docs = _load_text(text, metadata)
    return ingest_documents(docs)


# ── CLI Entrypoint ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the ChromaDB knowledge base."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=str, help="Path to PDF/DOCX/TXT file")
    group.add_argument("--url", type=str, help="URL to scrape and ingest")
    group.add_argument("--text", type=str, help="Raw text string to ingest")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    try:
        if args.source:
            n = ingest_file(args.source)
        elif args.url:
            n = ingest_url(args.url)
        else:
            n = ingest_text(args.text)

        print(f"✅ Successfully indexed {n} chunks into ChromaDB.")
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        sys.exit(1)

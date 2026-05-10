"""
graph/nodes/retriever_node.py
──────────────────────────────
Retriever Node — fetches documents from either ChromaDB or Tavily.

Two sub-paths:
  retrieve_from_vectorstore  → ChromaDB semantic search
  retrieve_from_websearch    → Tavily API live search

Both paths normalise results into List[Document] so downstream
nodes are data-source-agnostic.
"""

from __future__ import annotations

import logging
from typing import List

import logfire
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document

from config import settings
from graph.state import GraphState
from knowledge_base.chroma_store import get_chroma_store

logger = logging.getLogger(__name__)


# ── VectorStore Retriever ──────────────────────────────────────────────────────

def retrieve_from_vectorstore(state: GraphState) -> dict:
    """
    LangGraph node: semantic search over ChromaDB.

    State mutations
    ---------------
    - Sets `retrieved_documents`
    - Appends to `reasoning_trace`
    """
    question = state["question"]

    with logfire.span("node.retriever.vectorstore", question=question):
        store = get_chroma_store()
        docs: List[Document] = store.similarity_search(
            query=question,
            k=settings.top_k_docs,
        )

        logfire.info(
            "VectorStore retrieval complete",
            question=question,
            docs_retrieved=len(docs),
        )
        logger.info(
            "VectorStore retrieved %d docs for: '%s'",
            len(docs),
            question,
        )

        trace_entry = (
            f"[Retriever/VectorStore] Retrieved {len(docs)} chunks "
            f"for query: '{question}'"
        )
        return {
            "retrieved_documents": docs,
            "reasoning_trace": [trace_entry],
        }


# ── Web Search Retriever ───────────────────────────────────────────────────────

def retrieve_from_websearch(state: GraphState) -> dict:
    """
    LangGraph node: live web search via Tavily API.

    Tavily results are converted to LangChain Documents so the
    Grade Documents node can operate without branching.

    State mutations
    ---------------
    - Sets `retrieved_documents`
    - Appends to `reasoning_trace`
    """
    question = state["question"]

    with logfire.span("node.retriever.websearch", question=question):
        tavily = TavilySearchResults(
            max_results=settings.top_k_docs,
            tavily_api_key=settings.tavily_api_key,
        )
        raw_results = tavily.invoke({"query": question})

        # Normalise Tavily dicts → LangChain Documents
        docs: List[Document] = [
            Document(
                page_content=r.get("content", ""),
                metadata={
                    "source": r.get("url", "unknown"),
                    "title": r.get("title", ""),
                    "score": r.get("score", 0.0),
                },
            )
            for r in raw_results
            if r.get("content")
        ]

        logfire.info(
            "WebSearch retrieval complete",
            question=question,
            docs_retrieved=len(docs),
        )
        logger.info(
            "WebSearch retrieved %d results for: '%s'",
            len(docs),
            question,
        )

        trace_entry = (
            f"[Retriever/WebSearch] Retrieved {len(docs)} web results "
            f"for query: '{question}'"
        )
        return {
            "retrieved_documents": docs,
            "reasoning_trace": [trace_entry],
        }

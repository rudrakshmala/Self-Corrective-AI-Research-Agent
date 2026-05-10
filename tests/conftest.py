"""
tests/conftest.py
──────────────────
Shared Pytest fixtures for the Self-Corrective RAG test suite.
All LLM calls and external dependencies are mocked to keep tests fast,
deterministic, and free of API costs.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from graph.state import GraphState
from models.schemas import (
    GradeDocuments,
    GradeHallucinations,
    GeneratedAnswer,
    RouteQuery,
)


# ── Sample Data ────────────────────────────────────────────────────────────────

SAMPLE_QUESTION = "What is Retrieval-Augmented Generation?"

SAMPLE_CONTEXT = (
    "Retrieval-Augmented Generation (RAG) is an AI framework that combines "
    "retrieval-based methods with generative language models. It retrieves "
    "relevant documents from a knowledge base and uses them as context to "
    "generate more accurate, grounded answers, reducing hallucinations."
)

SAMPLE_ANSWER = (
    "Retrieval-Augmented Generation (RAG) is an AI framework that combines "
    "retrieval with generation. It fetches relevant documents and uses them "
    "as grounding context for the LLM."
)

SAMPLE_DOCS: List[Document] = [
    Document(
        page_content=SAMPLE_CONTEXT,
        metadata={"source": "rag_paper.pdf", "page": 1},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications.",
        metadata={"source": "langgraph_docs.html", "page": 0},
    ),
]


# ── Base State Fixture ─────────────────────────────────────────────────────────

@pytest.fixture
def base_state() -> GraphState:
    """Minimal valid GraphState for node testing."""
    return {
        "question": SAMPLE_QUESTION,
        "query_source": "vectorstore",
        "retrieved_documents": SAMPLE_DOCS,
        "document_grade": "relevant",
        "generation_attempt": SAMPLE_ANSWER,
        "hallucination_score": 0.1,
        "hallucination_verdict": "grounded",
        "iteration_count": 0,
        "max_iterations": 3,
        "reasoning_trace": [],
        "final_answer": "",
        "termination_reason": "verified",
    }


@pytest.fixture
def empty_state() -> GraphState:
    """State with no retrieved documents (edge case)."""
    return {
        "question": SAMPLE_QUESTION,
        "query_source": "vectorstore",
        "retrieved_documents": [],
        "document_grade": "not_relevant",
        "generation_attempt": "",
        "hallucination_score": 0.0,
        "hallucination_verdict": "hallucinated",
        "iteration_count": 0,
        "max_iterations": 3,
        "reasoning_trace": [],
        "final_answer": "",
        "termination_reason": "verified",
    }


@pytest.fixture
def max_iter_state(base_state) -> GraphState:
    """State where max iterations have been reached."""
    return {**base_state, "iteration_count": 3, "max_iterations": 3}


# ── Structured Output Mocks ────────────────────────────────────────────────────

@pytest.fixture
def mock_route_query_vectorstore() -> RouteQuery:
    return RouteQuery(datasource="vectorstore", reasoning="Domain-specific question.")


@pytest.fixture
def mock_route_query_websearch() -> RouteQuery:
    return RouteQuery(datasource="websearch", reasoning="Requires live web data.")


@pytest.fixture
def mock_grade_relevant() -> GradeDocuments:
    return GradeDocuments(binary_score="yes", reasoning="Document directly addresses the question.")


@pytest.fixture
def mock_grade_irrelevant() -> GradeDocuments:
    return GradeDocuments(binary_score="no", reasoning="Document is about an unrelated topic.")


@pytest.fixture
def mock_grade_grounded() -> GradeHallucinations:
    return GradeHallucinations(
        binary_score="no",
        hallucination_score=0.05,
        reasoning="All claims are directly supported by the source documents.",
    )


@pytest.fixture
def mock_grade_hallucinated() -> GradeHallucinations:
    return GradeHallucinations(
        binary_score="yes",
        hallucination_score=0.85,
        reasoning="Answer claims X was invented in 1990, not found in any source.",
    )


@pytest.fixture
def mock_generated_answer() -> GeneratedAnswer:
    return GeneratedAnswer(
        reasoning_steps=[
            "Step 1: Identify the definition of RAG in the source documents.",
            "Step 2: Note the key components: retrieval + generation.",
            "Step 3: Synthesise a concise answer from these facts.",
        ],
        answer=SAMPLE_ANSWER,
        confidence=0.92,
    )


# ── LLM Chain Patcher ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_chain():
    """Context manager factory to mock any LLM chain's invoke call."""

    def _make(return_value):
        mock = MagicMock()
        mock.invoke.return_value = return_value
        return mock

    return _make

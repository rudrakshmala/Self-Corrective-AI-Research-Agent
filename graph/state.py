"""
graph/state.py
──────────────
Canonical TypedDict state shared across every node in the LangGraph pipeline.

Design notes:
  - `reasoning_trace` uses `operator.add` so any node can APPEND without
    overwriting previous steps (LangGraph merges using the reducer).
  - All optional fields default to empty / zero so nodes can safely read
    state without KeyError.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Literal, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    The single source of truth flowing through every LangGraph node.

    ┌─────────────────────────────────────────────────────────┐
    │  question          ← set once at graph entry            │
    │  query_source      ← set by Router Node                 │
    │  retrieved_docs    ← set by Retriever Node              │
    │  document_grade    ← set by Grade Documents Node        │
    │  generation_attempt← set/updated by Generate Node       │
    │  hallucination_score ← set by Hallucination Grader      │
    │  iteration_count   ← incremented each Generate loop     │
    │  reasoning_trace   ← APPENDED by every node (reducer)   │
    │  final_answer      ← set when graph exits               │
    │  termination_reason← why the loop ended                 │
    └─────────────────────────────────────────────────────────┘
    """

    # ── Input ──────────────────────────────────────────────────────────────────
    question: str
    """The original user question — immutable throughout the graph."""

    # ── Routing ────────────────────────────────────────────────────────────────
    query_source: Literal["vectorstore", "websearch"]
    """Set by the Router Node. Determines which retriever branch to follow."""

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retrieved_documents: List[Document]
    """
    Top-k document chunks returned by the active retriever.
    Reset each time the Retriever Node runs (after query rewriting).
    """

    # ── Document Grading ───────────────────────────────────────────────────────
    document_grade: Literal["relevant", "not_relevant"]
    """
    Aggregated verdict from the Grade Documents Node.
    'not_relevant' triggers query rewriting → web search fallback.
    """

    # ── Generation ─────────────────────────────────────────────────────────────
    generation_attempt: str
    """
    Latest synthesised answer from the Generate Node.
    Overwritten on every generation cycle.
    """

    # ── Hallucination Checking ─────────────────────────────────────────────────
    hallucination_score: float
    """
    Float 0.0–1.0. 0.0 = fully grounded, 1.0 = fully hallucinated.
    Set by the Hallucination Grader Node.
    """

    hallucination_verdict: Literal["grounded", "hallucinated"]
    """Binary verdict mapped from hallucination_score vs. threshold."""

    # ── Cycle Control ──────────────────────────────────────────────────────────
    iteration_count: int
    """
    Tracks how many times the Generate → Hallucination Grader loop has run.
    Graph exits when this reaches MAX_ITERATIONS (circuit breaker).
    """

    # ── Audit Trail ────────────────────────────────────────────────────────────
    reasoning_trace: Annotated[List[str], operator.add]
    """
    Append-only log. Each node appends its step summary.
    operator.add reducer prevents nodes from overwriting each other.
    """

    # ── Output ─────────────────────────────────────────────────────────────────
    final_answer: str
    """The accepted answer surfaced to the user after loop termination."""

    termination_reason: Literal["verified", "max_iterations_reached", "converged"]
    """Explains why the RAV loop stopped."""

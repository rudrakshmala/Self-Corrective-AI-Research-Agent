"""
graph/graph_builder.py
───────────────────────
Compiles the Self-Corrective RAG StateGraph.

Graph Topology
──────────────

  START
    │
    ▼
  [router_node]  ─── "vectorstore" ──► [retrieve_vectorstore]
                 └── "websearch"   ──► [retrieve_websearch]
                                              │
                                              ▼
                                     [grade_documents]
                                              │
                         ┌────────────────────┘
                         │
               "relevant"│                  │"not_relevant"
                         ▼                  ▼
                      [generate]    [query_transformer]
                         │                  │
                         ▼                  └──► [retrieve_websearch]
                [hallucination_grader]
                         │
          ┌──────────────┴────────────────┐
          │"grounded"              "hallucinated"
          │                               │
          │                   ┌───────────┘
          │                   │  iter < MAX_ITERATIONS
          │                   ▼
          │              [generate]  ◄─ (loop back, max 3 times)
          │
          ▼
        [END]

All conditional routing is handled by pure functions returning string keys
matched to the `path_map` dicts in `add_conditional_edges`.
"""

from __future__ import annotations

import logging

import logfire
from langgraph.graph import END, START, StateGraph

from config import settings
from graph.state import GraphState
from graph.nodes.router_node import router_node
from graph.nodes.retriever_node import retrieve_from_vectorstore, retrieve_from_websearch
from graph.nodes.grade_documents_node import grade_documents_node
from graph.nodes.query_transformer_node import query_transformer_node
from graph.nodes.generate_node import generate_node
from graph.nodes.hallucination_grader_node import hallucination_grader_node

logger = logging.getLogger(__name__)


# ── Conditional Edge Functions ─────────────────────────────────────────────────

def _route_after_router(state: GraphState) -> str:
    """Route based on the Router Node's datasource decision."""
    source = state.get("query_source", "vectorstore")
    logger.debug("Edge: router → %s", source)
    return source  # "vectorstore" or "websearch"


def _route_after_grading(state: GraphState) -> str:
    """
    Route based on document relevance:
      - "relevant"     → generate
      - "not_relevant" → transform_query
    """
    grade = state.get("document_grade", "not_relevant")
    logger.debug("Edge: grade_documents → %s", grade)
    return grade  # "relevant" or "not_relevant"


def _route_after_hallucination_check(state: GraphState) -> str:
    """
    Route based on hallucination verdict and iteration count.

    Returns
    -------
    "end"      → answer is grounded OR max iterations reached
    "generate" → answer is hallucinated AND iterations remain
    """
    verdict = state.get("hallucination_verdict", "hallucinated")
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", settings.max_iterations)

    if verdict == "grounded":
        logger.debug("Edge: hallucination_grader → END (grounded)")
        return "end"

    if iteration >= max_iter:
        logger.debug(
            "Edge: hallucination_grader → END (max_iterations=%d reached)", max_iter
        )
        return "end"

    logger.debug(
        "Edge: hallucination_grader → generate (hallucinated, iter=%d/%d)",
        iteration,
        max_iter,
    )
    return "generate"


# ── Graph Builder ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the Self-Corrective RAG StateGraph.

    Returns
    -------
    CompiledStateGraph
        A callable LangGraph object — invoke with:
        ``graph.invoke({"question": "...", "max_iterations": 3})``
    """
    with logfire.span("graph.build"):
        graph = StateGraph(GraphState)

        # ── Register Nodes ─────────────────────────────────────────────────────
        graph.add_node("router", router_node)
        graph.add_node("retrieve_vectorstore", retrieve_from_vectorstore)
        graph.add_node("retrieve_websearch", retrieve_from_websearch)
        graph.add_node("grade_documents", grade_documents_node)
        graph.add_node("query_transformer", query_transformer_node)
        graph.add_node("generate", generate_node)
        graph.add_node("hallucination_grader", hallucination_grader_node)

        # ── Entry Point ────────────────────────────────────────────────────────
        graph.add_edge(START, "router")

        # ── Router → Retriever (conditional) ──────────────────────────────────
        graph.add_conditional_edges(
            "router",
            _route_after_router,
            {
                "vectorstore": "retrieve_vectorstore",
                "websearch": "retrieve_websearch",
            },
        )

        # ── Both retrievers flow into grade_documents ──────────────────────────
        graph.add_edge("retrieve_vectorstore", "grade_documents")
        graph.add_edge("retrieve_websearch", "grade_documents")

        # ── Grade Documents → Generate OR Query Transform (conditional) ────────
        graph.add_conditional_edges(
            "grade_documents",
            _route_after_grading,
            {
                "relevant": "generate",
                "not_relevant": "query_transformer",
            },
        )

        # ── Query Transform → Web Search (always escalate) ─────────────────────
        graph.add_edge("query_transformer", "retrieve_websearch")

        # ── Generate → Hallucination Grader ────────────────────────────────────
        graph.add_edge("generate", "hallucination_grader")

        # ── Hallucination Grader → END or re-Generate (cyclic!) ───────────────
        graph.add_conditional_edges(
            "hallucination_grader",
            _route_after_hallucination_check,
            {
                "end": END,
                "generate": "generate",  # ← The cycle that eliminates hallucinations
            },
        )

        compiled = graph.compile()

        logfire.info("StateGraph compiled successfully")
        logger.info("✅ Self-Corrective RAG graph compiled.")

        return compiled


# ── Singleton compiled graph ───────────────────────────────────────────────────

_compiled_graph = None


def get_graph():
    """Return (or lazily build) the singleton compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph

"""
tests/test_graph.py
────────────────────
Integration tests for the compiled LangGraph pipeline.
Tests the full graph routing logic using mocked node functions
to verify the state machine wiring is correct.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document

from graph.graph_builder import build_graph, _route_after_router, _route_after_grading, _route_after_hallucination_check
from graph.state import GraphState


# ── Edge Routing Logic Tests ───────────────────────────────────────────────────

class TestRouterEdge:
    def test_routes_vectorstore(self, base_state):
        base_state["query_source"] = "vectorstore"
        assert _route_after_router(base_state) == "vectorstore"

    def test_routes_websearch(self, base_state):
        base_state["query_source"] = "websearch"
        assert _route_after_router(base_state) == "websearch"

    def test_defaults_to_vectorstore_when_missing(self, base_state):
        state = dict(base_state)
        state.pop("query_source", None)
        assert _route_after_router(state) == "vectorstore"


class TestGradingEdge:
    def test_relevant_routes_to_generate(self, base_state):
        base_state["document_grade"] = "relevant"
        assert _route_after_grading(base_state) == "relevant"

    def test_not_relevant_routes_to_transform(self, base_state):
        base_state["document_grade"] = "not_relevant"
        assert _route_after_grading(base_state) == "not_relevant"

    def test_defaults_to_not_relevant(self, base_state):
        state = dict(base_state)
        state.pop("document_grade", None)
        assert _route_after_grading(state) == "not_relevant"


class TestHallucinationEdge:
    def test_grounded_routes_to_end(self, base_state):
        base_state["hallucination_verdict"] = "grounded"
        base_state["iteration_count"] = 1
        assert _route_after_hallucination_check(base_state) == "end"

    def test_hallucinated_routes_to_generate(self, base_state):
        base_state["hallucination_verdict"] = "hallucinated"
        base_state["iteration_count"] = 1
        base_state["max_iterations"] = 3
        assert _route_after_hallucination_check(base_state) == "generate"

    def test_max_iterations_forces_end(self, base_state):
        base_state["hallucination_verdict"] = "hallucinated"
        base_state["iteration_count"] = 3
        base_state["max_iterations"] = 3
        assert _route_after_hallucination_check(base_state) == "end"

    def test_one_below_max_still_loops(self, base_state):
        base_state["hallucination_verdict"] = "hallucinated"
        base_state["iteration_count"] = 2
        base_state["max_iterations"] = 3
        assert _route_after_hallucination_check(base_state) == "generate"


# ── Graph Compilation Test ─────────────────────────────────────────────────────

class TestGraphCompilation:
    def test_graph_compiles_without_error(self):
        """Verify the StateGraph compiles without raising exceptions."""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_invoke_method(self):
        graph = build_graph()
        assert callable(getattr(graph, "invoke", None))

    def test_graph_get_graph_returns_callable(self):
        from graph.graph_builder import get_graph
        g = get_graph()
        assert g is not None


# ── Full Graph Invocation (Integration, heavily mocked) ────────────────────────

class TestGraphInvocation:
    """
    End-to-end graph invocation with all LLM calls mocked.
    Verifies the full state is populated correctly after a run.
    """

    def test_happy_path_vectorstore_grounded(
        self,
        base_state,
        mock_route_query_vectorstore,
        mock_grade_relevant,
        mock_generated_answer,
        mock_grade_grounded,
        mock_llm_chain,
    ):
        mock_docs = [Document(page_content="RAG context", metadata={"source": "test.pdf"})]
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = mock_docs

        router_chain = mock_llm_chain(mock_route_query_vectorstore)
        grade_chain = mock_llm_chain(mock_grade_relevant)
        generate_chain = mock_llm_chain(mock_generated_answer)
        halluc_chain = mock_llm_chain(mock_grade_grounded)

        with (
            patch("graph.nodes.router_node._build_router_chain", return_value=router_chain),
            patch("graph.nodes.retriever_node.get_chroma_store", return_value=mock_store),
            patch("graph.nodes.grade_documents_node._build_grader_chain", return_value=grade_chain),
            patch("graph.nodes.generate_node._build_generate_chain", return_value=generate_chain),
            patch("graph.nodes.hallucination_grader_node._build_grader_chain", return_value=halluc_chain),
        ):
            graph = build_graph()
            result = graph.invoke(
                {
                    "question": "What is RAG?",
                    "iteration_count": 0,
                    "max_iterations": 3,
                    "reasoning_trace": [],
                    "retrieved_documents": [],
                    "generation_attempt": "",
                    "hallucination_score": 0.0,
                    "final_answer": "",
                }
            )

        assert result["final_answer"] != ""
        assert result["termination_reason"] == "verified"
        assert result["hallucination_verdict"] == "grounded"
        assert result["iteration_count"] == 1
        assert len(result["reasoning_trace"]) >= 4  # Router + Retriever + Grader + Generate + Halluc

    def test_hallucination_loop_terminates_at_max(
        self,
        base_state,
        mock_route_query_vectorstore,
        mock_grade_relevant,
        mock_generated_answer,
        mock_grade_hallucinated,
        mock_llm_chain,
    ):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(page_content="Some context", metadata={"source": "doc.pdf"})
        ]

        router_chain = mock_llm_chain(mock_route_query_vectorstore)
        grade_chain = mock_llm_chain(mock_grade_relevant)
        generate_chain = mock_llm_chain(mock_generated_answer)
        halluc_chain = mock_llm_chain(mock_grade_hallucinated)

        with (
            patch("graph.nodes.router_node._build_router_chain", return_value=router_chain),
            patch("graph.nodes.retriever_node.get_chroma_store", return_value=mock_store),
            patch("graph.nodes.grade_documents_node._build_grader_chain", return_value=grade_chain),
            patch("graph.nodes.generate_node._build_generate_chain", return_value=generate_chain),
            patch("graph.nodes.hallucination_grader_node._build_grader_chain", return_value=halluc_chain),
        ):
            graph = build_graph()
            result = graph.invoke(
                {
                    "question": "What is RAG?",
                    "iteration_count": 0,
                    "max_iterations": 3,
                    "reasoning_trace": [],
                    "retrieved_documents": [],
                    "generation_attempt": "",
                    "hallucination_score": 0.0,
                    "final_answer": "",
                }
            )

        # Should terminate at max_iterations with warning
        assert result["termination_reason"] == "max_iterations_reached"
        assert "⚠️" in result["final_answer"]
        assert result["iteration_count"] == 3

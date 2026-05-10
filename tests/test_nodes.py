"""
tests/test_nodes.py — Unit tests for every graph node.
All LLM/API calls are mocked for speed and determinism.
"""

from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document

from graph.nodes.router_node import router_node
from graph.nodes.retriever_node import retrieve_from_vectorstore, retrieve_from_websearch
from graph.nodes.grade_documents_node import grade_documents_node
from graph.nodes.query_transformer_node import query_transformer_node
from graph.nodes.generate_node import generate_node
from graph.nodes.hallucination_grader_node import hallucination_grader_node


class TestRouterNode:
    def test_routes_to_vectorstore(self, base_state, mock_route_query_vectorstore, mock_llm_chain):
        chain = mock_llm_chain(mock_route_query_vectorstore)
        with patch("graph.nodes.router_node._build_router_chain", return_value=chain):
            result = router_node(base_state)
        assert result["query_source"] == "vectorstore"
        assert "[Router]" in result["reasoning_trace"][0]

    def test_routes_to_websearch(self, base_state, mock_route_query_websearch, mock_llm_chain):
        chain = mock_llm_chain(mock_route_query_websearch)
        with patch("graph.nodes.router_node._build_router_chain", return_value=chain):
            result = router_node(base_state)
        assert result["query_source"] == "websearch"


class TestRetrieverNode:
    def test_vectorstore_retrieval(self, base_state):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(page_content="RAG info", metadata={"source": "test"})
        ]
        with patch("graph.nodes.retriever_node.get_chroma_store", return_value=mock_store):
            result = retrieve_from_vectorstore(base_state)
        assert len(result["retrieved_documents"]) == 1
        assert "[Retriever/VectorStore]" in result["reasoning_trace"][0]

    def test_websearch_normalises_to_documents(self, base_state):
        mock_tavily = MagicMock()
        mock_tavily.invoke.return_value = [
            {"content": "RAG explained", "url": "https://example.com", "title": "RAG", "score": 0.9}
        ]
        with patch("graph.nodes.retriever_node.TavilySearchResults", return_value=mock_tavily):
            result = retrieve_from_websearch(base_state)
        docs = result["retrieved_documents"]
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].metadata["source"] == "https://example.com"

    def test_websearch_filters_empty_content(self, base_state):
        mock_tavily = MagicMock()
        mock_tavily.invoke.return_value = [
            {"content": "", "url": "https://bad.com"},
            {"content": "Valid content", "url": "https://good.com"},
        ]
        with patch("graph.nodes.retriever_node.TavilySearchResults", return_value=mock_tavily):
            result = retrieve_from_websearch(base_state)
        assert len(result["retrieved_documents"]) == 1


class TestGradeDocumentsNode:
    def test_relevant_docs(self, base_state, mock_grade_relevant, mock_llm_chain):
        chain = mock_llm_chain(mock_grade_relevant)
        with patch("graph.nodes.grade_documents_node._build_grader_chain", return_value=chain):
            result = grade_documents_node(base_state)
        assert result["document_grade"] == "relevant"

    def test_irrelevant_docs(self, base_state, mock_grade_irrelevant, mock_llm_chain):
        chain = mock_llm_chain(mock_grade_irrelevant)
        with patch("graph.nodes.grade_documents_node._build_grader_chain", return_value=chain):
            result = grade_documents_node(base_state)
        assert result["document_grade"] == "not_relevant"

    def test_empty_docs_returns_not_relevant(self, empty_state):
        result = grade_documents_node(empty_state)
        assert result["document_grade"] == "not_relevant"


class TestQueryTransformerNode:
    def test_rewrites_query_and_forces_websearch(self, base_state):
        mock_response = MagicMock()
        mock_response.content = "  Rewritten query for better search  "
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response
        with patch("graph.nodes.query_transformer_node._build_transformer_chain", return_value=mock_chain):
            result = query_transformer_node(base_state)
        assert result["question"] == "Rewritten query for better search"
        assert result["query_source"] == "websearch"
        assert "[QueryTransformer]" in result["reasoning_trace"][0]


class TestGenerateNode:
    def test_generates_answer_and_increments_iteration(self, base_state, mock_generated_answer, mock_llm_chain):
        chain = mock_llm_chain(mock_generated_answer)
        with patch("graph.nodes.generate_node._build_generate_chain", return_value=chain):
            result = generate_node(base_state)
        assert result["generation_attempt"] == mock_generated_answer.answer
        assert result["iteration_count"] == 1

    def test_iteration_increments_correctly(self, base_state, mock_generated_answer, mock_llm_chain):
        base_state["iteration_count"] = 2
        chain = mock_llm_chain(mock_generated_answer)
        with patch("graph.nodes.generate_node._build_generate_chain", return_value=chain):
            result = generate_node(base_state)
        assert result["iteration_count"] == 3

    def test_trace_contains_cot_steps(self, base_state, mock_generated_answer, mock_llm_chain):
        chain = mock_llm_chain(mock_generated_answer)
        with patch("graph.nodes.generate_node._build_generate_chain", return_value=chain):
            result = generate_node(base_state)
        assert "Chain-of-Thought" in result["reasoning_trace"][0]


class TestHallucinationGraderNode:
    def test_grounded_promotes_to_final_answer(self, base_state, mock_grade_grounded, mock_llm_chain):
        chain = mock_llm_chain(mock_grade_grounded)
        with patch("graph.nodes.hallucination_grader_node._build_grader_chain", return_value=chain):
            result = hallucination_grader_node(base_state)
        assert result["hallucination_verdict"] == "grounded"
        assert result["final_answer"] == base_state["generation_attempt"]
        assert result["termination_reason"] == "verified"

    def test_hallucinated_no_final_answer(self, base_state, mock_grade_hallucinated, mock_llm_chain):
        chain = mock_llm_chain(mock_grade_hallucinated)
        with patch("graph.nodes.hallucination_grader_node._build_grader_chain", return_value=chain):
            result = hallucination_grader_node(base_state)
        assert result["hallucination_verdict"] == "hallucinated"
        assert result.get("final_answer", "") == ""

    def test_max_iterations_forces_exit_with_warning(self, max_iter_state, mock_grade_hallucinated, mock_llm_chain):
        chain = mock_llm_chain(mock_grade_hallucinated)
        with patch("graph.nodes.hallucination_grader_node._build_grader_chain", return_value=chain):
            result = hallucination_grader_node(max_iter_state)
        assert result["termination_reason"] == "max_iterations_reached"
        assert "⚠️" in result["final_answer"]

    def test_score_stored_correctly(self, base_state, mock_grade_grounded, mock_llm_chain):
        chain = mock_llm_chain(mock_grade_grounded)
        with patch("graph.nodes.hallucination_grader_node._build_grader_chain", return_value=chain):
            result = hallucination_grader_node(base_state)
        assert result["hallucination_score"] == pytest.approx(0.05)

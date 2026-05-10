"""
graph/nodes/router_node.py
──────────────────────────
Router Node — first node executed in the graph.

Responsibility
--------------
Analyse the user's question and decide which data source to query:
  - "vectorstore" → internal ChromaDB (domain knowledge, static docs)
  - "websearch"   → Tavily live search (current events, unknown topics)

Uses structured output (RouteQuery Pydantic model) to guarantee
the LLM returns a parseable, type-safe decision.
"""

from __future__ import annotations

import logging

import logfire
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import settings
from graph.state import GraphState
from models.schemas import RouteQuery

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert query router for a Retrieval-Augmented Generation system.

Your task is to decide which data source should be queried to best answer the user's question.

Rules:
1. Route to **vectorstore** when:
   - The question is about a specific domain (e.g., internal company docs, research papers, technical manuals)
   - The answer is likely in static, pre-indexed documents
   - The question does NOT involve recent events or real-time data

2. Route to **websearch** when:
   - The question involves current events, news, or live data
   - The question is about a topic unlikely to be in internal documents
   - The question explicitly asks for "latest", "recent", or "today"

Be decisive. Choose exactly one data source."""

_HUMAN_PROMPT = "Question: {question}"

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
)

# ── LLM with Structured Output ────────────────────────────────────────────────

def _build_router_chain():
    llm = ChatGroq(
        model=settings.router_model,
        temperature=0.0,
        groq_api_key=settings.groq_api_key,
    )
    return _PROMPT | llm.with_structured_output(RouteQuery)


# ── Node Function ─────────────────────────────────────────────────────────────

def router_node(state: GraphState) -> dict:
    """
    LangGraph node: routes the query to vectorstore or websearch.

    State mutations
    ---------------
    - Sets `query_source`
    - Appends to `reasoning_trace`
    """
    question = state["question"]

    with logfire.span("node.router", question=question):
        chain = _build_router_chain()
        result: RouteQuery = chain.invoke({"question": question})

        logfire.info(
            "Router decision",
            datasource=result.datasource,
            reasoning=result.reasoning,
        )
        logger.info(
            "Router → %s | reason: %s",
            result.datasource,
            result.reasoning,
        )

        trace_entry = (
            f"[Router] Source='{result.datasource}' | "
            f"Reasoning='{result.reasoning}'"
        )

        return {
            "query_source": result.datasource,
            "reasoning_trace": [trace_entry],
        }

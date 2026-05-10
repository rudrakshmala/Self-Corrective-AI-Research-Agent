"""
graph/nodes/query_transformer_node.py
──────────────────────────────────────
Query Transformer Node — rewrites the original question.

Triggered when the Grade Documents Node returns "not_relevant".

Responsibility
--------------
Use an LLM to reformulate the user's query into a semantically
richer, more search-engine-friendly version, then route back
to Web Search retrieval for a second attempt.

This implements the "Corrective RAG" pattern:
  Grade → not_relevant → Transform → WebSearch → Grade → ...
"""

from __future__ import annotations

import logging

import logfire
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import settings
from graph.state import GraphState

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert query reformulation specialist for search systems.

Your task: Rewrite the user's question into a better search query.

Techniques to apply:
1. Expand abbreviations and acronyms
2. Add synonyms for key terms
3. Make implicit context explicit
4. Decompose compound questions
5. Remove filler words that confuse retrievers

Output ONLY the rewritten query — no explanations, no preamble."""

_HUMAN_PROMPT = """Original question: {question}

Rewritten search query:"""

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
)


def _build_transformer_chain():
    llm = ChatGroq(
        model=settings.grader_model,
        temperature=0.0,
        groq_api_key=settings.groq_api_key,
    )
    return _PROMPT | llm


# ── Node Function ─────────────────────────────────────────────────────────────

def query_transformer_node(state: GraphState) -> dict:
    """
    LangGraph node: rewrites the question for better web search retrieval.

    State mutations
    ---------------
    - Overwrites `question` with the improved query
    - Forces `query_source` → "websearch" (always falls back to web)
    - Appends to `reasoning_trace`
    """
    original_question = state["question"]

    with logfire.span("node.query_transformer", original=original_question):
        chain = _build_transformer_chain()
        rewritten = chain.invoke({"question": original_question})

        # Extract string from AIMessage
        new_question: str = (
            rewritten.content
            if hasattr(rewritten, "content")
            else str(rewritten)
        ).strip()

        logfire.info(
            "Query transformed",
            original=original_question,
            rewritten=new_question,
        )
        logger.info(
            "Query Transformer: '%s' → '%s'",
            original_question,
            new_question,
        )

        trace_entry = (
            f"[QueryTransformer] Original: '{original_question}' → "
            f"Rewritten: '{new_question}' | Forcing WebSearch fallback"
        )

        return {
            "question": new_question,
            "query_source": "websearch",  # Always escalate to web after transform
            "reasoning_trace": [trace_entry],
        }

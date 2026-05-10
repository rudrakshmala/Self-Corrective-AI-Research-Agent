"""
graph/nodes/generate_node.py
─────────────────────────────
Generate Node — synthesises the final answer using Chain-of-Thought.

Responsibility
--------------
Given the retrieved (and graded) documents and the user's question,
produce a precise, grounded answer by:
  1. Listing explicit CoT reasoning steps
  2. Citing only facts present in the context
  3. Flagging uncertainty when context is insufficient

Uses GeneratedAnswer structured output to enforce CoT at the schema level.
Increments `iteration_count` each time it runs (loop counter for the
hallucination re-generation circuit breaker).
"""

from __future__ import annotations

import logging
from typing import List

import logfire
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import settings
from graph.state import GraphState
from models.schemas import GeneratedAnswer

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a precise research assistant that answers questions strictly from provided context.

CRITICAL RULES:
1. ONLY use information present in the provided context documents
2. If the context does not contain the answer, say "The provided context does not contain enough information to answer this question."
3. Do NOT use any prior knowledge or training data beyond what is in the context
4. Structure your reasoning step-by-step before giving the final answer
5. Every factual claim MUST be traceable to the context

Hallucination = stating facts not in the context. This is strictly forbidden."""

_HUMAN_PROMPT = """Question: {question}

Context Documents:
---
{context}
---

Using ONLY the information in the context above, answer the question step-by-step."""

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
)


def _format_context(documents: List[Document]) -> str:
    """Format retrieved docs into a numbered context block."""
    if not documents:
        return "No context documents available."
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Document {i}] (Source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def _build_generate_chain():
    llm = ChatGroq(
        model=settings.generator_model,
        temperature=settings.temperature,
        groq_api_key=settings.groq_api_key,
    )
    return _PROMPT | llm.with_structured_output(GeneratedAnswer)


# ── Node Function ─────────────────────────────────────────────────────────────

def generate_node(state: GraphState) -> dict:
    """
    LangGraph node: synthesises a grounded answer using Chain-of-Thought.

    State mutations
    ---------------
    - Sets `generation_attempt` (the new answer text)
    - Increments `iteration_count`
    - Appends to `reasoning_trace`
    """
    question = state["question"]
    documents: List[Document] = state.get("retrieved_documents", [])
    iteration = state.get("iteration_count", 0)

    with logfire.span(
        "node.generate",
        question=question,
        iteration=iteration,
        doc_count=len(documents),
    ):
        context = _format_context(documents)
        chain = _build_generate_chain()

        result: GeneratedAnswer = chain.invoke(
            {"question": question, "context": context}
        )

        logfire.info(
            "Generation complete",
            iteration=iteration + 1,
            confidence=result.confidence,
            cot_steps=len(result.reasoning_steps),
        )
        logger.info(
            "Generate [iter=%d]: confidence=%.2f | CoT steps=%d",
            iteration + 1,
            result.confidence,
            len(result.reasoning_steps),
        )

        cot_summary = "\n".join(
            f"  Step {j}: {s}" for j, s in enumerate(result.reasoning_steps, 1)
        )
        trace_entry = (
            f"[Generate] Iteration={iteration + 1} | "
            f"Confidence={result.confidence:.2f}\n"
            f"Chain-of-Thought:\n{cot_summary}\n"
            f"Answer: {result.answer[:200]}..."
        )

        return {
            "generation_attempt": result.answer,
            "iteration_count": iteration + 1,
            "reasoning_trace": [trace_entry],
        }

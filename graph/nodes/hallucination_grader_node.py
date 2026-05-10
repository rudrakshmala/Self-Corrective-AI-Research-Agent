"""
graph/nodes/hallucination_grader_node.py
─────────────────────────────────────────
Hallucination Grader Node — the central auditor of the RAV loop.

Responsibility
--------------
Perform a strict binary check:
  Is EVERY factual claim in `generation_attempt` fully supported
  by the retrieved documents in `retrieved_documents`?

Uses GradeHallucinations structured output (Pydantic) to guarantee:
  - A parseable binary_score ("yes" = hallucinated, "no" = grounded)
  - A float hallucination_score for observability dashboards
  - A specific critique for the reasoning trace

Routing logic (handled by graph_builder conditional edges):
  - binary_score = "no"  AND hallucination_score < threshold → VERIFIED
  - binary_score = "yes" AND iteration < max               → re-Generate
  - iteration >= max                                        → force exit
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
from models.schemas import GradeHallucinations

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a rigorous hallucination detection expert for AI systems.

Your task: Audit the generated answer against the provided source documents.

Detection criteria — flag as hallucinated ("yes") if the answer:
  1. States any fact NOT present in the source documents
  2. Makes numerical claims not in the source (wrong dates, stats, figures)
  3. Names entities (people, places, organisations) not mentioned in sources
  4. Draws conclusions that go beyond what the documents support
  5. Contradicts information in the source documents

Grade as grounded ("no") ONLY if EVERY factual claim is directly traceable
to the provided documents.

hallucination_score: Confidence that hallucination exists.
  0.0 = 100% grounded in sources
  1.0 = 100% hallucinated / fabricated
  Use values in between for partial hallucination."""

_HUMAN_PROMPT = """Source Documents:
---
{context}
---

Generated Answer:
---
{answer}
---

Does the generated answer contain hallucinations or unsupported claims?"""

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
)


def _format_context(documents: List[Document]) -> str:
    if not documents:
        return "No source documents provided."
    return "\n\n".join(
        f"[Doc {i}]: {doc.page_content}" for i, doc in enumerate(documents, 1)
    )


def _build_grader_chain():
    llm = ChatGroq(
        model=settings.grader_model,
        temperature=0.0,
        groq_api_key=settings.groq_api_key,
    )
    return _PROMPT | llm.with_structured_output(GradeHallucinations)


# ── Node Function ─────────────────────────────────────────────────────────────

def hallucination_grader_node(state: GraphState) -> dict:
    """
    LangGraph node: audits the generated answer for hallucinations.

    State mutations
    ---------------
    - Sets `hallucination_score`
    - Sets `hallucination_verdict` → "grounded" | "hallucinated"
    - If grounded: sets `final_answer` and `termination_reason`
    - Appends to `reasoning_trace`
    """
    question = state["question"]
    answer = state.get("generation_attempt", "")
    documents: List[Document] = state.get("retrieved_documents", [])
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", settings.max_iterations)

    with logfire.span(
        "node.hallucination_grader",
        iteration=iteration,
        answer_length=len(answer),
    ):
        context = _format_context(documents)
        chain = _build_grader_chain()

        result: GradeHallucinations = chain.invoke(
            {"context": context, "answer": answer}
        )

        # Map structured output → internal verdict
        is_hallucinated = (
            result.binary_score == "yes"
            or result.hallucination_score >= settings.hallucination_threshold
        )
        verdict = "hallucinated" if is_hallucinated else "grounded"

        # Determine termination
        force_exit = iteration >= max_iter
        if not is_hallucinated or force_exit:
            final_answer = answer
            if force_exit and is_hallucinated:
                reason = "max_iterations_reached"
                final_answer = (
                    f"⚠️ [Max iterations reached — answer may contain inaccuracies]\n\n"
                    f"{answer}"
                )
            else:
                reason = "verified"
        else:
            final_answer = ""
            reason = ""

        logfire.info(
            "Hallucination grading complete",
            verdict=verdict,
            hallucination_score=result.hallucination_score,
            iteration=iteration,
            force_exit=force_exit,
        )
        logger.info(
            "Hallucination Grader [iter=%d]: verdict=%s score=%.2f | %s",
            iteration,
            verdict,
            result.hallucination_score,
            result.reasoning[:100],
        )

        trace_entry = (
            f"[HallucinationGrader] Iteration={iteration} | "
            f"Verdict='{verdict}' | Score={result.hallucination_score:.2f}\n"
            f"  Critique: {result.reasoning}"
        )

        update: dict = {
            "hallucination_score": result.hallucination_score,
            "hallucination_verdict": verdict,
            "reasoning_trace": [trace_entry],
        }

        if final_answer:
            update["final_answer"] = final_answer
            update["termination_reason"] = reason

        return update

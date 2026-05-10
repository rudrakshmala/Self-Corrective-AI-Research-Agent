"""
models/schemas.py
─────────────────
All Pydantic structured-output models used by grading nodes.
Using structured output ensures the graph NEVER breaks on LLM parse errors.
"""

from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field


# ── Router ────────────────────────────────────────────────────────────────────

class RouteQuery(BaseModel):
    """
    Structured output from the Router Node.
    Decides whether the question should be answered from the
    internal VectorStore or via live Web Search (Tavily).
    """

    datasource: Literal["vectorstore", "websearch"] = Field(
        description=(
            "Route to 'vectorstore' for domain-specific / internal knowledge. "
            "Route to 'websearch' for current events, news, or unknown topics."
        )
    )
    reasoning: str = Field(
        description="One-sentence explanation of the routing decision."
    )


# ── Document Grader ───────────────────────────────────────────────────────────

class GradeDocuments(BaseModel):
    """
    Structured output from the Grade Documents Node.
    Binary relevance gate — irrelevant docs trigger query rewriting.
    """

    binary_score: Literal["yes", "no"] = Field(
        description=(
            "'yes' if the document contains information relevant to the question. "
            "'no' if the document is off-topic or unhelpful."
        )
    )
    reasoning: str = Field(
        description="Brief citation-level explanation of the relevance decision."
    )


# ── Hallucination Grader ──────────────────────────────────────────────────────

class GradeHallucinations(BaseModel):
    """
    Structured output from the Hallucination Grader Node.
    Performs a strict grounding check — every claim must be traceable
    to the retrieved documents.
    """

    binary_score: Literal["yes", "no"] = Field(
        description=(
            "'yes' if the answer contains hallucinations or unsupported claims. "
            "'no' if the answer is fully grounded in the provided context."
        )
    )
    hallucination_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence that hallucination exists. "
            "0.0 = fully grounded, 1.0 = fully hallucinated."
        ),
    )
    reasoning: str = Field(
        description=(
            "Specific claims flagged as ungrounded, or confirmation that all "
            "claims are sourced."
        )
    )


# ── Answer Grader (Answer Relevance) ──────────────────────────────────────────

class GradeAnswer(BaseModel):
    """
    Structured output confirming the final answer addresses the question.
    Used as an optional post-filter before surfacing to the user.
    """

    binary_score: Literal["yes", "no"] = Field(
        description=(
            "'yes' if the answer directly addresses the user's question. "
            "'no' if the answer is off-topic or incomplete."
        )
    )
    reasoning: str = Field(description="Brief explanation.")


# ── Generated Answer ──────────────────────────────────────────────────────────

class GeneratedAnswer(BaseModel):
    """
    Structured output from the Generate Node.
    Enforces Chain-of-Thought by requiring explicit reasoning_steps.
    """

    reasoning_steps: List[str] = Field(
        description=(
            "Ordered list of reasoning steps (Chain-of-Thought) used to "
            "synthesise the answer from the retrieved context."
        )
    )
    answer: str = Field(
        description=(
            "The final, concise answer derived exclusively from the provided context. "
            "Do NOT introduce information not present in the documents."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-assessed confidence in the answer given available context.",
    )

"""
graph/nodes/grade_documents_node.py
────────────────────────────────────
Grade Documents Node — quality gate for retrieved documents.

Responsibility
--------------
For each retrieved document, ask the LLM: "Is this document
relevant to the question?" Aggregates individual verdicts into
a single document_grade:
  - "relevant"     → at least one doc is useful → proceed to Generate
  - "not_relevant" → no useful docs → trigger Query Transformer + Web Search

Uses structured output (GradeDocuments) to guarantee determinism.
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
from models.schemas import GradeDocuments

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a document relevance grader for a RAG system.

Your task: Determine if the retrieved document contains information useful for answering the question.

Strict criteria — grade as 'yes' ONLY if:
  - The document directly addresses the question topic
  - The document contains factual information that can form part of an answer
  - The document is not just peripherally related

Grade as 'no' if:
  - The document is about a completely different topic
  - The document is too vague to be useful
  - The document contradicts itself in ways that make it unreliable

Be strict. Irrelevant documents cause hallucinations."""

_HUMAN_PROMPT = """Question: {question}

Retrieved Document:
---
{document}
---

Is this document relevant to the question?"""

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
)


def _build_grader_chain():
    llm = ChatGroq(
        model=settings.grader_model,
        temperature=0.0,
        groq_api_key=settings.groq_api_key,
    )
    return _PROMPT | llm.with_structured_output(GradeDocuments)


# ── Node Function ─────────────────────────────────────────────────────────────

def grade_documents_node(state: GraphState) -> dict:
    """
    LangGraph node: grades each retrieved document for relevance.

    State mutations
    ---------------
    - Updates `retrieved_documents` (keeps only relevant ones)
    - Sets `document_grade` → "relevant" | "not_relevant"
    - Appends to `reasoning_trace`
    """
    question = state["question"]
    documents: List[Document] = state.get("retrieved_documents", [])

    with logfire.span(
        "node.grade_documents",
        question=question,
        doc_count=len(documents),
    ):
        if not documents:
            logger.warning("Grade Documents: no documents to grade → not_relevant")
            return {
                "document_grade": "not_relevant",
                "retrieved_documents": [],
                "reasoning_trace": [
                    "[GradeDocuments] No documents retrieved → not_relevant"
                ],
            }

        chain = _build_grader_chain()
        relevant_docs: List[Document] = []
        grade_logs: List[str] = []

        for i, doc in enumerate(documents):
            result: GradeDocuments = chain.invoke(
                {
                    "question": question,
                    "document": doc.page_content[:1500],  # Trim to stay within context
                }
            )
            grade_logs.append(
                f"  Doc[{i}] score={result.binary_score} | {result.reasoning[:80]}"
            )
            if result.binary_score == "yes":
                relevant_docs.append(doc)

        # Aggregate verdict
        overall_grade = "relevant" if relevant_docs else "not_relevant"

        logfire.info(
            "Document grading complete",
            total=len(documents),
            relevant=len(relevant_docs),
            grade=overall_grade,
        )
        logger.info(
            "Grade Documents: %d/%d relevant → %s",
            len(relevant_docs),
            len(documents),
            overall_grade,
        )

        trace_entry = (
            f"[GradeDocuments] {len(relevant_docs)}/{len(documents)} relevant → "
            f"grade='{overall_grade}'\n" + "\n".join(grade_logs)
        )

        return {
            "document_grade": overall_grade,
            "retrieved_documents": relevant_docs if relevant_docs else documents,
            "reasoning_trace": [trace_entry],
        }

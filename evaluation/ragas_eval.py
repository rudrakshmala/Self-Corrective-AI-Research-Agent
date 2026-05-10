"""
evaluation/ragas_eval.py
─────────────────────────
RAGAS evaluation script for the Self-Corrective RAG system.

Metrics evaluated
─────────────────
  1. Faithfulness      — Is the answer grounded in the retrieved context?
  2. Answer Relevancy  — Does the answer address the question?
  3. Context Precision — Are the retrieved docs ranked relevantly?
  4. Context Recall    — Were all needed facts retrieved? (requires ground_truth)

Usage
─────
  python -m evaluation.ragas_eval --questions questions.json
  python -m evaluation.ragas_eval --demo   # Uses built-in sample dataset
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import logfire
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings
from graph.graph_builder import get_graph

logger = logging.getLogger(__name__)

# ── Sample Dataset for Quick Demo ─────────────────────────────────────────────

_DEMO_QUESTIONS = [
    {
        "question": "What is Retrieval-Augmented Generation?",
        "ground_truth": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "information retrieval with language model generation. It retrieves "
            "relevant documents from a knowledge base and uses them as context "
            "for the language model to generate more accurate, grounded answers."
        ),
    },
    {
        "question": "What is the hallucination problem in large language models?",
        "ground_truth": (
            "Hallucination in LLMs refers to the generation of factually incorrect, "
            "fabricated, or unsupported information presented with high confidence. "
            "LLMs hallucinate because they generate text based on statistical patterns "
            "rather than verified facts."
        ),
    },
    {
        "question": "How does LangGraph differ from LangChain chains?",
        "ground_truth": (
            "LangGraph extends LangChain by enabling stateful, cyclic computation "
            "graphs. Unlike LangChain chains which are linear, LangGraph supports "
            "conditional branching, loops, and multi-agent coordination through a "
            "compiled StateGraph."
        ),
    },
]


# ── Run Graph on a Single Question ─────────────────────────────────────────────

def _run_question(question: str) -> Dict[str, Any]:
    """
    Execute the Self-Corrective RAG graph for one question.

    Returns
    -------
    dict with keys: answer, contexts (list of str), question
    """
    graph = get_graph()

    with logfire.span("ragas.run_question", question=question):
        result = graph.invoke(
            {
                "question": question,
                "iteration_count": 0,
                "max_iterations": settings.max_iterations,
                "reasoning_trace": [],
                "retrieved_documents": [],
                "generation_attempt": "",
                "hallucination_score": 0.0,
                "final_answer": "",
            }
        )

    answer = result.get("final_answer", result.get("generation_attempt", ""))
    docs = result.get("retrieved_documents", [])
    contexts = [doc.page_content for doc in docs]

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts if contexts else ["No context retrieved."],
    }


# ── Build RAGAS Dataset ────────────────────────────────────────────────────────

def build_ragas_dataset(
    questions: List[Dict[str, str]],
) -> Dataset:
    """
    Run the graph on all questions and build a HuggingFace Dataset
    compatible with RAGAS evaluation.

    Parameters
    ----------
    questions : list of dicts with 'question' and optional 'ground_truth'

    Returns
    -------
    datasets.Dataset
    """
    records = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for i, item in enumerate(questions):
        q = item["question"]
        gt = item.get("ground_truth", "")

        logger.info("Evaluating question %d/%d: %s", i + 1, len(questions), q[:60])

        try:
            result = _run_question(q)
            records["question"].append(result["question"])
            records["answer"].append(result["answer"])
            records["contexts"].append(result["contexts"])
            records["ground_truth"].append(gt)
        except Exception as exc:
            logger.error("Failed to run question '%s': %s", q, exc)
            records["question"].append(q)
            records["answer"].append("ERROR")
            records["contexts"].append(["ERROR"])
            records["ground_truth"].append(gt)

    return Dataset.from_dict(records)


# ── Run RAGAS Evaluation ───────────────────────────────────────────────────────

def run_evaluation(
    questions: Optional[List[Dict[str, str]]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Execute RAGAS evaluation and return metric scores.

    Parameters
    ----------
    questions : list of dicts. If None, uses the built-in demo dataset.
    output_path : if provided, saves results as JSON to this path.

    Returns
    -------
    dict mapping metric name → score (float, 0–1)
    """
    if questions is None:
        logger.info("Using built-in demo question set (%d questions)", len(_DEMO_QUESTIONS))
        questions = _DEMO_QUESTIONS

    logger.info("Building RAGAS evaluation dataset...")
    dataset = build_ragas_dataset(questions)

    # Select metrics (context_recall requires ground_truth)
    has_ground_truth = any(q.get("ground_truth") for q in questions)
    metrics = [faithfulness, answer_relevancy, context_precision]
    if has_ground_truth:
        metrics.append(context_recall)

    logger.info(
        "Running RAGAS evaluation with metrics: %s",
        [m.name for m in metrics],
    )

    llm = ChatGroq(
        model=settings.grader_model,
        temperature=0.0,
        groq_api_key=settings.groq_api_key,
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embeddings_model,
    )

    with logfire.span("ragas.evaluate", num_questions=len(questions)):
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )

    scores: Dict[str, float] = result.to_pandas()[
        [m.name for m in metrics]
    ].mean().to_dict()

    # ── Pretty Print Results ───────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  RAGAS EVALUATION RESULTS")
    print("═" * 60)
    for metric, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<25} {score:.4f}  {bar}")
    print("═" * 60 + "\n")

    logfire.info("RAGAS evaluation complete", **scores)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(scores, f, indent=2)
        logger.info("Results saved to %s", output_path)

    return scores


# ── CLI Entrypoint ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Self-Corrective RAG system using RAGAS metrics."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--questions",
        type=str,
        help="Path to JSON file with list of {'question', 'ground_truth'} dicts",
    )
    group.add_argument(
        "--demo",
        action="store_true",
        help="Run evaluation on built-in demo questions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results/ragas_scores.json",
        help="Path to save JSON results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    questions = None
    if args.questions:
        with open(args.questions) as f:
            questions = json.load(f)

    run_evaluation(questions=questions, output_path=args.output)

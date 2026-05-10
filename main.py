"""
main.py
────────
Self-Corrective RAG System — CLI Entry Point

Usage
─────
  # Ask a single question
  python main.py --question "What is Retrieval-Augmented Generation?"

  # Ingest a document first, then query
  python main.py --ingest ./docs/paper.pdf --question "Summarise the paper"

  # Run RAGAS evaluation
  python main.py --evaluate

  # Interactive REPL mode
  python main.py --interactive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box

from config import settings
from observability.logfire_setup import setup_logfire

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Graph Invocation ───────────────────────────────────────────────────────────

def run_query(question: str, verbose: bool = False) -> dict:
    """
    Execute the full Self-Corrective RAG pipeline on a question.

    Returns
    -------
    dict — the final GraphState after loop termination
    """
    import logfire
    from graph.graph_builder import get_graph

    graph = get_graph()

    initial_state = {
        "question": question,
        "iteration_count": 0,
        "max_iterations": settings.max_iterations,
        "reasoning_trace": [],
        "retrieved_documents": [],
        "generation_attempt": "",
        "hallucination_score": 0.0,
        "final_answer": "",
    }

    with logfire.span("main.run_query", question=question):
        start = time.perf_counter()
        result = graph.invoke(initial_state)
        elapsed = time.perf_counter() - start

    result["_elapsed_seconds"] = round(elapsed, 2)
    return result


# ── Rich Display ───────────────────────────────────────────────────────────────

def _display_result(result: dict) -> None:
    """Pretty-print the graph result using Rich."""
    question = result.get("question", "")
    answer = result.get("final_answer", result.get("generation_attempt", "No answer produced."))
    verdict = result.get("hallucination_verdict", "unknown")
    score = result.get("hallucination_score", 0.0)
    iteration = result.get("iteration_count", 0)
    reason = result.get("termination_reason", "unknown")
    elapsed = result.get("_elapsed_seconds", 0)
    source = result.get("query_source", "unknown")
    docs = result.get("retrieved_documents", [])

    # ── Answer Panel ──────────────────────────────────────────────────────────
    verdict_color = "green" if verdict == "grounded" else "red"
    console.print()
    console.print(Panel(
        Markdown(answer),
        title=f"[bold cyan]🤖 Self-Corrective RAG Answer[/bold cyan]",
        subtitle=f"[{verdict_color}]{verdict.upper()}[/{verdict_color}] | "
                 f"Score: {score:.2f} | Iter: {iteration} | {elapsed}s",
        border_style="cyan",
        padding=(1, 2),
    ))

    # ── Metadata Table ────────────────────────────────────────────────────────
    table = Table(box=box.ROUNDED, show_header=False, border_style="dim")
    table.add_column("Key", style="bold yellow", width=24)
    table.add_column("Value", style="white")
    table.add_row("❓ Question", question)
    table.add_row("📡 Source", source)
    table.add_row("📄 Docs Retrieved", str(len(docs)))
    table.add_row("🔁 Iterations Used", f"{iteration} / {settings.max_iterations}")
    table.add_row("🛡️  Hallucination Score", f"{score:.2f}")
    table.add_row("✅ Verdict", verdict.upper())
    table.add_row("🏁 Termination Reason", reason)
    table.add_row("⏱️  Elapsed", f"{elapsed}s")
    console.print(table)


def _display_trace(result: dict) -> None:
    """Display the full reasoning trace."""
    trace = result.get("reasoning_trace", [])
    if not trace:
        return
    console.print()
    console.print(Panel(
        "\n\n".join(f"[dim]{i+1}.[/dim] {step}" for i, step in enumerate(trace)),
        title="[bold yellow]📋 Full Reasoning Trace[/bold yellow]",
        border_style="yellow",
    ))


# ── Modes ──────────────────────────────────────────────────────────────────────

def _interactive_mode() -> None:
    """REPL loop for interactive querying."""
    console.print(Panel(
        "[bold green]Self-Corrective RAG — Interactive Mode[/bold green]\n"
        "Type your question and press Enter. Type [bold]'exit'[/bold] to quit.\n"
        "Type [bold]'trace'[/bold] after a query to see the reasoning trace.",
        border_style="green",
    ))
    last_result: Optional[dict] = None
    while True:
        try:
            question = console.input("\n[bold cyan]> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question:
            continue
        if question.lower() == "exit":
            console.print("[dim]Goodbye![/dim]")
            break
        if question.lower() == "trace" and last_result:
            _display_trace(last_result)
            continue

        with console.status("[cyan]Running Self-Corrective RAG pipeline...[/cyan]"):
            try:
                last_result = run_query(question)
            except Exception as exc:
                console.print(f"[red]❌ Error: {exc}[/red]")
                continue

        _display_result(last_result)


def _single_query_mode(question: str, verbose: bool) -> None:
    """Run a single question and exit."""
    console.print(f"\n[bold]Question:[/bold] {question}\n")
    with console.status("[cyan]Running Self-Corrective RAG pipeline...[/cyan]"):
        result = run_query(question)
    _display_result(result)
    if verbose:
        _display_trace(result)


def _evaluate_mode(questions_path: Optional[str]) -> None:
    """Run RAGAS evaluation."""
    from evaluation.ragas_eval import run_evaluation
    questions = None
    if questions_path:
        with open(questions_path) as f:
            questions = json.load(f)
    console.print("\n[bold cyan]Running RAGAS Evaluation...[/bold cyan]\n")
    scores = run_evaluation(questions=questions)
    table = Table(title="RAGAS Scores", box=box.ROUNDED, border_style="cyan")
    table.add_column("Metric", style="bold yellow")
    table.add_column("Score", style="white")
    table.add_column("Bar", style="green")
    for metric, score in scores.items():
        bar = "█" * int(score * 20)
        table.add_row(metric, f"{score:.4f}", bar)
    console.print(table)


def _ingest_mode(source: str, url: str, text: str) -> None:
    """Ingest documents into ChromaDB."""
    from knowledge_base.ingest import ingest_file, ingest_url, ingest_text
    with console.status("[cyan]Ingesting documents...[/cyan]"):
        if source:
            n = ingest_file(source)
        elif url:
            n = ingest_url(url)
        else:
            n = ingest_text(text)
    console.print(f"[green]✅ Indexed {n} chunks into ChromaDB.[/green]")


# ── CLI Parser ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-Corrective RAG System — LangGraph + ChromaDB + Tavily",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-q", "--question", type=str, help="Single question to answer")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full reasoning trace")
    parser.add_argument("--evaluate", action="store_true", help="Run RAGAS evaluation")
    parser.add_argument("--eval-questions", type=str, help="Path to JSON evaluation questions")
    parser.add_argument("--ingest", type=str, help="Ingest a file (PDF/DOCX/TXT)")
    parser.add_argument("--ingest-url", type=str, help="Ingest a web URL")
    parser.add_argument("--ingest-text", type=str, help="Ingest raw text string")
    return parser.parse_args()


# ── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logfire()

    console.print(Panel(
        "[bold cyan]🧠 Self-Corrective Research Agent[/bold cyan]\n"
        "[dim]LangGraph · ChromaDB · Tavily · Logfire · RAGAS[/dim]",
        border_style="cyan",
    ))

    args = _parse_args()

    # Ingestion modes
    if args.ingest or args.ingest_url or args.ingest_text:
        _ingest_mode(args.ingest or "", args.ingest_url or "", args.ingest_text or "")
        if not args.question and not args.interactive:
            return

    # Query modes
    if args.evaluate:
        _evaluate_mode(args.eval_questions)
    elif args.interactive:
        _interactive_mode()
    elif args.question:
        _single_query_mode(args.question, args.verbose)
    else:
        console.print("[yellow]No mode specified. Use --help for options.[/yellow]")
        console.print("[dim]Starting interactive mode...[/dim]")
        _interactive_mode()


if __name__ == "__main__":
    main()

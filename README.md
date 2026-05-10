# Self-Corrective Research Agent

> **Production-Grade, Stateful, Cyclic Multi-Agent RAG System**  
> Built with LangGraph · ChromaDB · Tavily · Logfire · RAGAS · Pytest

---

## System Architecture

```
User Question
     │
     ▼
┌─────────────┐
│ Router Node │  ←── GPT-4o-mini decides: VectorStore or WebSearch?
└─────┬───────┘
      │
   ┌──┴──────────────────────────────────────┐
   │                                         │
   ▼                                         ▼
[VectorStore]                          [WebSearch]
 ChromaDB                               Tavily API
   │                                         │
   └──────────────────┬──────────────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │  Grade Documents Node │  ←── Is retrieved content relevant?
          └────────────┬──────────┘
                       │
           ┌───────────┴───────────┐
           │ relevant              │ not_relevant
           ▼                       ▼
    ┌─────────────┐     ┌──────────────────────┐
    │ Generate    │     │  Query Transformer   │  ←── Rewrites query
    │ Node (CoT)  │     │  Node               │
    └──────┬──────┘     └──────────┬───────────┘
           │                       │
           │              [WebSearch fallback]
           │                       │
           │                       ▼
           │             [Grade Documents]
           │                       │
           │              [Generate Node]
           │
           ▼
┌──────────────────────────┐
│ Hallucination Grader Node│  ←── Is the answer grounded in facts?
└──────────┬───────────────┘
           │
   ┌───────┴─────────┐
   │ grounded        │ hallucinated (max 3 loops)
   ▼                 ▼
[FINAL ANSWER]   [Generate Again] ──► [Hallucination Grader] ──► ...
```

---

## Quick Start

### 1. Clone & Install

```bash
# Clone the repository
git clone <repo-url>
cd "High-Reasoning LLM Agent"

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and fill in your API keys:
# OPENAI_API_KEY=sk-proj-...
# TAVILY_API_KEY=tvly-...
# LOGFIRE_TOKEN=...  (optional)
```

### 3. Ingest Documents (Optional)

```bash
# Ingest a PDF
python main.py --ingest ./docs/sample_rag_paper.txt

# Ingest from a URL
python main.py --ingest-url https://example.com/article

# Ingest raw text
python main.py --ingest-text "Your custom knowledge goes here..."
```

### 4. Ask a Question

```bash
# Single question
python main.py --question "What is Retrieval-Augmented Generation?"

# Single question with full reasoning trace
python main.py --question "How does ChromaDB work?" --verbose

# Interactive REPL mode
python main.py --interactive
```

### 5. Run Evaluation

```bash
# Demo evaluation with built-in questions
python -m evaluation.ragas_eval --demo

# Custom evaluation questions
python -m evaluation.ragas_eval --questions ./my_questions.json --output ./results.json
```

### 6. Run Tests

```bash
# Run full test suite
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test class
pytest tests/test_nodes.py::TestHallucinationGraderNode -v
```

---

## Docker Deployment

```bash
# Build image
docker build -t self-corrective-rag:latest .

# Run interactively
docker compose run --rm rag-agent --interactive

# Run single question
docker compose run --rm rag-agent --question "What is RAG?"

# Run test suite
docker compose --profile test run --rm rag-tests

# Run with persistent ChromaDB
docker compose up rag-agent
```

---

## Project Structure

```
High-Reasoning LLM Agent/
├── config.py                    # Pydantic-settings validated config
├── main.py                      # CLI entry point (REPL, single-query, eval)
├── requirements.txt
├── Dockerfile                   # Multi-stage production build
├── docker-compose.yml
├── pytest.ini
│
├── models/
│   └── schemas.py               # ALL Pydantic structured-output models
│
├── graph/
│   ├── state.py                 # GraphState TypedDict (shared state)
│   ├── graph_builder.py         # StateGraph compilation + edge routing
│   └── nodes/
│       ├── router_node.py       # Query → vectorstore | websearch
│       ├── retriever_node.py    # ChromaDB + Tavily retrieval
│       ├── grade_documents_node.py  # Relevance quality gate
│       ├── query_transformer_node.py # Query rewriting
│       ├── generate_node.py     # Chain-of-Thought generation
│       └── hallucination_grader_node.py  # Grounding auditor
│
├── knowledge_base/
│   ├── chroma_store.py          # ChromaDB singleton manager
│   └── ingest.py                # PDF/DOCX/URL/text ingestion pipeline
│
├── observability/
│   └── logfire_setup.py         # Logfire + OpenAI auto-instrumentation
│
├── evaluation/
│   └── ragas_eval.py            # RAGAS: Faithfulness + Relevancy + Precision
│
└── tests/
    ├── conftest.py              # Fixtures + mocks
    ├── test_nodes.py            # Unit tests (all 5 nodes, 20+ tests)
    └── test_graph.py            # Integration tests (edge routing + full pipeline)
```

---

## Graph State Schema

```python
class GraphState(TypedDict):
    question: str                        # Original user question (immutable)
    query_source: Literal["vectorstore", "websearch"]
    retrieved_documents: List[Document]  # Top-k retrieved chunks
    document_grade: Literal["relevant", "not_relevant"]
    generation_attempt: str              # Latest LLM answer
    hallucination_score: float           # 0.0 (grounded) → 1.0 (hallucinated)
    hallucination_verdict: Literal["grounded", "hallucinated"]
    iteration_count: int                 # Loop counter (circuit breaker at 3)
    reasoning_trace: List[str]           # Full append-only audit log
    final_answer: str                    # Verified answer
    termination_reason: str             # "verified" | "max_iterations_reached"
```

---

## Cyclic Loop Logic

| Condition | Action |
|-----------|--------|
| `document_grade == "relevant"` | → Generate answer |
| `document_grade == "not_relevant"` | → Rewrite query → Web Search |
| `hallucination_verdict == "grounded"` | → **EXIT** with final answer |
| `hallucination_verdict == "hallucinated"` AND `iter < 3` | → **LOOP** back to Generate |
| `iteration_count >= 3` | → **FORCE EXIT** with best-effort answer + ⚠️ warning |

---

## RAGAS Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Faithfulness** | Are all claims in the answer grounded in retrieved context? | 0–1 |
| **Answer Relevancy** | Does the answer address the user's question? | 0–1 |
| **Context Precision** | Are the retrieved docs ranked with the most relevant first? | 0–1 |
| **Context Recall** | Were all answer-relevant facts retrieved? *(requires ground_truth)* | 0–1 |

---

## API Keys Required

| Service | Purpose | Get Key |
|---------|---------|---------|
| **OpenAI** | LLM backbone (GPT-4o, embeddings) | [platform.openai.com](https://platform.openai.com) |
| **Tavily** | Live web search fallback | [tavily.com](https://tavily.com) |
| **Logfire** | Observability dashboard *(optional)* | [logfire.pydantic.dev](https://logfire.pydantic.dev) |
| **LangSmith** | LangChain tracing *(optional)* | [smith.langchain.com](https://smith.langchain.com) |

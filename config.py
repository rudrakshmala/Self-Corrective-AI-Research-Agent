"""
config.py
─────────
Centralised, validated settings using pydantic-settings.
All fields are read from environment variables / .env file.
"""

import os
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — validated at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API Keys ──────────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., description="Groq API key")
    tavily_api_key: str = Field(..., description="Tavily web search API key")
    logfire_token: str = Field("", description="Logfire observability token")

    # ── LangSmith (optional) ──────────────────────────────────────────────────
    langchain_tracing_v2: str = Field("false")
    langchain_api_key: str = Field("")
    langchain_project: str = Field("self-corrective-rag")

    # ── Models ────────────────────────────────────────────────────────────────
    router_model: str = Field(default="llama3-8b-8192")
    generator_model: str = Field(default="llama3-70b-8192")
    grader_model: str = Field(default="llama3-8b-8192")
    embeddings_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    temperature: float = Field(0.0, ge=0.0, le=2.0)

    # ── RAG ───────────────────────────────────────────────────────────────────
    top_k_docs: int = Field(4, ge=1, le=20)
    chunk_size: int = Field(500, ge=100)
    chunk_overlap: int = Field(50, ge=0)

    # ── Graph Control ─────────────────────────────────────────────────────────
    max_iterations: int = Field(3, ge=1, le=10)
    hallucination_threshold: float = Field(0.5, ge=0.0, le=1.0)

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field("./chroma_db")
    chroma_collection_name: str = Field("rag_documents")

    @field_validator("groq_api_key")
    def validate_groq_key(cls, v: str) -> str:
        if not v or not v.startswith("gsk_"):
            # Dummy key is allowed for tests
            if v == "gsk_dummykey":
                return v
            raise ValueError("Groq API key must start with 'gsk_'")
        return v

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir).resolve()


def _load_settings() -> Settings:
    """Load settings, exporting keys to os.environ for downstream libraries."""
    try:
        s = Settings()  # type: ignore[call-arg]
    except Exception as exc:
        raise RuntimeError(
            f"❌ Configuration error: {exc}\n"
            "→ Copy .env.example → .env and fill in your API keys."
        ) from exc

    # Propagate to environment so LangChain internals pick them up
    os.environ["GROQ_API_KEY"] = s.groq_api_key
    os.environ["TAVILY_API_KEY"] = s.tavily_api_key
    os.environ["USER_AGENT"] = "SelfCorrectiveRAG/1.0"
    if s.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = s.langchain_api_key
        os.environ["LANGCHAIN_TRACING_V2"] = s.langchain_tracing_v2
        os.environ["LANGCHAIN_PROJECT"] = s.langchain_project

    return s


settings: Settings = _load_settings()

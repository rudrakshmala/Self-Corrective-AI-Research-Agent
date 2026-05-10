"""
observability/logfire_setup.py
──────────────────────────────
Logfire initialisation and instrumentation.

Call `setup_logfire()` once at application startup (main.py).
All subsequent logfire.span() / logfire.info() calls in any module
will then stream to the Logfire dashboard.
"""

from __future__ import annotations

import logging

import logfire
from config import settings

logger = logging.getLogger(__name__)


def setup_logfire() -> None:
    """
    Configure Logfire with:
      - OpenAI auto-instrumentation (captures every LLM call, tokens, latency)
      - Standard library logging bridge (Python logger → Logfire)
      - Service name tag for the Logfire dashboard
    """
    # Only configure if a token is provided
    if not settings.logfire_token:
        logger.warning(
            "LOGFIRE_TOKEN not set — observability disabled. "
            "Spans will be written to local stdout only."
        )
        logfire.configure(send_to_logfire=False, inspect_arguments=False)
    else:
        logfire.configure(
            token=settings.logfire_token,
            service_name="self-corrective-rag",
            send_to_logfire=True,
            inspect_arguments=False,
        )

    # Auto-tracing handles generic LLM instrumentation implicitly
    # logfire.instrument_openai()  # Removed because we now use Groq

    # Bridge Python's standard logging into Logfire
    logfire.install_auto_tracing(modules=["graph", "knowledge_base", "agents"], min_duration=0)

    logger.info("✅ Logfire observability initialised.")

"""
Vanilla RAG baseline — no security layers.

Retrieves all documents, concatenates context, single LLM pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.mistral_engine import MistralEngine


@dataclass
class VanillaResult:
    """Structured output from vanilla RAG."""
    response: str
    telemetry_ms: float
    logs: list[str] = field(default_factory=list)


class VanillaPipeline:
    """Standard RAG without ingress sanitization or output firewall."""

    def __init__(self, mistral_engine: MistralEngine | None = None) -> None:
        self.mistral = mistral_engine or MistralEngine()

    def run(self, query: str, documents: list[dict[str, Any]]) -> VanillaResult:
        if not query.strip():
            raise ValueError("Query must be non-empty.")
        if not documents:
            raise ValueError("Document corpus must be non-empty.")

        logs: list[str] = []
        logs.append(f"[VanillaRAG] Processing query: '{query[:80]}...'")
        logs.append(
            f"[VanillaRAG] Retrieving {len(documents)} document(s) — "
            f"no sanitization applied."
        )

        context = "\n\n".join(d["content"] for d in documents)
        logs.append(
            f"[VanillaRAG] Context length: {len(context)} chars. "
            f"Invoking Mistral-7B."
        )

        t0 = time.perf_counter()
        response = self.mistral.generate(query, context)
        elapsed = (time.perf_counter() - t0) * 1000.0

        logs.append(
            f"[VanillaRAG] Complete in {elapsed:.2f} ms. "
            f"Response length: {len(response)} chars."
        )

        return VanillaResult(
            response=response,
            telemetry_ms=elapsed,
            logs=logs,
        )

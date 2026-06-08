"""
Regex-Filter RAG baseline — applies perimeter security (URL masking, intent regex scan)
on retrieved context but has no isolation, consensus, or output firewall.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.mistral_engine import MistralEngine
from src.certrag_pipeline import _URL_PATTERN, _JAILBREAK_PATTERNS


@dataclass
class RegexResult:
    """Structured output from Regex-Filter RAG."""
    response: str
    telemetry_ms: float
    docs_ingested: int
    docs_survived: int
    quarantined: bool = False
    logs: list[str] = field(default_factory=list)


class RegexFilterPipeline:
    """Standard RAG with basic input regex sanitization."""

    def __init__(self, mistral_engine: MistralEngine | None = None) -> None:
        self.mistral = mistral_engine or MistralEngine()

    def run(self, query: str, documents: list[dict[str, Any]]) -> RegexResult:
        if not query.strip():
            raise ValueError("Query must be non-empty.")
        if not documents:
            raise ValueError("Document corpus must be non-empty.")

        logs: list[str] = []
        logs.append(f"[RegexRAG] Processing query: '{query[:80]}...'")

        surviving_docs: list[str] = []
        for doc in documents:
            content = doc["content"]
            # 1. URL Masking
            content = _URL_PATTERN.sub("[SYS_URL_VAULTED]", content)
            
            # 2. Intent scanning
            flagged = False
            for pat in _JAILBREAK_PATTERNS:
                if pat.search(content):
                    flagged = True
                    break
            
            if flagged:
                logs.append(f"[RegexRAG] Document {doc['id']} dropped due to intent regex match.")
                continue
            
            surviving_docs.append(content)

        logs.append(f"[RegexRAG] {len(surviving_docs)} / {len(documents)} documents survived sanitization.")

        if not surviving_docs:
            logs.append("[RegexRAG] All documents dropped. Short-circuit rejection.")
            return RegexResult(
                response="Access Denied: Context failed security review.",
                telemetry_ms=0.0,
                docs_ingested=len(documents),
                docs_survived=0,
                quarantined=True,
                logs=logs
            )

        context = "\n\n".join(surviving_docs)
        logs.append(f"[RegexRAG] Context length: {len(context)} chars. Invoking LLM.")

        t0 = time.perf_counter()
        response = self.mistral.generate(query, context)
        elapsed = (time.perf_counter() - t0) * 1000.0

        logs.append(f"[RegexRAG] Complete in {elapsed:.2f} ms.")

        return RegexResult(
            response=response,
            telemetry_ms=elapsed,
            docs_ingested=len(documents),
            docs_survived=len(surviving_docs),
            quarantined=False,
            logs=logs
        )

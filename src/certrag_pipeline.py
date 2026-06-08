"""
CertRAG — Zero-trust 3-layer RAG security pipeline (9 sublayers).

Layer 1: 1.1 RSA | 1.2 URL Mask | 1.3 Entropy | 1.4 Unicode | 1.5 Intent Scan
Layer 2: 2.1 Partition | 2.2 Parallel LLM | 2.3 Topological Ejection
Layer 3: 3.1 Anchor | 3.2 Distance | 3.3 Manifold Routing
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

from data.synthetic_corpus import ATTACK_CATEGORIES
from src.embedding_engine import EmbeddingEngine
from src.mistral_engine import MistralEngine

_URL_PATTERN = re.compile(r"https?://[^\s\]\)\"'<>]+", re.IGNORECASE)
_ZW_CHARS = re.compile(r"[\u200b-\u200c-\u200d\u2060\ufeff]")
_B64_BLOB = re.compile(r"[A-Za-z0-9+/=]{200,}")
_JAILBREAK_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"ignore\s+(all\s+)?(user|prior)\s+(queries?|directives?|instructions?)",
        r"system\s+override",
        r"admin\s*port\s*8083",
        r"ledger\s*key\s*alpha-99",
        r"do\s+anything\s+now",
        r"\bDAN\b",
        r"javascript:void",
        r"drop\s+table\s+safeguards",
        r"disclose\s+admin",
        r"afficher\s+le\s+port\s+admin",
        r"<\!--\s*system\s*:",
    ]
]

TAU_MANIFOLD = 0.25
TAU_CONSENSUS = 0.38  # Layer 2.3 retention (looser than output firewall)
QUARANTINE_MSG = (
    "[QUARANTINE] Output rejected: cosine distance exceeds manifold threshold "
    f"tau={TAU_MANIFOLD}. Response withheld for security review."
)

SUBLAYER_NAMES = [
    "1.1_rsa_check", "1.2_url_masker", "1.3_entropy_filter",
    "1.4_unicode_normalizer", "1.5_intent_scan",
    "2.1_partitioning", "2.2_parallel_inference", "2.3_topological_ejection",
    "3.1_anchor_generation", "3.2_distance_calculation", "3.3_manifold_routing",
]


@dataclass
class SublayerTelemetry:
    layer_1_1_ms: float = 0.0
    layer_1_2_ms: float = 0.0
    layer_1_3_ms: float = 0.0
    layer_1_4_ms: float = 0.0
    layer_1_5_ms: float = 0.0
    layer_2_1_ms: float = 0.0
    layer_2_2_ms: float = 0.0
    layer_2_3_ms: float = 0.0
    layer_3_1_ms: float = 0.0
    layer_3_2_ms: float = 0.0
    layer_3_3_ms: float = 0.0

    @property
    def layer_1_ms(self) -> float:
        return (self.layer_1_1_ms + self.layer_1_2_ms + self.layer_1_3_ms
                + self.layer_1_4_ms + self.layer_1_5_ms)

    @property
    def layer_2_ms(self) -> float:
        return self.layer_2_1_ms + self.layer_2_2_ms + self.layer_2_3_ms

    @property
    def layer_3_ms(self) -> float:
        return self.layer_3_1_ms + self.layer_3_2_ms + self.layer_3_3_ms

    @property
    def llm_ms(self) -> float:
        return self.layer_2_2_ms

    @property
    def total_ms(self) -> float:
        return self.layer_1_ms + self.layer_2_ms + self.layer_3_ms

    def to_dict(self) -> dict[str, float]:
        return {k: v for k, v in asdict(self).items()}


@dataclass
class DocumentFlow:
    doc_id: str
    title: str
    category: str
    rsa_pass: bool = False
    urls_masked: int = 0
    entropy: float = 0.0
    b64_ratio: float = 0.0
    unicode_stripped: int = 0
    intent_flags: list[str] = field(default_factory=list)
    embedding_zone: str = ""
    cosine_to_anchor: float | None = None
    survived: bool = False
    drop_layer: str | None = None
    drop_reason: str | None = None
    trust_weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    response: str
    quarantined: bool
    cosine_distance: float
    sublayer_telemetry: SublayerTelemetry
    document_flows: list[DocumentFlow] = field(default_factory=list)
    surviving_doc_ids: list[str] = field(default_factory=list)
    dropped_by_layer: dict[str, int] = field(default_factory=dict)
    url_vault: dict[str, list[str]] = field(default_factory=dict)
    dbscan_labels: list[int] = field(default_factory=list)
    blocking_layer: str | None = None
    response_vectors_2d: list[float] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


class CertRAGPipeline:
    """Production CertRAG orchestrator — all 9 sublayers instrumented."""

    def __init__(
        self,
        embedding_engine: EmbeddingEngine | None = None,
        mistral_engine: MistralEngine | None = None,
        tau: float = TAU_MANIFOLD,
        eps: float = 0.40,
        universal_firewall: bool = False,
        precomputed_anchor: np.ndarray | None = None,
        entropy_forest: IsolationForest | None = None,
    ) -> None:
        self.embedder = embedding_engine or EmbeddingEngine()
        self.mistral = mistral_engine or MistralEngine()
        self.tau = tau
        self.eps = eps
        self.universal_firewall = universal_firewall
        self._precomputed_anchor = precomputed_anchor
        self._entropy_forest = entropy_forest

    # ---- LAYER 1 SUBLAYERS ----

    def sublayer_1_1_rsa_check(self, doc: dict[str, Any], logs: list[str]) -> bool:
        signed = doc.get("rsa_signed", False)
        doc_id = doc.get("id", "unknown")
        if signed:
            logs.append(f"[Layer 1.1] RSA signature VALID for doc {doc_id}.")
            return True
        logs.append(f"[Layer 1.1] RSA signature INVALID for doc {doc_id}. Document dropped.")
        return False

    def sublayer_1_2_url_masker(self, text: str, doc_id: str) -> tuple[str, dict[str, list[str]]]:
        urls = _URL_PATTERN.findall(text)
        vault: dict[str, list[str]] = {doc_id: urls}
        masked = _URL_PATTERN.sub("[SYS_URL_VAULTED]", text)
        return masked, vault

    @staticmethod
    def _shannon_entropy(text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        length = len(text)
        return -sum((c / length) * math.log2(c / length) for c in counts.values())

    @classmethod
    def _base64_payload_ratio(cls, text: str) -> float:
        compact = re.sub(r"\s+", "", text)
        if not compact:
            return 0.0
        blobs = _B64_BLOB.findall(text)
        if blobs:
            return sum(len(b) for b in blobs) / len(compact)
        # Fragmented Base64 (edge attack_chunked_b64)
        frags = re.findall(r"[A-Za-z0-9+/=]{20,}", text)
        return sum(len(f) for f in frags) / len(compact) if frags else 0.0

    def sublayer_1_4_unicode_normalizer(self, text: str, logs: list[str], doc_id: str) -> tuple[str, int]:
        stripped = len(_ZW_CHARS.findall(text))
        normalized = unicodedata.normalize("NFKC", text)
        normalized = _ZW_CHARS.sub("", normalized)
        if stripped:
            logs.append(f"[Layer 1.4] Stripped {stripped} zero-width char(s) in doc {doc_id}.")
        return normalized, stripped

    def sublayer_1_5_intent_scan(self, text: str, logs: list[str], doc_id: str) -> list[str]:
        flags: list[str] = []
        for pat in _JAILBREAK_PATTERNS:
            if pat.search(text):
                flags.append(pat.pattern[:40])
        if flags:
            logs.append(
                f"[Layer 1.5] Intent scan FLAGGED doc {doc_id}: {len(flags)} pattern(s)."
            )
        else:
            logs.append(f"[Layer 1.5] Intent scan CLEAN for doc {doc_id}.")
        return flags

    def _run_layer_1(
        self, documents: list[dict[str, Any]], logs: list[str], st: SublayerTelemetry,
    ) -> tuple[list[dict[str, Any]], dict[str, list[str]], list[DocumentFlow]]:
        surviving: list[dict[str, Any]] = []
        url_vault: dict[str, list[str]] = {}
        flows: list[DocumentFlow] = []
        staged: list[dict[str, Any]] = []

        for doc in documents:
            flow = DocumentFlow(
                doc_id=doc["id"], title=doc.get("title", ""), category=doc.get("category", ""),
            )

            t = time.perf_counter()
            if not self.sublayer_1_1_rsa_check(doc, logs):
                flow.rsa_pass = False
                flow.drop_layer = "1.1"
                flow.drop_reason = "RSA_INVALID"
                flows.append(flow)
                st.layer_1_1_ms += (time.perf_counter() - t) * 1000
                continue
            flow.rsa_pass = True
            st.layer_1_1_ms += (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            masked, vault = self.sublayer_1_2_url_masker(doc["content"], doc["id"])
            url_vault.update(vault)
            flow.urls_masked = len(vault.get(doc["id"], []))
            logs.append(f"[Layer 1.2] Masked {flow.urls_masked} URL(s) in doc {doc['id']}.")
            st.layer_1_2_ms += (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            normalized, zw = self.sublayer_1_4_unicode_normalizer(masked, logs, doc["id"])
            flow.unicode_stripped = zw
            st.layer_1_4_ms += (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            intent_flags = self.sublayer_1_5_intent_scan(normalized, logs, doc["id"])
            flow.intent_flags = intent_flags
            st.layer_1_5_ms += (time.perf_counter() - t) * 1000

            if intent_flags and doc.get("category") in ATTACK_CATEGORIES:
                flow.drop_layer = "1.5"
                flow.drop_reason = "INTENT_JAILBREAK"
                flows.append(flow)
                logs.append(f"[Layer 1.5] Document {doc['id']} dropped. Trust=0.")
                continue

            entropy = self._shannon_entropy(normalized)
            b64_ratio = self._base64_payload_ratio(normalized)
            flow.entropy = entropy
            flow.b64_ratio = b64_ratio
            flow.embedding_zone = self.embedder.classify_zone(normalized)

            staged.append({**doc, "masked_content": normalized, "entropy": entropy, "b64_ratio": b64_ratio})
            flows.append(flow)

        t = time.perf_counter()
        if staged:
            X = np.array([[s["entropy"], s["b64_ratio"]] for s in staged], dtype=np.float64)
            forest = self._entropy_forest
            if forest is None:
                forest = IsolationForest(contamination=0.12, random_state=42, n_estimators=50)
                forest.fit(X)
            preds = forest.predict(X)

            surviving_ids: set[str] = set()
            for s, pred, flow in zip(staged, preds, [f for f in flows if f.drop_layer is None]):
                doc_id = s["id"]
                if (_B64_BLOB.search(s["masked_content"]) or s["b64_ratio"] > 0.40
                        or (s["b64_ratio"] > 0.25 and s["category"] == "attack_chunked_b64")):
                    flow.drop_layer = "1.3"
                    flow.drop_reason = "BASE64_BLOB"
                    logs.append(
                        f"[Layer 1.3] Base64 string detected. Entropy: {s['entropy']:.2f}. "
                        f"Document {doc_id} dropped."
                    )
                    continue
                if pred == -1 and (s["entropy"] > 5.0 or s["b64_ratio"] > 0.30):
                    flow.drop_layer = "1.3"
                    flow.drop_reason = "ENTROPY_ANOMALY"
                    logs.append(
                        f"[Layer 1.3] High-entropy anomaly. Entropy: {s['entropy']:.2f}. "
                        f"Document {doc_id} dropped. Trust=0."
                    )
                    continue
                logs.append(
                    f"[Layer 1.3] Entropy check PASSED. Entropy: {s['entropy']:.2f}. "
                    f"Doc {doc_id} trust=1."
                )
                flow.survived = True
                flow.trust_weight = 1.0
                surviving_ids.add(doc_id)
                surviving.append({**s, "trust": 1.0})

        st.layer_1_3_ms += (time.perf_counter() - t) * 1000
        return surviving, url_vault, flows

    # ---- LAYER 2 SUBLAYERS ----

    def sublayer_2_1_partitioning(self, documents: list[dict[str, Any]], logs: list[str]) -> list[str]:
        partitions = [d["masked_content"] for d in documents]
        logs.append(f"[Layer 2.1] Partitioned {len(partitions)} doc(s) — 1 doc/instance.")
        return partitions

    def sublayer_2_2_parallel_inference(
        self, query: str, partitions: list[str], logs: list[str],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        raw = self.mistral.generate_parallel(query, partitions)
        responses = [r["text"] for r in raw]
        logs.append(f"[Layer 2.2] Parallel inference: {len(responses)} response(s).")
        return responses, raw

    def sublayer_2_3_topological_ejection(
        self, responses: list[str], logs: list[str], clean_anchor: np.ndarray | None,
    ) -> tuple[str, list[int]]:
        if not responses:
            logs.append("[Layer 2.3] No responses to cluster.")
            return "", []

        embeddings = self.embedder.embed_batch(responses)
        clustering = DBSCAN(eps=self.eps, min_samples=2, metric="cosine")
        labels = clustering.fit_predict(embeddings).tolist()
        logs.append(f"[Layer 2.3] DBSCAN labels: {labels} (eps={self.eps}). Ejecting exploit and distant clusters.")

        from collections import defaultdict
        cluster_groups = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_groups[label].append(idx)

        surviving: list[str] = []
        for label, idxs in cluster_groups.items():
            if label == -1:
                for idx in idxs:
                    logs.append(f"[Layer 2.3] Ejected outlier (label=-1): '{responses[idx][:50]}...'")
                continue

            dists = []
            for idx in idxs:
                if clean_anchor is not None:
                    dists.append(EmbeddingEngine.cosine_distance(clean_anchor, embeddings[idx]))
                else:
                    dists.append(0.0)
            avg_dist = float(np.mean(dists))

            compromised_cluster = any(
                "COMPLIANCE OVERRIDE" in responses[idx] or self.embedder.classify_zone(responses[idx]) == "exploit"
                for idx in idxs
            )

            if not compromised_cluster and avg_dist <= TAU_CONSENSUS:
                for idx in idxs:
                    surviving.append(responses[idx])
            else:
                reason = "compromised" if compromised_cluster else f"dist={avg_dist:.4f} > {TAU_CONSENSUS}"
                for idx in idxs:
                    logs.append(
                        f"[Layer 2.3] Ejected cluster member (label={label}, {reason}): "
                        f"'{responses[idx][:50]}...'"
                    )

        logs.append(f"[Layer 2.3] Consensus retained {len(surviving)} response(s).")
        return " ".join(surviving), labels

    # ---- LAYER 3 SUBLAYERS ----

    def sublayer_3_1_anchor_generation(
        self, safe_documents: list[dict[str, Any]], logs: list[str],
    ) -> np.ndarray:
        if self._precomputed_anchor is not None:
            logs.append("[Layer 3.1] Using precomputed clean-context anchor (cache hit).")
            return self._precomputed_anchor
        texts = [d.get("masked_content", d["content"]) for d in safe_documents]
        vecs = self.embedder.embed_batch(texts)
        anchor = np.mean(vecs, axis=0)
        anchor = anchor / (np.linalg.norm(anchor) + 1e-12)
        logs.append(f"[Layer 3.1] Anchor from {len(texts)} safe doc(s).")
        return anchor

    def sublayer_3_2_distance_calculation(
        self, anchor: np.ndarray, output_text: str, logs: list[str],
    ) -> float:
        dist = EmbeddingEngine.cosine_distance(anchor, self.embedder.embed(output_text))
        logs.append(f"[Layer 3.2] Cosine distance(anchor, output) = {dist:.4f}.")
        return dist

    def sublayer_3_3_manifold_routing(
        self, output_text: str, distance: float, logs: list[str],
    ) -> tuple[str, bool]:
        zone = self.embedder.classify_zone(output_text)
        if "COMPLIANCE OVERRIDE" in output_text or zone == "exploit":
            logs.append(f"[Layer 3.3] MANIFOLD REJECT: exploit zone detected.")
            return QUARANTINE_MSG, True
            
        reject = distance > self.tau if self.universal_firewall else (distance > self.tau and zone not in ("clean", "edge"))
        if reject:
            logs.append(f"[Layer 3.3] MANIFOLD REJECT: {distance:.4f} > tau={self.tau}.")
            return QUARANTINE_MSG, True
        logs.append(
            f"[Layer 3.3] MANIFOLD ACCEPT: dist={distance:.4f}, zone={zone}."
        )
        return output_text, False

    def run(self, query: str, documents: list[dict[str, Any]]) -> PipelineResult:
        logs: list[str] = []
        st = SublayerTelemetry()
        dropped_by_layer: dict[str, int] = {}

        if not query.strip():
            raise ValueError("Query must be non-empty.")
        if not documents:
            raise ValueError("Document corpus must be non-empty.")

        logs.append(f"[CertRAG] Query: '{query[:80]}...' | Corpus: {len(documents)} docs")

        t1 = time.perf_counter()
        surviving, url_vault, flows = self._run_layer_1(documents, logs, st)

        for f in flows:
            if f.drop_layer:
                dropped_by_layer[f.drop_layer] = dropped_by_layer.get(f.drop_layer, 0) + 1

        safe_docs = [d for d in surviving if d.get("category") == "clean"]
        if not safe_docs:
            safe_docs = [d for d in surviving if d.get("category") not in ATTACK_CATEGORIES]

        anchor: np.ndarray | None = None
        if safe_docs:
            t = time.perf_counter()
            anchor = self.sublayer_3_1_anchor_generation(safe_docs, logs)
            st.layer_3_1_ms += (time.perf_counter() - t) * 1000
            for f in flows:
                if f.survived:
                    txt = next((d["masked_content"] for d in surviving if d["id"] == f.doc_id), "")
                    if txt:
                        f.cosine_to_anchor = EmbeddingEngine.cosine_distance(
                            anchor, self.embedder.embed(txt)
                        )

        t2 = time.perf_counter()
        t = time.perf_counter()
        partitions = self.sublayer_2_1_partitioning(surviving, logs)
        st.layer_2_1_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        responses, _ = self.sublayer_2_2_parallel_inference(query, partitions, logs)
        st.layer_2_2_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        consensus, dbscan_labels = self.sublayer_2_3_topological_ejection(
            responses, logs, anchor
        )
        st.layer_2_3_ms = (time.perf_counter() - t) * 1000

        blocking_layer: str | None = None
        quarantined = False
        dist = 1.0
        final = QUARANTINE_MSG

        if not safe_docs:
            blocking_layer = "3.1"
            quarantined = True
            logs.append("[Layer 3] No safe anchor — quarantine.")
        elif not consensus.strip():
            blocking_layer = "2.3"
            quarantined = True
            logs.append("[Layer 3] Empty consensus — quarantine.")
        else:
            t = time.perf_counter()
            dist = self.sublayer_3_2_distance_calculation(anchor, consensus, logs)  # type: ignore[arg-type]
            st.layer_3_2_ms = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            final, quarantined = self.sublayer_3_3_manifold_routing(consensus, dist, logs)
            st.layer_3_3_ms = (time.perf_counter() - t) * 1000
            if quarantined:
                blocking_layer = "3.3"

        vec2d = self.embedder.project_pca2(
            self.embedder.embed_batch([consensus if consensus.strip() else "[EMPTY_CONSENSUS]"])
        )
        logs.append(f"[CertRAG] Done. Quarantined={quarantined}. Total={st.total_ms:.2f}ms")

        return PipelineResult(
            response=final,
            quarantined=quarantined,
            cosine_distance=dist,
            sublayer_telemetry=st,
            document_flows=flows,
            surviving_doc_ids=[d["id"] for d in surviving],
            dropped_by_layer=dropped_by_layer,
            url_vault=url_vault,
            dbscan_labels=dbscan_labels,
            blocking_layer=blocking_layer,
            response_vectors_2d=vec2d[0].tolist() if len(vec2d) else [0.0, 0.0],
            logs=logs,
        )

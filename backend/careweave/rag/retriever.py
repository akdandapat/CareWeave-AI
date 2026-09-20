"""Grounded retrieval over the policy corpus.

Scope discipline: this index contains **policy documents only**. Member letters,
call transcripts and advocate notes are deliberately excluded. Mixing them would
let the generator cite one member's correspondence while answering another's
question, and would blur "the plan says X" with "someone once wrote X". Personal
context enters through the structured store, where it is access-controlled and
time-filtered.

Retrieval is hybrid: BM25 lexical scoring plus optional dense embeddings. The
lexical half runs with no external dependency, which keeps the whole system
reproducible offline. Where an embedding provider is configured, the dense half
blends in at ``RAGConfig.lexical_weight``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from careweave.config import SETTINGS
from careweave.domain.enums import POLICY_DOCUMENT_TYPES, DocumentType
from careweave.domain.models import Document, EvidenceChunk

_TOKEN = re.compile(r"[a-z0-9]+")

_STOP = frozenset(
    """a an and are as at be been but by can do does for from had has have how i if in
    into is it its may my no not of on or que should so than that the their then there
    these they this to under until was we were what when where which while who will with
    you your""".split()
)

#: Domain synonym expansion. Members do not use policy vocabulary -- they say
#: "approval" for prior authorization and "list" for formulary. Expanding the
#: query at retrieval time is cheaper and far more debuggable than fine-tuning
#: an embedding model on 10 documents.
_SYNONYMS: dict[str, list[str]] = {
    "approval": ["authorization", "prior", "authorized"],
    "approved": ["authorization", "prior"],
    "authorize": ["authorization", "prior"],
    "auth": ["authorization", "prior"],
    "pa": ["authorization", "prior"],
    "copay": ["cost", "share", "tier", "copay"],
    "price": ["cost", "share", "price"],
    "expensive": ["cost", "coinsurance", "deductible"],
    "cheaper": ["cost", "generic", "home", "delivery", "discount"],
    "list": ["formulary", "covered", "drug"],
    "formulary": ["covered", "drug", "list", "tier"],
    "denied": ["denial", "denied", "appeal"],
    "rejected": ["rejection", "reject", "claim"],
    "mail": ["home", "delivery", "mail"],
    "delivery": ["home", "delivery", "mail"],
    "refill": ["refill", "supply", "schedule"],
    "soon": ["refill", "schedule", "early"],
    "runout": ["supply", "emergency", "interim"],
    "supply": ["supply", "emergency", "days"],
    "appeal": ["appeal", "denial", "60"],
    "specialty": ["specialty", "tier", "coinsurance"],
    "deductible": ["deductible", "calendar", "year"],
    "network": ["network", "pharmacy", "out-of-network"],
    "escalate": ["escalation", "escalated"],
    "sideeffects": ["pharmacist", "clinical"],
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def expand_query(tokens: list[str]) -> list[str]:
    out = list(tokens)
    for t in tokens:
        out.extend(_SYNONYMS.get(t, []))
    return out


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    document_type: DocumentType
    title: str
    section: str
    text: str
    tokens: list[str] = field(default_factory=list)
    source_label: str = ""


def chunk_document(doc: Document, *, target_tokens: int | None = None) -> list[Chunk]:
    """Split on markdown headings first, then pack to a token budget.

    Heading-aware chunking matters here: policy documents are organised by
    question ("Refill too soon", "Approval duration"), so a heading boundary is
    almost always a semantic boundary. Blind fixed-width chunking would split
    the answer to "how long is an approval valid" across two chunks.
    """
    target = target_tokens or SETTINGS.rag.chunk_tokens
    lines = doc.body.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = doc.title
    buf: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if buf:
                sections.append((current_title, buf))
            current_title = line[3:].strip()
            buf = []
        elif line.startswith("# "):
            current_title = line[2:].strip()
        else:
            buf.append(line)
    if buf:
        sections.append((current_title, buf))

    chunks: list[Chunk] = []
    for si, (section, body_lines) in enumerate(sections):
        text = "\n".join(body_lines).strip()
        if not text:
            continue
        words = text.split()
        if len(words) <= target:
            pieces = [text]
        else:
            step = target - SETTINGS.rag.chunk_overlap
            pieces = [
                " ".join(words[i: i + target]) for i in range(0, len(words), max(1, step))
            ]
            pieces = [p for p in pieces if len(p.split()) > 20]
        for pi, piece in enumerate(pieces):
            full = f"{section}\n{piece}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.document_id}#s{si}p{pi}",
                    document_id=doc.document_id,
                    document_type=doc.document_type,
                    title=doc.title,
                    section=section,
                    text=piece,
                    tokens=tokenize(full),
                    source_label=doc.source_label,
                )
            )
    return chunks


class BM25Index:
    """Okapi BM25. Small, transparent, and inspectable in a review."""

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.4, b: float = 0.72) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.N = len(chunks)
        self.doc_len = [len(c.tokens) for c in chunks]
        self.avgdl = sum(self.doc_len) / max(1, self.N)
        self.tf: list[Counter] = [Counter(c.tokens) for c in chunks]
        df: Counter = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def score(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.N
        qc = Counter(query_tokens)
        for term, qn in qc.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            weight = 1.0 + 0.25 * (qn - 1)  # mild boost for repeated/expanded terms
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.doc_len[i] / max(1e-9, self.avgdl)
                )
                scores[i] += weight * idf * (f * (self.k1 + 1)) / denom
        return scores


class PolicyRetriever:
    def __init__(self, documents: list[Document], embedder=None) -> None:
        policy = [d for d in documents if d.document_type in POLICY_DOCUMENT_TYPES]
        if not policy:
            raise ValueError("No policy documents supplied to the retriever")
        self.documents = {d.document_id: d for d in policy}
        self.chunks: list[Chunk] = []
        for d in policy:
            self.chunks.extend(chunk_document(d))
        self.bm25 = BM25Index(self.chunks)
        self.embedder = embedder
        self._dense = None
        if embedder is not None:
            self._dense = embedder.embed_many([c.text for c in self.chunks])

    @property
    def index_version(self) -> str:
        return SETTINGS.rag.index_version

    def retrieve(
        self, query: str, *, top_k: int | None = None,
        doc_types: set[DocumentType] | None = None,
        plan_id: str | None = None,
    ) -> list[EvidenceChunk]:
        """Retrieve policy evidence, optionally scoped to a member's plan.

        ``plan_id`` is not an optimisation. Benefit summaries are plan-specific
        and state different deductibles and copays, so an unscoped search can
        return another plan's numbers and ground an answer in them. That is a
        confidently-wrong answer of exactly the kind this system exists to
        avoid, so plan scoping is enforced here rather than left to the caller.
        """
        k = top_k or SETTINGS.rag.top_k
        q = expand_query(tokenize(query))
        lex = self.bm25.score(q)
        mx = max(lex) or 1.0
        scores = [s / mx for s in lex]

        if self._dense is not None and self.embedder is not None:
            qv = self.embedder.embed_one(query)
            dense = self.embedder.similarities(qv, self._dense)
            w = SETTINGS.rag.lexical_weight
            scores = [w * s + (1 - w) * d for s, d in zip(scores, dense)]

        ranked = sorted(
            range(len(self.chunks)), key=lambda i: scores[i], reverse=True
        )
        out: list[EvidenceChunk] = []
        seen_sections: set[tuple[str, str]] = set()
        for i in ranked:
            c = self.chunks[i]
            if doc_types and c.document_type not in doc_types:
                continue
            if (
                plan_id is not None
                and c.document_type == DocumentType.BENEFIT_SUMMARY
                and not c.document_id.endswith(plan_id)
            ):
                continue
            if scores[i] < SETTINGS.rag.min_evidence_score:
                continue
            key = (c.document_id, c.section)
            if key in seen_sections:  # one chunk per section keeps evidence diverse
                continue
            seen_sections.add(key)
            out.append(
                EvidenceChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    document_type=c.document_type,
                    title=f"{c.title} — {c.section}",
                    text=c.text,
                    score=round(float(scores[i]), 4),
                    source_label=c.source_label,
                )
            )
            if len(out) >= k:
                break
        return out

    def sufficiency(self, evidence: list[EvidenceChunk]) -> tuple[bool, str]:
        """Is there enough retrieved evidence to answer at all?

        A deliberately conservative gate. Answering from a single weak chunk is
        how a grounded system quietly becomes an ungrounded one.
        """
        if not evidence:
            return False, "no evidence above the retrieval score floor"
        top = evidence[0].score
        if top < SETTINGS.rag.min_evidence_score * 1.6:
            return False, f"top evidence score {top:.3f} below confidence floor"
        if len(evidence) == 1 and top < 0.45:
            return False, "single weak chunk retrieved"
        return True, "sufficient"


def load_retriever(store=None) -> PolicyRetriever:
    from careweave.data.store import get_store

    store = store or get_store()
    return PolicyRetriever(store.policy_documents)

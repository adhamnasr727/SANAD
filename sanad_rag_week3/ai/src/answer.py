"""
SANAD Grounded Clinical Answer Service (RAG-3, v1).

Summarizes a single authorized patient's synthetic medical timeline,
strictly grounded in retrieved record chunks, with a verifiable citation
for every medical fact stated. Refuses diagnosis/medication-advice
requests and cross-patient queries. Reports missing data explicitly
rather than inferring it.

See ai/src/prompts/grounded.md for the versioned prompt contract this
module implements, and ai/chunking-experiment.md / ai/index-manifest.json
for the retrieval design this builds on (Week 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .guardrails import cross_patient_request, diagnosis_or_medication_advice
from .retriever import Chunk, Retriever

PROMPT_VERSION = "grounded-v1"

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


@dataclass(frozen=True)
class Citation:
    doc_id: str
    chunk_id: str
    quote_span: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    refused: bool = False
    reason: str | None = None


def _quote_span(chunk: Chunk, max_words: int = 25) -> str:
    """A short excerpt of the chunk, taken as an exact PREFIX SLICE of
    chunk.text (not re-joined from .split()), so it is guaranteed to be a
    literal, byte-for-byte substring of the source -- whitespace/newlines
    included -- for mechanical citation verification."""
    import re as _re
    word_matches = list(_re.finditer(r"\S+", chunk.text))
    if len(word_matches) <= max_words:
        return chunk.text.strip()
    end = word_matches[max_words - 1].end()
    return chunk.text[:end]


def _detect_missing(question: str, chunks: list[Chunk]) -> list[str]:
    """Explicit, code-driven gap reporting -- never inferred from the LLM/
    text-generation side. Each entry names a concrete gap."""
    missing: list[str] = []
    for c in chunks:
        if c.date is None:
            missing.append(
                f"Chunk {c.chunk_id} (document {c.document_id}) has no reliable "
                f"date on record; it is excluded from any 'most recent' ordering."
            )
    recency_asked = any(w in question.lower() for w in ("recent", "latest", "current", "last"))
    if recency_asked and not any(c.date for c in chunks):
        missing.append(
            "The question asks for the most recent record, but none of the "
            "retrieved chunks carry a reliable date -- recency cannot be confirmed."
        )
    return missing


def _compose_text(chunks: list[Chunk], citations: list[Citation]) -> str:
    lines = [
        "Summary of the patient's own recorded data (not a diagnosis):",
    ]
    for c, cite in zip(chunks, citations):
        tag = f"[{cite.chunk_id}]"
        lines.append(f"- ({c.record_type}, {c.date or 'date unknown'}) {_first_fact_line(c.text)} {tag}")
    return "\n".join(lines)


def _first_fact_line(text: str) -> str:
    # Skip the "## Type -- date" header line; return the first content line.
    body_lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("##")]
    return body_lines[0] if body_lines else text.strip()


def answer(question: str, *, tenant_id: str) -> Answer:
    """tenant_id is the authorized patient_id for this call -- it MUST come
    from the caller's authenticated session, never parsed from `question`."""

    # 1) Safety guardrails run on the question text alone, before any
    #    retrieval, so a refusal never depends on what would be retrieved.
    reason = diagnosis_or_medication_advice(question)
    if reason:
        return Answer(text="", citations=[], missing=[], refused=True, reason=reason)

    reason = cross_patient_request(question, tenant_id)
    if reason:
        return Answer(text="", citations=[], missing=[], refused=True, reason=reason)

    # 2) Retrieval is pre-filtered to tenant_id only (mandatory isolation
    #    filter -- see ai/index-manifest.json isolation_policy).
    retriever = _get_retriever()
    if tenant_id not in retriever.known_patient_ids():
        return Answer(
            text="", citations=[], missing=[], refused=True,
            reason=f"No authorized record exists for tenant_id={tenant_id!r}.",
        )

    # 3) Topic-absence check -- BEFORE trusting similarity-ranked retrieval.
    #    TF-IDF similarity will happily return the "closest" chunks even when
    #    none of them are actually about what was asked (e.g. asking for a
    #    TSH result on a patient who only has HbA1c on record): those chunks
    #    share enough generic lab-report vocabulary to rank as "similar".
    #    Silently presenting them would violate the never-infer-missing-data
    #    rule, so a specifically-named test/imaging type that doesn't appear
    #    anywhere in this patient's own record is reported as missing and the
    #    answer does NOT fall back to unrelated chunks.
    absent_terms = retriever.terms_absent_for_patient(question, tenant_id)
    if absent_terms:
        missing = [f"No {term} record found for this patient." for term in absent_terms]
        return Answer(
            text=(
                "No matching records were found for this patient for the specific "
                f"test(s)/record(s) asked about: {', '.join(absent_terms)}. "
                "This is reported as missing rather than inferred from other records."
            ),
            citations=[], missing=missing, refused=False, reason=None,
        )

    chunks = retriever.retrieve(question, tenant_id=tenant_id, k=3, token_budget=300)

    # 4) Empty retrieval -> refuse rather than let a downstream generation
    #    step invent an answer with no grounding.
    if not chunks:
        return Answer(
            text="", citations=[], missing=[], refused=True,
            reason="No relevant records were found for this patient and question.",
        )

    # 5) Build mandatory citations -- every fact in `text` is backed by a
    #    Citation whose quote_span is a literal substring of the source
    #    chunk, so it can be mechanically verified (see tests/test_grounding.py).
    citations = [Citation(doc_id=c.document_id, chunk_id=c.chunk_id, quote_span=_quote_span(c)) for c in chunks]

    # 6) Missing data is reported explicitly and only from code-level checks
    #    (date presence, recency askable-or-not) -- never left to free text.
    missing = _detect_missing(question, chunks)

    text = _compose_text(chunks, citations)
    return Answer(text=text, citations=citations, missing=missing, refused=False, reason=None)

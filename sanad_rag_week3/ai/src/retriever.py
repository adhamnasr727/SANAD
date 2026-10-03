"""
Retrieval layer for the SANAD Grounded Answer Service.

Re-uses the semantic/document-aware chunking strategy selected and
validated in ai/chunking-experiment.md (Week 1): one chunk = one clinical
record (encounter / prescription / lab result), with precise per-chunk
`date` metadata that enables date-aware re-ranking for "most recent"
style questions.

patient_id (here: tenant_id) is applied as a MANDATORY pre-ranking filter,
per the isolation_policy in ai/index-manifest.json -- the retriever never
ranks or returns a chunk belonging to a different patient.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DOCS_DIR = os.path.join(DATA_DIR, "documents")
MANIFEST_PATH = os.path.join(DATA_DIR, "documents_manifest.json")

SECTION_RE = re.compile(r"^##\s+(?P<type>[A-Za-z /]+?)\s+--\s+(?P<date>\d{4}-\d{2}-\d{2})\s*$", re.M)
RECORD_TYPE_MAP = {"laboratory_result": "lab_result"}  # align with data-dictionary.csv

RECENCY_WORDS = ("recent", "latest", "current", "last", "newest", "most up to date", "up-to-date")

# "Lab order: X" / "Imaging order: X" -- used to build a corpus-wide
# vocabulary of known test/imaging names, so the service can tell the
# difference between "no data retrieved" and "this specific test was
# asked about but this patient has none on record" (see
# Retriever.terms_absent_for_patient).
_TEST_NAME_RE = re.compile(r"(?:Lab order|Imaging order):\s*([^,]+?),")


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    patient_id: str
    provider: str
    specialty: str
    document_type: str
    record_type: str
    date: Optional[str]
    consent_scope: str
    is_multi_event: bool
    token_count: int
    text: str


def _load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return {d["patient_id"]: d for d in json.load(f)}


def _semantic_chunks(text: str, doc_meta: dict) -> list[Chunk]:
    matches = list(SECTION_RE.finditer(text))
    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        record_type = m.group("type").strip().lower().replace(" ", "_")
        record_type = RECORD_TYPE_MAP.get(record_type, record_type)
        chunks.append(Chunk(
            chunk_id=f"{doc_meta['document_id']}-SM{i:02d}",
            document_id=doc_meta["document_id"],
            patient_id=doc_meta["patient_id"],
            provider=doc_meta["provider"],
            specialty=doc_meta["specialty"],
            document_type=doc_meta["document_type"],
            record_type=record_type,
            date=m.group("date"),
            consent_scope=doc_meta["consent_scope"],
            is_multi_event=False,
            token_count=len(chunk_text.split()),
            text=chunk_text,
        ))
    return chunks


class Retriever:
    """Loads the synthetic corpus once and serves tenant-isolated retrieval."""

    def __init__(self):
        manifest = _load_manifest()
        self._chunks: list[Chunk] = []
        for patient_id, meta in manifest.items():
            path = os.path.join(DOCS_DIR, meta["file"])
            with open(path) as f:
                text = f.read()
            self._chunks.extend(_semantic_chunks(text, meta))

        self._corpus_text = [c.text for c in self._chunks]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._corpus_text)

        # corpus-wide vocabulary of known test/imaging names (e.g. "HbA1c",
        # "Thyroid Function (TSH)"), and which patients actually have each
        # one on record -- built once at load time.
        self._known_test_names: set[str] = set()
        self._patient_text: dict[str, str] = {}
        for c in self._chunks:
            self._patient_text.setdefault(c.patient_id, "")
            self._patient_text[c.patient_id] += " " + c.text
        for text in self._patient_text.values():
            for m in _TEST_NAME_RE.finditer(text):
                self._known_test_names.add(m.group(1).strip())

    def known_patient_ids(self) -> set[str]:
        return {c.patient_id for c in self._chunks}

    def terms_absent_for_patient(self, question: str, tenant_id: str) -> list[str]:
        """Which corpus-known test/imaging names does the question mention
        that do NOT appear anywhere in this patient's own full record?
        Used to explicitly report "this specific thing was asked about and
        this patient has none on record" instead of silently substituting
        unrelated retrieved chunks."""
        q_lower = question.lower()
        patient_text_lower = self._patient_text.get(tenant_id, "").lower()
        absent = []
        for name in self._known_test_names:
            name_lower = name.lower()
            # also check the short form inside parentheses, e.g. "TSH" from
            # "Thyroid Function (TSH)", since questions often use just that
            short = name_lower.split("(")[-1].rstrip(")") if "(" in name_lower else None
            mentioned = name_lower in q_lower or (short and short in q_lower)
            if mentioned and name_lower not in patient_text_lower:
                absent.append(name)
        return absent

    def retrieve(self, question: str, tenant_id: str, k: int = 3, token_budget: int = 300) -> list[Chunk]:
        """Mandatory pre-ranking filter on tenant_id, then TF-IDF similarity,
        then date-aware re-ranking if the question asks for something recent."""
        idxs = [i for i, c in enumerate(self._chunks) if c.patient_id == tenant_id]
        if not idxs:
            return []

        q_vec = self._vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self._matrix[idxs]).flatten()
        ranked = sorted(zip(idxs, sims), key=lambda x: -x[1])

        if any(w in question.lower() for w in RECENCY_WORDS):
            shortlist = ranked[:8]
            dated = sorted(
                [(i, s) for i, s in shortlist if self._chunks[i].date],
                key=lambda x: self._chunks[x[0]].date, reverse=True,
            )
            undated = [(i, s) for i, s in shortlist if not self._chunks[i].date]
            ranked = dated + undated

        selected, used = [], 0
        for i, _ in ranked:
            c = self._chunks[i]
            if len(selected) >= k and used >= token_budget:
                break
            if used + c.token_count > token_budget and selected:
                continue
            selected.append(c)
            used += c.token_count
            if len(selected) >= k:
                break
        return selected

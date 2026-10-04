"""
Safety guardrail checks for the SANAD Grounded Answer Service.

These run BEFORE retrieval, on the question text alone, so a refusal never
depends on what the retriever happens to return. This matches the task's
"Questions naming or querying another patient's data must be refused
immediately" and "must strictly refuse any request to diagnose conditions
or recommend medications/dosages" requirements.

This is a regex/keyword-based guard, not an NLU classifier -- it is
deliberately conservative (over-refuses ambiguous phrasing rather than
risk under-refusing). This limitation is documented in ai/README.md.
"""
from __future__ import annotations

import re

# Patterns that indicate the caller wants a diagnosis or treatment/medication
# decision made FOR them, rather than a summary of their own recorded data.
_DIAGNOSIS_PATTERNS = [
    r"\bdiagnos\w*\b",
    r"\bwhat (disease|condition|disorder) do i have\b",
    r"\bdo i have (cancer|diabetes|covid|an? \w+ (disease|disorder|condition))\b",
    r"\bis this (cancer|serious|dangerous|normal)\b",
    r"\bwhat'?s wrong with me\b",
    r"\bwhat should i (take|do) for\b",
    r"\bshould i (take|start|stop|increase|decrease)\b",
    r"\bwhat (medication|drug|dose|dosage) should\b",
    r"\bprescri\w*\b",
    r"\brecommend\w* (a |an |)(medication|drug|treatment|dose|dosage)\b",
    r"\bwhat treatment\b",
    r"\bhow much (\w+ )?(mg|medication|dose)\b",
    r"\b(right|correct|appropriate|best|safe|recommended|ideal)\s+(dose|dosage)\b",
    r"\b(dose|dosage)\b.*\bfor me\b",
]
_DIAGNOSIS_RE = re.compile("|".join(_DIAGNOSIS_PATTERNS), re.IGNORECASE)

# A question naming another patient_id, or asking to compare/aggregate
# across patients, is cross-patient and must be refused regardless of
# what the retriever would return.
_PATIENT_ID_RE = re.compile(r"\bPAT-\d+\b", re.IGNORECASE)
_CROSS_PATIENT_PATTERNS = [
    r"\ball patients\b",
    r"\bother patients\b",
    r"\bevery patient\b",
    r"\bcompare\b.*\bpatient\b",
    r"\bacross patients\b",
]
_CROSS_PATIENT_RE = re.compile("|".join(_CROSS_PATIENT_PATTERNS), re.IGNORECASE)


def diagnosis_or_medication_advice(question: str) -> str | None:
    """Returns a refusal reason string if the question asks for a
    diagnosis or medication/dosage recommendation, else None."""
    if _DIAGNOSIS_RE.search(question):
        return (
            "This service summarizes recorded clinical data only. It does not "
            "diagnose conditions or recommend medications or dosages. Please "
            "consult a licensed clinician."
        )
    return None


def cross_patient_request(question: str, tenant_id: str) -> str | None:
    """Returns a refusal reason string if the question names a patient_id
    other than tenant_id, or asks for cross-patient comparison/aggregation."""
    named_ids = {m.group(0).upper() for m in _PATIENT_ID_RE.finditer(question)}
    other_ids = named_ids - {tenant_id.upper()}
    if other_ids:
        return (
            f"This question references patient record(s) outside the authorized "
            f"tenant ({tenant_id}). Cross-patient queries are not permitted."
        )
    if _CROSS_PATIENT_RE.search(question):
        return (
            "This question requests data across multiple patients. Each query "
            "is strictly scoped to a single authorized patient."
        )
    return None

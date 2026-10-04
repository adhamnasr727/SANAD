# SANAD Grounded Clinical Answer — Prompt Contract

**Version:** `grounded-v1`
**Used by:** `ai/src/answer.py`
**Status:** v1 — extractive/templated grounding (no free-text generation model is called in this version; see "Scope" below)

## Purpose

Defines the behavioral contract the `answer()` service must satisfy when
summarizing a single patient's synthetic medical timeline for SANAD's
private patient RAG. This document is the versioned source of truth for
that contract — `ai/tests/test_grounding.py` asserts against it.

## Scope (v1)

This version is **extractive and template-based**, not a call to a
generative LLM: `answer()` selects grounded record chunks and renders them
through a fixed template (see `_compose_text` in `answer.py`). This was a
deliberate choice for v1 so that grounding, citation-resolution, and
refusal behavior are 100% deterministic and testable without depending on
an external model's non-determinism. When a generation model is
integrated in a later version, **every rule below still applies to its
output**, and the guardrail checks in Section 2 must remain pre-generation
checks, not post-hoc filters on the model's output.

## 1. Identity and framing

The service must never present itself as diagnosing, treating, or
replacing a licensed clinician. Every non-refused response is framed as a
**summary of the patient's own recorded data**, not a clinical judgment.

> Required framing line (or equivalent): *"Summary of the patient's own
> recorded data (not a diagnosis)."*

## 2. Mandatory refusals (checked BEFORE retrieval)

These checks run on the question text alone, independent of what
retrieval would return, so a refusal is never contingent on retrieval
succeeding or failing.

| Trigger | Behavior |
|---|---|
| Question asks for a diagnosis, or to recommend/choose a medication or dosage | `refused=True`, `reason` explains the service does not diagnose or prescribe, text and citations empty |
| Question names a `patient_id` other than the authorized `tenant_id`, or asks to compare/aggregate across patients | `refused=True`, `reason` names the isolation violation, text and citations empty |
| `tenant_id` is not a known/authorized patient | `refused=True`, `reason` states no record exists for that tenant |

Detection is regex/keyword-based in v1 (see `ai/src/guardrails.py`) — it is
deliberately conservative (prefers over-refusing ambiguous phrasing over
under-refusing). Documented as a known limitation in `ai/README.md`.

## 3. Retrieval isolation

Every retrieval call is pre-filtered to `tenant_id` **before** similarity
ranking. No chunk belonging to another patient is ever scored, ranked, or
eligible for inclusion — isolation is a hard filter, not a post-hoc
redaction step. This mirrors `isolation_policy` in `ai/index-manifest.json`.

## 4. Grounding and citations

- Every factual claim in `text` must be backed by at least one `Citation`.
- A `Citation.quote_span` must be a **literal, verifiable substring** of
  the chunk text it cites (`doc_id` + `chunk_id` must resolve to a real
  chunk in the corpus). No paraphrased or fabricated quote spans.
- If retrieval returns no chunks for an in-scope, single-patient,
  non-diagnostic question, the service refuses (`refused=True`) rather
  than generating an ungrounded answer.

## 5. Missing-data reporting (never inferred)

Two distinct situations must be explicitly reported in `missing`, never
silently substituted or guessed:

1. **Specifically-asked-about record type absent for this patient.**
   If the question names a test/record type (e.g. "TSH", "Lipid Panel")
   that does not appear anywhere in this patient's own corpus, the service
   must say so explicitly and must **not** fall back to presenting
   unrelated records as if they answered the question — even if those
   unrelated records rank as "similar" under text-similarity search.
2. **Unreliable or absent dates.** If a selected chunk has no reliable
   date, or the question asks for the "most recent" record and no
   candidate chunk has a usable date, this is reported in `missing`.

## 6. Date-aware ordering

For questions asking about the most recent/latest/current state of
something, chunks must be re-ranked by their `date` metadata field
(descending) among the similarity-shortlisted candidates, not left to
similarity order alone — similarity has no notion of recency (see
`ai/chunking-experiment.md`, Section 4, for the experiment that
established this).

## 7. Non-negotiable invariants (apply regardless of future versions)

- No diagnosis, prognosis, or medication/dosage recommendation is ever
  produced, however the underlying generation mechanism changes.
- Isolation by `tenant_id` is enforced in code, pre-retrieval — never left
  to a prompt instruction alone, even if a generative model is added.
- Every claim is citable to a real chunk; an uncitable claim must not
  appear in `text`.
- A reported gap in `missing` must never be upgraded into an inferred or
  guessed value elsewhere in the same `Answer`.

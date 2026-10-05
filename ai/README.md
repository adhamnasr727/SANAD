# SANAD — AI/RAG Engineering: Grounded Clinical Answer Service (RAG-3, v1)

Week 3 deliverable. Builds on Week 1's chunking/metadata work
(`ai/chunking-experiment.md`, `ai/index-manifest.json`) to implement an
isolated, grounded, citation-backed answer service for a single
authorized patient's synthetic medical timeline.

## Repository layout

```
ai/
  src/
    __init__.py
    answer.py          # Clinical answering service & safety guardrails
    guardrails.py       # Pre-retrieval refusal checks (diagnosis/med advice, cross-patient)
    retriever.py        # Semantic chunking + TF-IDF retrieval + isolation filter
    prompts/
      grounded.md        # Versioned clinical prompt contract
  tests/
    __init__.py
    test_grounding.py    # Clinical safety & grounding pytest suite
  samples/
    generate_samples.py  # Regenerates qa-10.json by actually running answer()
    qa-10.json            # 10 real execution logs
  README.md               # This file
```

## Dependencies

- Python 3.11
- `scikit-learn` (TF-IDF retrieval) — `pip install scikit-learn`
- `pytest` (test suite) — `pip install pytest`

No LLM API is called in this version — see "Scope / known limitations"
below.

## Indexing

There is no separate offline indexing step or persisted index file in v1.
`Retriever.__init__()` (in `src/retriever.py`) builds the index **in
memory, on first use**, each time the process starts:

1. Reads every `data/documents/*.txt` timeline listed in
   `data/documents_manifest.json`.
2. Splits each one into semantic chunks (one chunk per `## Type -- date`
   clinical record — see `ai/chunking-experiment.md` for why this strategy
   was chosen over fixed-token windows).
3. Fits a single TF-IDF vectorizer over the full chunk corpus
   (`sklearn.feature_extraction.text.TfidfVectorizer`) and builds a
   corpus-wide vocabulary of known test/imaging names (used by
   `terms_absent_for_patient`, see Architecture below).

`answer.py` lazily constructs one `Retriever` instance per process
(`_get_retriever()`) and reuses it across calls — the corpus is small
enough (16 synthetic patients) that re-indexing on every call would be
wasteful but harmless; a production corpus would need a persisted,
incrementally-updatable index instead of rebuilding from flat files in
memory on every process start.

## Synthetic data disclaimer

All patient data under `data/documents/` is **entirely fabricated** for
development/testing. No real patient data is used anywhere in this
repository, per FR-SAN-34. See `data/generate_data_v3.py` for the
generator (seeded, reproducible) and `ai/chunking-experiment.md` for a
full description of the corpus (16 synthetic patients, 8 diagnosis
categories, 2 "confusable" patients per category).

## Running

```bash
# from repo root
cd ai

# run the safety/grounding test suite
python3 -m pytest tests/test_grounding.py -q

# regenerate the 10 sample execution logs (writes samples/qa-10.json)
python3 samples/generate_samples.py
```

## Using the service

```python
from src.answer import answer

result = answer("What was PAT-3003's most recent HbA1c result?", tenant_id="PAT-3003")
print(result.refused, result.text, result.citations, result.missing)
```

`tenant_id` **must** come from the caller's authenticated session — never
parsed from the question text itself. The service treats it as the sole
source of truth for which patient's records may be retrieved.

## Architecture

```
question, tenant_id
       │
       ▼
1. guardrails.diagnosis_or_medication_advice(question)
       │  (refuse before any retrieval)
       ▼
2. guardrails.cross_patient_request(question, tenant_id)
       │  (refuse before any retrieval)
       ▼
3. retriever.terms_absent_for_patient(question, tenant_id)
       │  (specifically-asked-about test/record type not in this
       │   patient's corpus -> report in `missing`, do not substitute
       │   unrelated chunks)
       ▼
4. retriever.medications_absent_for_patient(question, tenant_id)
       │  (named medication absent from this patient's prescriptions
       │   -> report in `missing`, do not substitute another medication)
       ▼
5. retriever.retrieve(question, tenant_id)
       │  (patient_id is a mandatory PRE-ranking filter; zero-match -> refuse)
       │  (medication questions rank only prescription/encounter chunks)
       │  (current medication questions select the newest eligible record)
       ▼
6. build Citations (doc_id, chunk_id, quote_span -- a literal, exact
   substring of the source chunk) + detect missing dates
       ▼
7. Answer(text, citations, missing, refused, reason)
```

### Why a topic-absence check exists (Section 3 above)

Early testing showed that similarity-based retrieval (TF-IDF, and this
would equally apply to an embedding model) will happily return the
"closest" chunks even when none of them actually address what was asked
— e.g. asking a diabetes patient for their TSH (thyroid) result returns
their HbA1c records instead, because both are lab-report chunks sharing
enough generic vocabulary to rank as similar. Silently presenting that as
an answer would violate the "missing data must be explicitly reported,
never inferred" requirement. `terms_absent_for_patient` builds a
corpus-wide vocabulary of known test/imaging names and checks whether a
name mentioned in the question is actually present anywhere in *this*
patient's own record before trusting similarity ranking at all.

## Scope / known limitations (v1)

- **Extractive, not generative.** `answer()` selects and templates
  grounded chunks; it does not call a generative LLM. This was a
  deliberate choice so that grounding, citation-resolution, and refusal
  behavior are fully deterministic and testable. `ai/src/prompts/grounded.md`
  documents which rules still apply unchanged if/when a generation model
  is added — in particular, guardrail checks must remain pre-generation,
  not a filter on the model's free-text output.
- **Guardrails are regex/keyword-based, not NLU.** `guardrails.py` is
  deliberately conservative (prefers over-refusing ambiguous phrasing
  over under-refusing a real diagnosis/cross-patient request), but it can
  still be evaded by sufficiently indirect phrasing, and can over-refuse
  legitimate questions that happen to contain a trigger word. A
  production version should add an NLU/LLM-based classifier as a second
  layer, not a replacement, for these checks.
- **Retrieval uses TF-IDF**, not a production embedding model — carried
  over from the Week 1 finding that this disadvantages lexical-mismatch
  queries. Medication-intent queries receive a small synonym expansion and
  are restricted to prescription/encounter records, but other lexical
  mismatches may still return no result. Zero-similarity queries are refused
  rather than answered with arbitrary chunks. See `ai/chunking-experiment.md`
  Section 5.
- **`terms_absent_for_patient` is heuristic**, built from literal
  `"Lab order: X,"` / `"Imaging order: X,"` patterns in this synthetic
  corpus. A production system needs a real controlled vocabulary of
  clinical test/record types, not a string extracted from report text.
- **Isolation is enforced in application code**, not at the vector
  database layer. `tests/test_grounding.py` confirms the application-level
  filter holds; it does not prove a vector-DB-level tenant boundary (e.g.
  per-patient namespaces/collections), which SANAD's production
  deployment should still add as defense-in-depth per
  `ai/index-manifest.json`'s `isolation_policy`.

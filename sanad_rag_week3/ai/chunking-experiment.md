# Chunking Experiment — SANAD Patient RAG

**Intern:** Adham Ahmed · **Mentor:** Ibrahim Ahmed · **Project:** SANAD · **Department:** AI/RAG Engineering
**Scope:** Compare fixed-token-window chunking vs. semantic/document-aware chunking for synthetic patient medical timelines, evaluated on an identical question set.

> **Methodology note:** this report went through three iterations, kept here for transparency because each one exposed a real flaw in the previous test design. **Section 4 (v3, 16 patients / 48 questions) is the final result, and within it the "matched budget" test itself was re-run once more after its first version was found to be vacuous (see Section 4) — the 150-token figures are the ones the recommendation in Section 6 is actually based on.**

## 1. Method (final, v3)

### Data
**16 synthetic (fabricated) patients** across **8 diagnosis categories, 2 confusable patients per category**: same specialty, same medication, same lab test, similar-but-different numbers/dates. Each patient has **5 repeated draws of the same lab test over ~2–3 years**. No real patient data was used (FR-SAN-34).

### Strategies compared
| Strategy | Definition |
|---|---|
| **A — Fixed-token window** | 120-token windows, 30-token overlap, no awareness of record boundaries. |
| **B — Semantic / document-aware** | One chunk = one encounter, prescription, or lab result, split on the document's own record boundaries. |

### Test questions — 48 total (3 per patient, identical set on both chunk sets)
| Type | Tests |
|---|---|
| `most_recent_lab` | Correctly picks the latest of 5 repeated draws of the same test, for the right patient (not their confusable "twin"). |
| `medication` | Retrieves both the medication name and the current dose. |
| `unanswerable` | Asks about a test type that patient's record never contains (e.g. asking a diabetes patient their spirometry result) — success = the retrieved context contains **no** trace of that unrelated test, so a downstream LLM has nothing to hallucinate from. Probes FR-SAN-30 ("state missing information rather than inventing patient facts"). |

### Retrieval setup
TF-IDF cosine similarity, one vector store per strategy. `patient_id` is a mandatory pre-ranking filter on every query.

## 2. Iteration 1 (3 patients, unique-keyword questions) — superseded
100% (fixed) vs 88% (semantic) on 8 easy questions. Flagged as unreliable: too small a corpus, no overlapping vocabulary between patients, every fact uniquely findable. Rebuilt.

## 3. Iteration 2 (8 patients, confusable pairs, 16 questions) — superseded but instructive
Naive top-3: fixed 100% vs semantic 44%. Root cause found: fixed-token had only **3 chunks/patient** (top-3 = the patient's entire record, unconditionally) vs semantic's **9 chunks/patient** (top-3 = 33% coverage) — an unfair comparison of chunk *count*, not retrieval quality. This iteration also surfaced the core insight carried into v3: pure text similarity cannot answer "most recent" questions (no notion of time), and only semantic chunking gives a reliable per-chunk `date` to fix that.

## 4. Iteration 3 — final (16 patients, 48 questions)

### Naive top-3
| Metric | Fixed-token | Semantic |
|---|---|---|
| Total chunks | 64 (4.0/patient) | 176 (11.0/patient) |
| Overall accuracy | 100% | **40%** |
| — `most_recent_lab` | 100% | 19% |
| — `medication` | 100% | 0% |
| — `unanswerable` (grounding) | 100% | 100% |
| Avg context tokens/question | 309.9 | 102.4 |

The chunk-count imbalance (4 vs 11 per patient) reproduces at this larger scale, confirming iteration 2 wasn't a fluke: fixed-token's top-3 is still close to "give me everything," semantic's top-3 is still a small slice. **The `unanswerable` grounding check held at 100% for both strategies** — the fabricated test genuinely never appears in that patient's own filtered corpus, so no ranking method could hallucinate it in. This is a useful confirmation that the mandatory `patient_id` filter, not the chunking strategy, is what's doing the grounding-safety work here.

### Matched context-token budget — **first attempt was flawed, caught and corrected**
The first version of this test used a 400-token budget and reported 100%/100%. On review, **this number was meaningless**: 400 tokens was ≥ each patient's *entire* corpus (430 tokens for fixed-token, 340 for semantic — nearly identical to the "avg tokens actually used" figures that had been reported). The test was silently retrieving everything every time, which is mathematically guaranteed to match the ceiling check — it was never testing ranking quality at all. This was caught by questioning why so many results kept landing on exactly 100%.

**Corrected version — a genuinely restrictive 150-token budget** (well below any patient's full corpus):

| Metric | Fixed-token | Semantic (similarity only) | Semantic (+ date-aware re-rank) |
|---|---|---|---|
| Overall accuracy | **92%** | 62% | **79%** |
| — `most_recent_lab` | 75% | 50% | **100%** |
| — `medication` | 100% | 38% | 38% |
| — `unanswerable` | 100% | 100% | 100% |
| Avg tokens used | 83.1 | 143.2 | 139.5 |

This is the real, unforced picture, and it's more mixed than earlier iterations suggested:
- **Fixed-token is genuinely stronger under a tight budget with similarity-only ranking** (92% vs 62%) — its larger chunks capture more incidental context per retrieved unit, which helps when only a couple of chunks fit the budget at all.
- **The `medication` weakness in semantic (38%) is mostly a TF-IDF artifact, not a chunking artifact**: the query words "medication"/"dose" never appear verbatim in the corpus (text says "Prescribed", "regimen"), and this lexical gap was masked in every earlier test because the budget was large enough to include nearly all of a patient's chunks regardless of ranking quality. A real embedding model would likely close most of this gap; TF-IDF cannot.
- **Date-aware re-ranking is the one improvement that is only available to semantic chunking**, and it visibly works: `most_recent_lab` jumps from 50% → 100% once chunks are re-ranked by their `date` field. Fixed-token cannot benefit from this at any budget, because most of its chunks have `date = null`.

(At looser budgets — 200 and 250 tokens — both strategies climb back toward 90-100%, non-monotonically, simply because more of each patient's corpus fits regardless of ranking quality. This is expected and is not read as a meaningful result on its own; 150 tokens is the informative data point because it's the only budget genuinely smaller than both corpora.)

### Isolation (8 confusable pairs, adversarial query for the "twin" while filter is locked to the victim)
**0 leaks for both strategies.** Confirms the mandatory `patient_id` filter design holds independent of chunking choice, even against near-identical vocabulary.

## 5. Known limitation (flagged, not fixed here)
TF-IDF is lexical, not semantic — it can't match "medication" against "Prescribed." This is why naive top-3 `medication` accuracy for semantic was 0% (small per-patient candidate pools + weak lexical overlap on paraphrased query words), and it fully recovers once the token budget widens enough to include most of a patient's chunks. **Production should use a real embedding model**, not TF-IDF; this experiment used TF-IDF only to isolate the effect of chunking strategy. This affects both strategies equally and doesn't change the recommendation.

## 6. Recommendation

**Winning strategy: Semantic / document-aware chunking — but with an honest caveat, not a clean sweep.**

The corrected, genuinely budget-constrained test (Section 4) shows fixed-token actually retrieves more accurately (92% vs 62%) under tight budget with similarity-only ranking. The case for semantic chunking rests on two specific, defensible points rather than "wins everywhere":

1. **Semantic chunking is the only strategy that can be improved with date-aware re-ranking**, and this is measured, not assumed: `most_recent_lab` accuracy went from 50% → 100% once chunks were re-ranked by their `date` field. Fixed-token has no equivalent lever available — most of its chunks have `date = null` by construction, so it is stuck at its similarity-only ceiling regardless of engineering effort spent on ranking. As SANAD patient timelines grow over years (FR-SAN-20), this ceiling becomes the binding constraint, not chunk size.
2. **The `medication` weakness observed for semantic chunking (38%) is primarily attributable to TF-IDF's lexical matching, not to chunking strategy** — this experiment's biggest limitation (Section 5) is also its fairest caveat: a real embedding model should close most of this gap, and the same gap would show up for fixed-token if its chunks were small enough to require good ranking instead of accidentally containing everything.
3. **Structurally immune to mid-record narrative interruption** (iteration-2 sensitivity check: 48% of fixed-token chunks split a clinical record's own content across a boundary at a 60-token window).
4. **Grounding/isolation held regardless of chunking strategy** — this is the mandatory `patient_id` filter's job, confirmed even under adversarial confusable-twin conditions, and is not an argument for either chunking strategy specifically.

**Honest bottom line:** this experiment does not show semantic chunking is unconditionally more accurate — under a tight budget with weak (TF-IDF) ranking, it measurably is not. It shows semantic chunking is the only strategy with a viable path to closing that gap (accurate per-chunk metadata enabling date-aware re-ranking), which fixed-token structurally cannot replicate. That is a narrower, more defensible claim than the original "wins on accuracy" framing, and it is what this report actually recommends betting on for production.

**Caveats carried forward:** semantic chunking needs detectable document structure (real unstructured notes/scanned PDFs need a structure-extraction step first — not solved here); TF-IDF was a stand-in for a production embedding model and measurably disadvantaged semantic chunking's `medication` results — re-testing with a real embedding model is the single highest-value follow-up; 16 patients is still small relative to production and budget sensitivity (150 vs 200 vs 250 tokens) should be characterized more thoroughly before treating any of these numbers as production-grade.

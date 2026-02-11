# Ralph-Ready Phase 0 Plan

## GovPal — Local RAG Prototype + Eval Harness (Nonstop Run)

### 0) Ralph Run Contract

**Mission:** Produce a working local RAG prototype (ingest → retrieve/rerank → answer w/ citations) plus an evaluation harness that yields a **binary decision**: **GO / GO* / NO-GO**.

**Single source of truth for completion:**
✅ Work is complete **only** when:

* `python scripts/check_done.py` returns exit code **0**

If it returns **1**, the run is not done, regardless of “it looks good.”

**Nonstop execution:**
This plan is designed for **continuous iteration** until DONE or a hard stop is reached (runtime/cost/no-progress).

---

## 1) Definition of Done (Binary)

A run is DONE when all of the following are true **from a fresh clone** (only `.env` edits allowed):

1. `make setup` succeeds
2. `make ingest` succeeds
3. `make eval` succeeds and writes `reports/eval-latest.json`
4. `make ui` launches Streamlit UI and answers questions with citations
5. Quality thresholds pass:

   * **GO*** thresholds met on the full eval set:

     * recall ≥ **70%**
     * citation coverage ≥ **80%**
     * hallucination ≤ **5%**
   * **GO** requires GO* thresholds also pass on **real-data subset** *if real docs exist*
6. `python scripts/check_done.py` exits **0**

---

## 2) Scope Boundaries (Phase 0)

### In scope

* Local Python prototype (no production infra)
* PDF ingestion pipeline:

  * OCR (Marker, local)
  * PII redaction (Presidio, local)
  * chunking
  * embeddings
  * local persistent vector store (ChromaDB)
* Query pipeline:

  * embed query → retrieve → rerank → generate answer with citations
* Evaluation harness:

  * gold Q/A set
  * claim-level scoring
  * JSON + Markdown report artifacts
* Streamlit UI prototype

### Out of scope

* Scraping automation at scale
* Production deployment, auth, billing, CI/CD
* “Real product” UX polish
* Cloud DB/hosting (beyond model APIs)

**Rule:** If it doesn’t move the metrics toward DONE, it’s out of scope.

---

## 3) Non-Negotiable Constraints (Guardrails)

These are hard rules to prevent false “GO” and unsafe behavior:

1. **Claim-level citations required**
   Every factual sentence must cite `[doc_id, page, snippet/section]` or the system must refuse.

2. **Tier behavior enforced**

   * Tier 1 (bylaws, OCPs, zoning) can be synthesized with citations.
   * Tier 2 (minutes/agenda) must be excerpt-first and avoid speculative synthesis.

3. **PII before vectors**
   PII must be removed **before embeddings** and must never enter the vector DB.

4. **Deterministic evaluation**
   All eval-time LLM calls use `temperature=0`. All configs logged.

5. **No legal opinions**
   Prompts must enforce: summarize the documents, don’t provide legal advice, don’t interpret beyond citation.

6. **No secrets in repo**
   `.env` must be ignored and never printed in logs.

7. **Pinned dependencies**
   Use a lockfile (uv/requirements) and keep a single reproducible setup path.

---

## 4) Inputs (What the run requires)

### 4.1 Source documents

Minimum:

* ≥ **20 PDFs** mixed across Tier 1 and Tier 2

Provenance:

* Real vs synthetic must be tracked in a manifest (so the eval can report both).

### 4.2 Evaluation dataset

Minimum:

* ≥ **30 Q/A items**
* Must include:

  * `expected_claims` (for claim-level scoring)
  * ≥ **3 unanswerable** questions
* The eval set becomes **immutable** after baseline is established (anti-gaming).

### 4.3 Directory conventions

Use stable directories so the agent can operate without guessing:

* `data/raw/...` for PDFs
* `data/processed/...` for OCR + redacted text + chunks
* `data/chroma/...` for persistent ChromaDB
* `eval/...` for the eval set
* `reports/...` for all benchmark outputs

---

## 5) Dependencies & External APIs

### External services (API calls)

* LLM: Claude (or the designated Anthropic model)
* Embeddings: Cohere embeddings
* Rerank: Cohere reranker

### Local dependencies

* OCR: Marker
* PII: Presidio
* Vector store: ChromaDB persistent

**Credential rule:** only `.env` contains keys; never hardcode.

---

## 6) Repo Deliverables (Hard requirements)

The run must output these artifacts (and `check_done.py` should validate presence + freshness):

### Core outputs

* `reports/eval-latest.json`
* `reports/eval-latest.md` (human-readable summary)
* `reports/eval-baseline.json`
* `reports/eval-best.json` (best run so far)
* `prototype/app.py` Streamlit UI
* Prompt files for Tier 1 + Tier 2 + judge/eval prompts
* `docs/architecture-phase0.md` (pipeline + data model summary)
* `reports/iteration-log.md` (every iteration recorded)

### Benchmarks (must exist, even if “rough”)

* `reports/ocr-benchmark.md`
* `reports/presidio-benchmark.md`
* `reports/chunking-benchmark.md`

---

## 7) Command Surface (One happy path)

A Ralph run only works if there’s a single deterministic way to run it.

Required Makefile targets:

* `make setup`
* `make ingest`
* `make eval`
* `make ui`
* `make check-done`
* `make iterate` (recommended): clean → ingest → eval

**Rule:** The agent must never invent its own run sequence; it uses these.

---

## 8) Internal API Contracts (So the agent can’t “wing it”)

Define stable module/function boundaries so iteration doesn’t turn into refactors.

### Ingestion

* Input: PDFs + manifest
* Output: OCR text, redacted text, chunk objects, metadata

### Chunking

* Strategy: one of a small set (e.g., fixed / markdown / sentence)
* Must preserve metadata: `doc_id`, `page`, `section`, offsets/snippets

### PII

* Must emit:

  * redacted text
  * PII report (counts/types)
* Must run before embedding

### Retrieval / Rerank

* Must output:

  * retrieved chunk IDs
  * reranked list
  * final cited support set

### Answering

* Must produce:

  * answer text
  * citation map per claim/sentence

### Evaluation

* Must compute:

  * recall (% claims matched)
  * citation coverage (% claims with valid citation)
  * hallucination (% claims unsupported)

---

# 9) Nonstop Ralph Loop (The heart of the plan)

## 9.1 Global Hard Stops

Stop immediately if any of these triggers:

* ✅ `python scripts/check_done.py` exits **0**
* Runtime exceeds **T** (default suggestion: 12 hours)
* Cost exceeds **$B** (default suggestion: $20)
* No improvement in the primary failing metric after **N** iterations (default: 6)
* A blocked action is proposed (prod deploy, secrets changes, deleting raw datasets, widening network access, etc.)

## 9.2 Baseline + Best-So-Far

* First successful end-to-end eval becomes **baseline**:

  * `reports/eval-baseline.json`
* Every subsequent iteration compares against:

  * baseline
  * best-so-far (`reports/eval-best.json`)

## 9.3 Allowed Changes (Phase 0 only)

The agent may only change:

* chunking strategy/parameters
* retrieval parameters (top_k, filters)
* rerank parameters (rerank_top_n)
* prompts (Tier 1/Tier 2/judge)
* PII thresholds/rules (only if benchmarks justify it)
* bugfixes in ingestion/citation propagation

**Not allowed:**

* new product features
* major refactors
* swapping stack
* building infra/deployments

## 9.4 Iteration Procedure (must follow every time)

Each iteration must do **exactly** this:

1. **Select target**
   Identify the single worst failing metric (or smallest margin to threshold).

2. **Diagnose**
   Write 1–3 bullets on why it’s failing, using logs + eval outputs.

3. **Make one atomic change**
   One conceptual change only (single knob or single prompt edit).

4. **Run**

   * `make iterate` (or `make ingest && make eval`)
   * ensure artifacts updated

5. **Compare**

   * If any gating metric regresses below threshold or worsens materially → **revert**
   * If target metric improves without unacceptable regressions → **keep**
   * Update `eval-best.json` only when it’s strictly better than current best

6. **Log**
   Append to `reports/iteration-log.md`:

   * iteration #
   * change summary
   * before/after metrics
   * keep/revert decision + reason

7. **Check DONE**

   * run `python scripts/check_done.py`

## 9.5 Checkpoints (for nonstop control)

Every **3 iterations** (or every **90 minutes**, whichever comes first), write `reports/checkpoint.md` containing:

* current best metrics
* top remaining failure modes
* next 1–2 planned interventions
* any anomalies (flaky OCR, citation mismatch, etc.)

If the agent cannot produce a coherent checkpoint → stop (it’s drifting).

---

## 10) Final Output: GO / GO* / NO-GO

At the end of the run, the agent must output:

* **GO*** if thresholds pass on the full eval set
* **GO** if thresholds pass on real-data subset (when real docs exist)
* **NO-GO** if it fails after iteration limit or hits hard stop

Also include a short final report:

* what changed
* what improved
* what is still weak
* recommended next step (Phase 1 or fix list)

---

# What you did well (already Ralph-friendly)

* You already defined **binary DONE** tied to an executable checker and a reproducible command path.
* You picked concrete thresholds (recall, citation coverage, hallucination) rather than vibes.
* You separated **GO vs GO*** to avoid synthetic-only confidence.
* You included anti-gaming ideas (unanswerables, claim-level scoring).
* You focused Phase 0 on proving the pipeline/evals instead of infra.

# What I improved in this rewrite

* Removed “days” and converted everything to a **continuous loop** with **hard stops** (time/cost/no-progress).
* Formalized the **iteration protocol** (atomic change → run → compare → keep/revert → log).
* Added **baseline + best-so-far** mechanics so progress is monotonic and measurable.
* Added explicit **allowed / not allowed changes** to prevent scope creep.
* Added **checkpoint artifacts** to keep long runs controlled and debuggable.
* Tightened deliverables into a **must-produce artifact list** so the agent can’t declare victory early.

---

If you want, I can also rewrite this into a **single “Agent Run Prompt”** (copy/paste) that includes: role, tools allowed, exact commands, iteration template, and strict refusal rules — so you can hand it directly to your agent runner.

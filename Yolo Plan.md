Here are the **updated sections** you can drop straight into your Ralph plan (and then I’m also including a **fully rewritten plan** with these changes integrated).

---

## Updated sections to insert into the Ralph plan

### A) Document Tiering Policy

**Rule:** Every ingested document must be assigned exactly one tier **before** any OCR/PII/chunking/embedding. Tier is persisted in (1) the raw manifest and (2) every chunk’s metadata in the vector store.

**Tier definitions**

* **Tier 1 — Normative / stable rules (low PII, policy text)**

  * Examples: bylaws, zoning bylaw text, OCP policies, official regulations, schedules that are purely regulatory.
* **Tier 2 — Proceedings / meeting artifacts (incidental PII is common)**

  * Examples: council minutes, agendas, public hearing minutes, committee minutes.
* **Tier 3 — Attachments / mixed / unstructured (highest PII + noise risk)**

  * Examples: appendices/exhibits, letters/emails from the public, submissions, forms, scanned attachments, complaint logs, anything “misc.”

**Classification method (deterministic)**

* Classification uses a deterministic rule set (filename patterns + folder conventions + first-page heuristics).
* **Safe default:** If uncertain, classify as **Tier 3**.

---

### B) Tier-Specific Processing Pipeline (PII + Embeddings)

**Core principle:** **Minutes must be PII scrubbed; bylaws do not need scrubbing.**
So PII is *not universal*—it’s tier-dependent.

#### Tier 1 processing

* OCR → text cleanup → chunking → embedding → index
* **PII stage: skipped by default**
* Rationale: keep maximal fidelity for retrieval and citations.

#### Tier 2 processing

* OCR → **PII detection + redaction** → text cleanup → chunking → embedding → index
* **PII redaction is mandatory before embeddings**
* Store:

  * `redacted_text`
  * `pii_report` summary (counts/types) per doc

#### Tier 3 processing

* OCR → **PII detection + redaction (stricter)** → cleanup → chunking → embedding → index
* **Stricter redaction** than Tier 2 (more detectors, lower thresholds).
* Optional safety valve:

  * If “PII density” exceeds a threshold (e.g., findings/page), mark doc as `do_not_embed=true` and store only an excerpt index or metadata-only record.

**Non-negotiable invariant:** PII from Tier 2/3 must never enter the vector store.

---

### C) Tier-Specific Answer Policy (Generation Rules)

The answering stage must respect tier semantics:

* **Tier 1**: May synthesize across multiple sources, but every factual statement must cite `[doc_id, page, snippet]`.
* **Tier 2**: Must be **excerpt-first**. Prefer quoting/snippets + citation. Avoid broad synthesis and speculation.
* **Tier 3**: Quote/cite only. If the request would require interpretation of messy attachments, the system should refuse or ask the user to narrow scope.

---

### D) Required Audit Artifact + DONE Checks

Add a new required report:

**`reports/tier-policy-audit.md`**
Must include:

* counts of docs by tier
* sample doc IDs per tier
* confirmation:

  * Tier 1 bypassed PII
  * Tier 2/3 ran PII redaction before embeddings
* PII summary for Tier 2 and Tier 3 (counts/types)
* count of docs flagged `do_not_embed=true`

**Update DONE logic (check_done.py) to verify:**

* 100% of ingested docs have a tier
* 100% of chunks in vector store contain `tier` metadata
* Tier 2/3 chunks are marked as `redacted=true` (or equivalent) and Tier 1 are `redacted=false`
* `tier-policy-audit.md` exists and references the latest run

---

# Fully rewritten Ralph plan (nonstop + tier-aware)

## 0) Ralph Run Contract

**Mission:** Build a local Phase 0 RAG prototype + eval harness that yields a binary decision (**GO / GO* / NO-GO**) using mechanical verification.

**Single DONE gate:** `python scripts/check_done.py` exits `0`.

**Nonstop mode:** Continuous iterations until DONE or hard stop triggers.

---

## 1) Scope (Phase 0)

### In scope

* PDF ingestion (OCR → tiering → tier-appropriate PII → chunk → embed → ChromaDB)
* Query pipeline (retrieve → rerank → answer w/ citations + tier rules)
* Eval harness (claim-level scoring + unanswerables + JSON reports)
* Streamlit prototype UI

### Out of scope

* Production infra, auth, CI/CD, deployment, large-scale scraping.

**Rule:** If it doesn’t measurably improve eval toward DONE, it’s out of scope.

---

## 2) Hard Stops (Safety + Convergence)

Stop immediately if:

* `check_done.py` returns `0`
* runtime exceeds `T` (default: 12h)
* spend exceeds `$B` (default: $20)
* no improvement after `N` iterations (default: 6)
* agent proposes blocked actions (deploy, secrets rotation, destructive ops, broad network access)

---

## 3) Inputs

### Documents

* Minimum: 20 PDFs total
* Provenance: real vs synthetic tracked in `data/raw/.../manifest.json`

### Eval set

* Minimum: 30 Q/A
* Includes `expected_claims` and ≥3 unanswerables
* Eval set becomes immutable after baseline is created

---

## 4) Tiering and Policies (Mandatory)

### 4.1 Tiering policy

(Insert **Document Tiering Policy** section A)

### 4.2 Tier-specific processing

(Insert **Tier-Specific Processing Pipeline** section B)

### 4.3 Tier-specific answering

(Insert **Tier-Specific Answer Policy** section C)

---

## 5) Required command surface (one happy path)

From a fresh clone (only `.env` edits):

* `make setup`
* `make ingest`
* `make eval`
* `make ui`
* `make check-done`
* `make iterate` (recommended): clean → ingest → eval

**Rule:** The agent must not invent a custom run sequence.

---

## 6) Required artifacts

### Core

* `reports/eval-latest.json` + `reports/eval-latest.md`
* `reports/eval-baseline.json`
* `reports/eval-best.json`
* `reports/iteration-log.md`
* `docs/architecture-phase0.md`
* `prototype/app.py` (Streamlit)
* prompt files (Tier 1, Tier 2, eval/judge)

### Benchmarks

* `reports/ocr-benchmark.md`
* `reports/presidio-benchmark.md`
* `reports/chunking-benchmark.md`

### New required audit

* `reports/tier-policy-audit.md` (see Section D)

---

## 7) Eval metrics and decision policy

**Quality thresholds (example; keep your current numbers):**

* recall ≥ 70%
* citation coverage ≥ 80%
* hallucination ≤ 5%

**Decisions**

* **GO***: thresholds met on full eval set
* **GO**: thresholds met on real-data subset (when real eval exists)
* **NO-GO**: thresholds not met after iteration limit / hard stop

---

## 8) Nonstop Ralph Loop (repeat until stop)

### 8.1 Baseline

* Run `make ingest && make eval`
* Save as `reports/eval-baseline.json`

### 8.2 Allowed changes (Phase 0 only)

* chunking strategy/params
* retrieval params (top_k, filters)
* rerank params (rerank_top_n)
* Tier 1/Tier 2 prompts, eval prompts
* Tier 2/3 PII thresholds/detectors
* bug fixes in ingestion/citation metadata propagation

**Not allowed:** architecture rewrites, stack swaps, new features.

### 8.3 Iteration procedure (mandatory)

Each iteration:

1. Identify weakest failing metric
2. Hypothesize cause (1–3 bullets)
3. Make **one** atomic change
4. Run `make iterate`
5. Compare to baseline + best

   * regressions on gating metrics → revert immediately
   * improvements without unacceptable regressions → keep, update `eval-best.json`
6. Append to `reports/iteration-log.md`
7. Run `make check-done`

### 8.4 Checkpoints

Every 3 iterations or 90 minutes:

* write `reports/checkpoint.md` with best metrics, remaining failures, next plan
* ensure `tier-policy-audit.md` still matches the current run

If the checkpoint can’t be produced coherently → stop.

---

## 9) Finalization

When done or stopped:

* write a final summary (GO/GO*/NO-GO)
* list what improved and what remains weak
* ensure all required artifacts exist and are consistent
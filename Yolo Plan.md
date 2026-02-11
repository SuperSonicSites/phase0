## 1. Executive Summary

**Objective:** Build a local RAG prototype + evaluation harness + simple chat UI in 3 days. Prove that municipal documents can be ingested, chunked, embedded, retrieved, and answered with claim-level citations that meet quality thresholds. Produce a binary GO/NO-GO decision.

**What "DONE" means (binary):**

1. `make setup && make ingest && make eval && make ui` succeeds from a fresh clone with only `.env` edits.
2. `reports/eval-latest.json` passes all GO* thresholds (recall >= 70%, citation coverage >= 80%, hallucination <= 5%).
3. If real Tofino docs are present: same thresholds must also pass on the real-data subset for full GO.
4. Streamlit UI accepts a question and returns a cited answer.

---

## 2. Scope & Out of Scope

### In Scope (Phase 0)

- Local Python prototype (no cloud infra except LLM/embedding API calls)
- PDF ingestion: OCR (Marker), PII redaction (Presidio), chunking, embedding, vector storage
- Local vector store (ChromaDB, persistent)
- Query pipeline: embed question -> retrieve -> rerank -> generate answer with citations
- Evaluation harness: gold Q/A set, automated claim-level scoring, JSON+Markdown report
- Streamlit chat UI
- OCR benchmark (>= 2 doc types)
- PII benchmark (>= 3 meeting minutes)
- Chunking strategy comparison (>= 2 strategies)
- Iteration loop: identify weakest metric -> change -> eval -> compare baseline -> keep/revert
- Baseline RAG eval report with GO*/GO determination

### Out of Scope (Phase 0)

- Production infrastructure (AWS, Supabase, Vercel, Step Functions, Capacitor)
- Authentication, RBAC, multi-tenancy, scrapers, CI/CD, DDL migrations
- Monitoring (Langfuse, PostHog, Sentry) — except local file logging
- Document versioning, deduplication at scale, network security, DPA templates

---

## 3. Non-Negotiable Constraints

| Constraint                           | Rule                                                                                                                                                                                                                     | Source               |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------- |
| **Citations required (claim-level)** | Every factual sentence in an answer MUST include a citation `[Source: {doc}, p.{page}, s.{section}]`. If the answer cannot be supported, respond exactly: "This information is not available in the provided documents." | README, ARCHITECTURE |
| **Tier behavior**                    | Tier 1 (bylaws, OCPs, budgets): full synthesis with citations. Tier 2 (meeting minutes): excerpt-only with document link, NO synthesis. Tier 3: excluded from index entirely.                                            | SCHEMA & DATA MODEL  |
| **PII before vectors**               | PII redaction occurs BEFORE embedding. Raw PII must NEVER enter the vector store. Verified by spot-check in PII benchmark.                                                                                               | SCHEMA & DATA MODEL  |
| **Determinism**                      | `temperature=0` for ALL LLM calls during eval runs. Config recorded in every eval report.                                                                                                                                | ARCHITECTURE         |
| **No legal opinions**                | System prompt enforces factual retrieval only. Never "Yes you can" / "No you can't."                                                                                                                                     | ARCHITECTURE         |
| **No secrets in repo**               | `.env` in `.gitignore`. `.env.example` committed with placeholders only.                                                                                                                                                 | Security posture     |
| **Pinned dependencies**              | `uv.lock` (or `requirements.txt` with `==` pins) committed. Fresh install uses locked versions.                                                                                                                          | Reproducibility      |

---

## 4. Inputs Inventory

### 4.1 Source Documents

|Input|Format|Min Count|Location|Access Tier|Fallback|
|---|---|---|---|---|---|
|Tofino bylaws (text-heavy)|PDF|5|`data/raw/tofino/bylaws/`|1|Generate synthetic with `scripts/generate_synthetic_docs.py`|
|Tofino zoning bylaws (tables)|PDF|3|`data/raw/tofino/zoning/`|1|Generate synthetic|
|Tofino meeting minutes (PII-rich)|PDF|5|`data/raw/tofino/minutes/`|2|Generate synthetic with fake PII|
|Tofino OCP sections|PDF|2|`data/raw/tofino/ocp/`|1|Generate synthetic|
|Tofino budgets (tabular)|PDF|2|`data/raw/tofino/budgets/`|1|Generate synthetic|

**Total:** >= 20 PDFs (mix of Tier 1 and Tier 2).

**Data classification:**

- **Real docs:** PDFs provided by Dan or downloaded from public municipal portals. Identified by `data/raw/tofino/.real_data_manifest.json` listing filenames and SHA-256 hashes.
- **Synthetic docs:** Generated by `scripts/generate_synthetic_docs.py`. Identified by `"synthetic": true` in manifest.

**How agent obtains inputs:**

1. Check `data/raw/tofino/` for existing PDFs.
2. If >= 20 PDFs and `.real_data_manifest.json` exists with >= 10 real entries -> use them.
3. If < 20 PDFs: run `make generate-docs` to fill to 20. Manifest marks these synthetic.
4. **Critical:** The eval report MUST tag each result with `data_source: "real" | "synthetic"` and report metrics separately for each subset.

### 4.2 Evaluation Dataset

|Input|Format|Location|Min Size|
|---|---|---|---|
|Gold Q/A pairs|JSONL|`data/eval/tofino-qa-pairs.jsonl`|30 pairs|

**Schema per line:**

json

````json
{
  "id": "q001",
  "question": "What is the maximum building height in Tofino's C-1 commercial zone?",
  "expected_answer": "The maximum building height in the C-1 zone is 10 metres.",
  "expected_claims": [
    "The maximum building height in the C-1 zone is 10 metres"
  ],
  "source_document": "tofino-zoning-bylaw-2020.pdf",
  "source_page": 14,
  "source_section": "Part 4 - Commercial Zones, Section 4.1.3",
  "tier": 1,
  "difficulty": "simple",
  "synthetic": false,
  "unanswerable": false
}
```

**Required fields:** `id`, `question`, `expected_answer`, `expected_claims`, `source_document`, `tier`, `synthetic`
**New field `expected_claims`:** List of atomic factual claims the answer must make. Used for claim-level faithfulness scoring (see Section 9).

**Eval set must include >= 3 unanswerable questions** (questions whose answer is NOT in the corpus) to test refusal behavior. Mark with `"unanswerable": true`.

**How agent obtains eval data:**
1. Check if `data/eval/tofino-qa-pairs.jsonl` exists with >= 30 entries. If yes, use it.
2. Fallback: Generate 30 synthetic Q/A pairs with `scripts/generate_eval_set.py`. Mark `"synthetic": true`. Include >= 3 unanswerable questions.

### 4.3 Directory Structure
```
data/
  raw/
    tofino/
      .real_data_manifest.json   # {"files": [{"name": "...", "sha256": "...", "synthetic": false}]}
      bylaws/
      zoning/
      minutes/
      ocp/
      budgets/
  processed/
    tofino/          # Marker OCR markdown output
  redacted/
    tofino/          # Post-Presidio redacted text
  eval/
    tofino-qa-pairs.jsonl
  chromadb/          # Persistent vector store
````

**Acceptance Check:**

bash

```bash
python -c "
import glob, json, pathlib
docs = glob.glob('data/raw/tofino/**/*.pdf', recursive=True)
assert len(docs) >= 20, f'Only {len(docs)} docs, need 20'
evals = [json.loads(l) for l in open('data/eval/tofino-qa-pairs.jsonl')]
assert len(evals) >= 30, f'Only {len(evals)} eval pairs, need 30'
claims = [e for e in evals if 'expected_claims' in e and len(e['expected_claims']) > 0]
assert len(claims) >= 25, f'Only {len(claims)} evals have expected_claims'
unans = [e for e in evals if e.get('unanswerable')]
assert len(unans) >= 3, f'Only {len(unans)} unanswerable questions, need 3'
print(f'OK: {len(docs)} docs, {len(evals)} eval pairs, {len(unans)} unanswerable')
"
```

---

## 5. External APIs & Dependencies

### 5.1 LLM — Anthropic Claude

|Field|Value|
|---|---|
|**Purpose**|Answer generation + faithfulness judging + eval set generation|
|**SDK**|`anthropic>=0.42.0,<1.0`|
|**Model (generation)**|`claude-sonnet-4-5-20250929`|
|**Model (judge)**|`claude-haiku-4-5-20251001` (cheaper for faithfulness checks)|
|**Auth**|`ANTHROPIC_API_KEY` env var|
|**Cost cap**|Track cumulative tokens. STOP eval if projected cost > $20. Log partial results.|
|**Retry**|3 retries, exponential backoff (1s, 2s, 4s), on 429/500/503|
|**Fallback**|If API unavailable after retries: attempt Ollama `llama3.1:8b` locally. Document quality difference.|
|**Validation**|`python -c "from anthropic import Anthropic; c=Anthropic(); r=c.messages.create(model='claude-haiku-4-5-20251001',max_tokens=10,messages=[{'role':'user','content':'hi'}]); print('OK')"`|

### 5.2 Embeddings — Cohere

|Field|Value|
|---|---|
|**Purpose**|Document chunk + query embedding|
|**SDK**|`cohere>=5.0,<6.0`|
|**Model**|`embed-english-v3.0` (1024 dims)|
|**Input types**|`search_document` for chunks, `search_query` for queries|
|**Auth**|`COHERE_API_KEY` env var|
|**Retry**|3 retries, exponential backoff|
|**Fallback**|`sentence-transformers` with `all-MiniLM-L6-v2` (384 dims). Document limitation.|
|**Validation**|`python -c "import cohere; c=cohere.ClientV2(); r=c.embed(texts=['test'], model='embed-english-v3.0', input_type='search_query', embedding_types=['float']); print(f'OK: {len(r.embeddings.float_[0])} dims')"`|

### 5.3 Reranker — Cohere

|Field|Value|
|---|---|
|**Purpose**|Rerank retrieved chunks|
|**Model**|`rerank-english-v3.0`|
|**Auth**|Same `COHERE_API_KEY`|
|**Fallback**|Skip reranking, use cosine similarity order. Document quality delta.|
|**Validation**|`python -c "import cohere; c=cohere.ClientV2(); r=c.rerank(query='test', documents=['a','b'], model='rerank-english-v3.0'); print(f'OK: {len(r.results)}')"`|

### 5.4 OCR — Marker (local)

|Field|Value|
|---|---|
|**Package**|`marker-pdf>=1.0`|
|**Fallback**|`pymupdf4llm` (text extraction only, no OCR). Document limitation for scanned docs.|
|**Validation**|`python -c "from marker.converters.pdf import PdfConverter; print('OK')"`|

### 5.5 PII — Presidio (local)

|Field|Value|
|---|---|
|**Packages**|`presidio-analyzer>=2.2`, `presidio-anonymizer>=2.2`|
|**NLP**|`en_core_web_lg` (spaCy)|
|**Fallback**|Regex-based PII detection for PERSON, EMAIL, PHONE. Document limitation.|
|**Validation**|`python -c "from presidio_analyzer import AnalyzerEngine; a=AnalyzerEngine(); r=a.analyze(text='John Smith 123 Main St', language='en'); print(f'OK: {len(r)} entities')"`|

### 5.6 Vector Store — ChromaDB (local)

|Field|Value|
|---|---|
|**Package**|`chromadb>=0.5,<1.0`|
|**Persistence**|`data/chromadb/`|
|**Fallback**|N/A (pure Python)|
|**Validation**|`python -c "import chromadb; c=chromadb.PersistentClient(path='data/chromadb'); print('OK')"`|

---

## 6. Credentials & Config Table

|ENV VAR|Required|Placeholder Format|Used By|Validation|
|---|---|---|---|---|
|`ANTHROPIC_API_KEY`|Yes|`sk-ant-api03-xxxx`|`generate.py`, `eval.py`|`python -c "import os; assert os.environ.get('ANTHROPIC_API_KEY','').startswith('sk-ant')"`|
|`COHERE_API_KEY`|Yes|`xxxxxxxxxxxxxxxx`|`ingest.py`, `retrieval.py`|`python -c "import os; assert len(os.environ.get('COHERE_API_KEY',''))>10"`|
|`CHROMA_PERSIST_DIR`|No|`data/chromadb`|`ingest.py`, `retrieval.py`|defaults to `data/chromadb`|
|`LLM_MODEL`|No|`claude-sonnet-4-5-20250929`|`generate.py`|defaults|
|`JUDGE_MODEL`|No|`claude-haiku-4-5-20251001`|`eval.py`|defaults|
|`EMBEDDING_MODEL`|No|`embed-english-v3.0`|`ingest.py`|defaults|
|`RERANK_MODEL`|No|`rerank-english-v3.0`|`retrieval.py`|defaults|
|`CHUNK_STRATEGY`|No|`markdown`|`ingest.py`|one of: `fixed`, `markdown`, `sentence`|
|`CHUNK_SIZE`|No|`512`|`ingest.py`|int, tokens|
|`CHUNK_OVERLAP`|No|`50`|`ingest.py`|int, tokens|
|`LOG_LEVEL`|No|`INFO`|all modules|defaults|
|`MAX_COST_USD`|No|`20`|`eval.py`|float, stop if exceeded|

**Rules:**

- `.env` MUST be in `.gitignore`. Verified: `grep -q "^\.env$" .gitignore`
- `.env.example` MUST be committed with ALL vars above using placeholder values.
- `uv.lock` MUST be committed for reproducible installs.
- `python --version` must be >= 3.11. Specified in `pyproject.toml` `requires-python`.

**Fresh install verification:**

bash

````bash
git clone <repo> && cd govpal-phase0
cp .env.example .env  # fill real keys
uv sync --frozen  # uses uv.lock, fails if lock is stale
python scripts/validate_env.py
```

---

## 7. Repository Layout & Deliverables

### 7.1 Structure
```
govpal-phase0/
├── .env.example
├── .gitignore
├── pyproject.toml             # requires-python = ">=3.11"
├── uv.lock                    # pinned dependency lock
├── Makefile
├── README.md
├── prototype/
│   ├── __init__.py
│   ├── config.py              # Centralized config from env vars
│   ├── ingest.py
│   ├── chunking.py
│   ├── pii.py
│   ├── retrieval.py
│   ├── generate.py
│   ├── query.py
│   ├── eval.py
│   └── app.py                 # Streamlit
├── prompts/
│   ├── system_prompt_tier1.txt
│   ├── system_prompt_tier2.txt
│   └── faithfulness_judge.txt
├── scripts/
│   ├── generate_synthetic_docs.py
│   ├── generate_eval_set.py
│   ├── validate_env.py
│   └── check_done.py          # Definition of Done checker
├── data/                      # gitignored (except eval/)
│   ├── raw/tofino/{bylaws,zoning,minutes,ocp,budgets}/
│   ├── processed/tofino/
│   ├── redacted/tofino/
│   ├── eval/tofino-qa-pairs.jsonl
│   └── chromadb/
├── reports/
│   ├── eval-{timestamp}.json
│   ├── eval-{timestamp}.md
│   ├── eval-latest.json       # copy of most recent
│   ├── eval-baseline.json     # first passing eval (for regression)
│   ├── ocr-benchmark.md
│   ├── presidio-benchmark.md
│   └── chunking-benchmark.md
└── docs/
    └── architecture-phase0.md
````

### 7.2 Makefile

makefile

```makefile
.PHONY: setup ingest eval ui validate clean benchmark-all generate-docs generate-eval iterate check-done

setup:
	uv sync --frozen
	python -m spacy download en_core_web_lg
	python scripts/validate_env.py

ingest:
	python -m prototype.ingest --input-dir data/raw/tofino --collection govpal_chunks

eval:
	python -m prototype.eval --eval-set data/eval/tofino-qa-pairs.jsonl --output-dir reports/

ui:
	streamlit run prototype/app.py

validate:
	python scripts/validate_env.py

clean:
	python -c "import shutil,pathlib; [shutil.rmtree(p,True) for p in [pathlib.Path('data/chromadb'),pathlib.Path('data/processed'),pathlib.Path('data/redacted')]]"

benchmark-all:
	python -m prototype.ingest --benchmark-ocr --output reports/ocr-benchmark.md
	python -m prototype.pii --benchmark --input-dir data/raw/tofino/minutes --output reports/presidio-benchmark.md
	python -m prototype.chunking --benchmark --eval-set data/eval/tofino-qa-pairs.jsonl --output reports/chunking-benchmark.md

generate-docs:
	python scripts/generate_synthetic_docs.py --output-dir data/raw/tofino --num-docs 20

generate-eval:
	python scripts/generate_eval_set.py --collection govpal_chunks --output data/eval/tofino-qa-pairs.jsonl --num-pairs 30

# Iteration: re-ingest with current config, re-eval against baseline
iterate:
	$(MAKE) clean
	$(MAKE) ingest
	python -m prototype.eval --eval-set data/eval/tofino-qa-pairs.jsonl --output-dir reports/ --baseline reports/eval-baseline.json

check-done:
	python scripts/check_done.py

# Full pipeline
all: setup
	@python -c "import glob; n=len(glob.glob('data/raw/tofino/**/*.pdf',recursive=True)); exit(0 if n>=20 else 1)" 2>/dev/null || $(MAKE) generate-docs
	$(MAKE) ingest
	@test -f data/eval/tofino-qa-pairs.jsonl || $(MAKE) generate-eval
	$(MAKE) eval
	$(MAKE) check-done
```

### 7.3 Deliverable Acceptance Criteria

|Deliverable|Pass/Fail Check|Command|
|---|---|---|
|Repo structure|All dirs exist|`python -c "from pathlib import Path; dirs=['prototype','prompts','scripts','data/eval','reports','docs']; assert all(Path(d).is_dir() for d in dirs)"`|
|`.env.example`|Has >= 2 API_KEY entries|`python -c "assert open('.env.example').read().count('API_KEY')>=2"`|
|`.gitignore`|Contains `.env`|`python -c "assert '.env\n' in open('.gitignore').read() or '.env' in open('.gitignore').read().splitlines()"`|
|`uv.lock`|Exists and `uv sync --frozen` works|`uv sync --frozen` exits 0|
|`ingest.py`|Ingests 1 PDF, produces > 0 chunks|`python -c "from prototype.ingest import ingest_document; r=ingest_document(next(iter(__import__('glob').glob('data/raw/tofino/**/*.pdf',recursive=True)))); assert r.success and r.num_chunks>0"`|
|`query.py`|Returns answer with citations or refusal|`python -m prototype.query "What is Tofino's zoning?"` prints answer|
|`eval.py`|Produces JSON+MD in reports/|`make eval` creates `reports/eval-latest.json`|
|`app.py`|Streamlit starts|`timeout 15 streamlit run prototype/app.py --server.headless true`|
|Eval metrics|Thresholds met|`python scripts/check_done.py` exits 0|
|Benchmarks|3 reports exist|`python -c "from pathlib import Path; assert all(Path(f'reports/{n}').exists() for n in ['ocr-benchmark.md','presidio-benchmark.md','chunking-benchmark.md'])"`|
|Prompts|Tier 1 + Tier 2 exist|`python -c "from pathlib import Path; assert Path('prompts/system_prompt_tier1.txt').exists() and Path('prompts/system_prompt_tier2.txt').exists()"`|

---

## 8. Internal API Contracts

### 8.1 Ingestion

python

```python
# prototype/ingest.py
from dataclasses import dataclass, field

@dataclass
class ChunkMetadata:
    document_id: str          # SHA-256 of source PDF
    document_title: str
    document_path: str
    chunk_index: int
    page_number: int | None
    section_header: str | None
    access_tier: int          # 1, 2, or 3
    char_start: int
    char_end: int
    is_synthetic: bool        # True if source doc is synthetic

@dataclass
class IngestResult:
    document_id: str
    document_path: str
    num_chunks: int
    num_pii_redactions: int
    ocr_char_count: int
    success: bool
    error: str | None = None

def ingest_document(pdf_path: str, access_tier: int = 1, collection_name: str = "govpal_chunks") -> IngestResult: ...
def ingest_directory(directory: str, access_tier: int = 1, collection_name: str = "govpal_chunks") -> list[IngestResult]: ...
```

### 8.2 Chunking

python

```python
# prototype/chunking.py
@dataclass
class Chunk:
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    page_number: int | None
    section_header: str | None

def chunk_text(text: str, strategy: str = "fixed", chunk_size: int = 512, chunk_overlap: int = 50) -> list[Chunk]: ...
# strategy: "fixed" | "markdown" | "sentence"
```

### 8.3 PII Redaction

python

```python
# prototype/pii.py
@dataclass
class RedactionResult:
    redacted_text: str
    num_redactions: int
    entities_found: list[dict]  # [{"type": "PERSON", "text": "John Smith", "score": 0.95}]

def redact_pii(text: str, confidence_threshold: float = 0.5, entities: list[str] | None = None) -> RedactionResult: ...
```

### 8.4 Retrieval + Rerank

python

```python
# prototype/retrieval.py
@dataclass
class RetrievedChunk:
    text: str
    metadata: ChunkMetadata
    similarity_score: float
    rerank_score: float | None

def retrieve(query: str, top_k: int = 10, rerank: bool = True, rerank_top_n: int = 5,
             collection_name: str = "govpal_chunks", tier_filter: list[int] | None = None) -> list[RetrievedChunk]: ...
```

### 8.5 Answer Generation

python

````python
# prototype/generate.py
@dataclass
class Citation:
    document_title: str
    page_number: int | None
    section_header: str | None
    chunk_text_excerpt: str   # First 200 chars of cited chunk (for faithfulness verification)

@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    refused: bool
    tier_behavior: str        # "synthesis" | "link_only"

def generate_answer(query: str, chunks: list[RetrievedChunk], tier: int = 1,
                    temperature: float = 0.0, model: str | None = None) -> AnswerResult: ...
```

**System prompt (Tier 1) — `prompts/system_prompt_tier1.txt`:**
```
You are GovPal AI, a municipal research assistant for Canadian elected officials.

RULES:
1. Answer ONLY from the provided source documents. Do not use outside knowledge.
2. Every factual SENTENCE must include a citation: [Source: {document_title}, p.{page}, s.{section}]
3. If the answer is not in the provided documents, respond EXACTLY: "This information is not available in the provided documents."
4. Never provide legal opinions. Say "According to [document], the regulation states..." not "You can/cannot."
5. Be concise. Councillors are busy.
6. If multiple sources support a claim, cite all of them.

SOURCE DOCUMENTS:
{chunks_with_metadata}

QUESTION: {query}
```

**System prompt (Tier 2) — `prompts/system_prompt_tier2.txt`:**
```
You are GovPal AI. The following documents are meeting minutes (semi-public records).

RULES:
1. Do NOT synthesize or summarize the content of meeting minutes.
2. Instead, provide the document title, page number, and a brief excerpt (max 2 sentences) directing the user to the relevant section.
3. Format: "See [document_title], page {page}: '{brief excerpt}...'"
4. If nothing relevant is found, respond: "No relevant meeting minutes found for this query."

DOCUMENTS:
{chunks_with_metadata}

QUESTION: {query}
````

### 8.6 Query Pipeline

python

```python
# prototype/query.py
@dataclass
class QueryResult:
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    model: str
    total_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    input_tokens: int
    output_tokens: int
    refused: bool

def query(question: str, top_k: int = 10, rerank_top_n: int = 5,
          tier: int = 1, temperature: float = 0.0) -> QueryResult: ...
```

### 8.7 Evaluation Runner

python

```python
# prototype/eval.py
@dataclass
class ClaimVerdict:
    claim: str                  # The expected claim
    supported: bool             # Was it found in the answer + supported by chunks?
    supporting_chunk_excerpt: str | None  # Chunk text that supports this claim (max 200 chars)
    citation_present: bool      # Did the answer cite a source for this claim?
    cited_correct_doc: bool     # Did the citation reference the expected source_document?

@dataclass
class EvalResult:
    question_id: str
    question: str
    expected_answer: str
    actual_answer: str
    expected_source: str
    retrieved_sources: list[str]
    correct_retrieval: bool
    claim_verdicts: list[ClaimVerdict]  # Per-claim breakdown
    citation_coverage: float    # fraction of claims with correct citation
    faithful: bool
    hallucinated: bool
    refused: bool
    expected_refused: bool      # True if unanswerable question
    latency_ms: float
    data_source: str            # "real" | "synthetic"

@dataclass
class EvalMetrics:
    retrieval_recall_at_5: float
    retrieval_recall_at_10: float
    citation_coverage: float       # avg fraction of claims with correct doc citation per answer
    claim_faithfulness: float      # fraction of claims supported by retrieved chunks
    hallucination_rate: float
    refusal_accuracy: float
    mean_latency_ms: float
    p95_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    # Split metrics
    real_data_recall_at_10: float | None   # None if no real data
    real_data_citation_coverage: float | None
    real_data_hallucination_rate: float | None

@dataclass
class EvalReport:
    metrics: EvalMetrics
    results: list[EvalResult]
    config: dict
    timestamp: str
    eval_set_path: str
    num_questions: int
    num_real: int               # count of real-data questions
    num_synthetic: int          # count of synthetic questions
    go_star: bool               # pipeline mechanics pass (all data)
    go: bool                    # product viability pass (real data only, or N/A if no real data)
    regression: dict | None     # comparison to baseline if provided

def run_eval(eval_path: str = "data/eval/tofino-qa-pairs.jsonl",
             output_dir: str = "reports/",
             baseline_path: str | None = None) -> EvalReport: ...
```

---

## 9. Evaluation Harness (Hard to Game)

### 9.1 Claim-Level Scoring

**Why:** A single correct citation attached to an otherwise hallucinated answer must NOT pass. Scoring is per-claim, not per-answer.

**Process for each eval question:**

1. Run `query()` to get answer + citations + retrieved chunks.
2. **Retrieval recall:** Check if `source_document` appears in top-10 retrieved chunk metadata.
3. **For each claim in `expected_claims`:** a. **Claim presence:** Use LLM judge (Haiku) to determine if the claim's meaning is present in the actual answer. Prompt: "Does the following answer contain the claim '{claim}'? Answer YES or NO." b. **Faithfulness:** Use LLM judge to check if the claim is supported by the retrieved chunks. Prompt: "Given these source chunks, is the claim '{claim}' supported by the text? Quote the supporting excerpt (max 200 chars) or say UNSUPPORTED." c. **Citation correctness:** Parse the citation attached to the sentence containing this claim. Check if the cited `document_title` matches `source_document` (fuzzy match: filename stem).
4. **Hallucination check:** Use LLM judge on the FULL answer: "List any factual claims in this answer that are NOT supported by the source chunks. If all claims are supported, say NONE." Count unsupported claims.
5. **Refusal check:** For unanswerable questions, verify `refused=True`.

### 9.2 Metrics & Thresholds

|Metric|Definition|GO* Threshold|GO Threshold (real data only)|
|---|---|---|---|
|`retrieval_recall_at_10`|% questions where expected doc in top-10|>= 70%|>= 70%|
|`citation_coverage`|avg(claims_with_correct_citation / total_claims) per answer|>= 80%|>= 80%|
|`claim_faithfulness`|avg(supported_claims / total_claims) per answer|>= 95%|>= 95%|
|`hallucination_rate`|% of answers with >= 1 unsupported claim|<= 5%|<= 5%|
|`refusal_accuracy`|% of unanswerable questions correctly refused|>= 66% (2/3)|>= 66%|
|`p95_latency_ms`|95th percentile end-to-end|<= 10s (informational)|<= 10s (informational)|

### 9.3 Report Output

**JSON:** `reports/eval-{timestamp}.json` with full `EvalReport` serialized (including per-question `claim_verdicts`).

**Markdown:** `reports/eval-{timestamp}.md`:

markdown

````markdown
# GovPal Phase 0 Eval Report

**Date:** {timestamp}
**Questions:** {num_questions} ({num_real} real, {num_synthetic} synthetic)
**Config:** {config summary}

## Overall Metrics (All Data) — GO* Check
| Metric | Value | Threshold | Pass |
|---|---|---|---|
| Retrieval Recall@10 | X% | >= 70% | PASS/FAIL |
| Citation Coverage | X% | >= 80% | PASS/FAIL |
| Hallucination Rate | X% | <= 5% | PASS/FAIL |
| Refusal Accuracy | X% | >= 66% | PASS/FAIL |

**GO*: {PASS/FAIL}**

## Real Data Metrics — GO Check
| Metric | Value | Threshold | Pass |
|---|---|---|---|
(same metrics, computed only on real-data questions)

**GO: {PASS/FAIL/N_A (no real data)}**

## Regression (vs. baseline)
| Metric | Baseline | Current | Delta | Status |
(flag if regressed > 5pp)

## Failure Analysis
(per-question breakdown of failures with claim verdicts)

## Cost
Estimated: ${cost}
```

**Latest symlink:** After each eval, copy to `reports/eval-latest.json`. After first passing GO\*, copy to `reports/eval-baseline.json`.

### 9.4 Anti-Gaming Rules

1. **Claim-level granularity:** Citation coverage is per-claim, not per-answer. One citation cannot cover multiple claims.
2. **Faithfulness requires chunk excerpts:** The judge must quote the supporting chunk text. "Supported" without an excerpt is scored as UNSUPPORTED.
3. **Real-data gate:** Full GO requires thresholds met on real-data subset. Synthetic-only can achieve GO\* (pipeline validated) but not GO (product viable).
4. **Unanswerable questions:** Must be included. Refusal accuracy is gated. An agent cannot remove unanswerable questions from the eval set.
5. **Eval set immutability:** Once the eval set is created, it MUST NOT be modified during iteration. Changes to the eval set require re-running from baseline.

---

## 10. Iteration & Regression Policy

### 10.1 Iteration Loop

After the first eval run (baseline), the agent MAY iterate to improve metrics:
```
LOOP (max 5 iterations):
  1. IDENTIFY weakest metric (lowest margin above threshold, or failing)
  2. HYPOTHESIZE root cause (log in reports/iteration-log.md)
  3. IMPLEMENT one change:
     - Chunking: adjust strategy/size/overlap in config
     - Prompts: edit system prompt files
     - Retrieval: adjust top_k, rerank_top_n
     - PII: adjust confidence threshold
  4. RUN: make iterate (clean -> ingest -> eval with baseline comparison)
  5. COMPARE to baseline:
     - If ALL thresholds still met AND weakest metric improved: KEEP. Update baseline.
     - If any threshold REGRESSED below minimum: REVERT change. Log reason.
     - If no improvement but no regression: KEEP only if justified. Log reason.
  6. REPEAT or STOP
````

### 10.2 Revert Policy

- **Hard revert:** If any gating metric drops below threshold after a change -> revert immediately.
- **Soft revert:** If the target metric didn't improve and another metric dropped > 3pp -> revert.
- **Revert mechanism:** Git. Each iteration is a commit. `git revert HEAD` to undo.

### 10.3 Stop Conditions

The iteration loop STOPS when ANY of:

1. All GO* thresholds pass (and GO thresholds pass if real data exists) -> **DONE**
2. 5 iterations exhausted without meeting thresholds -> **NO-GO**, log failure analysis
3. Cost cap ($20) reached -> **STOP**, report partial results

### 10.4 Iteration Log

File: `reports/iteration-log.md`

markdown

```markdown
| Iter | Change | Target Metric | Before | After | Kept? | Reason |
|---|---|---|---|---|---|---|
| 0 | baseline | - | - | recall=65% | - | Initial run |
| 1 | chunk_size 512->384 | recall@10 | 65% | 72% | Yes | Improved recall, no regression |
| 2 | add few-shot to prompt | citation_coverage | 75% | 83% | Yes | Improved citation |
```

---

## 11. Day 1-3 Execution Plan

### Day 1: Pipeline Setup

|Time|Task|Deliverable|Checkpoint|If Blocked|
|---|---|---|---|---|
|0:00-0:30|Scaffold repo (dirs, pyproject.toml, .env.example, .gitignore, Makefile, README)|Repo skeleton|`ls prototype/ scripts/ prompts/`|N/A|
|0:30-1:30|Install deps (`uv sync`), spaCy model, generate `uv.lock`|Working env|`uv sync --frozen && python scripts/validate_env.py`|Marker fails -> use pymupdf4llm. Presidio fails -> use regex.|
|1:30-2:00|Validate API keys (Anthropic + Cohere live calls)|APIs confirmed|`validate_env.py` passes|Anthropic down -> configure Ollama. Cohere down -> sentence-transformers.|
|2:00-3:00|Check/generate source docs (>= 20 PDFs)|Data ready|File count >= 20|`make generate-docs`|
|3:00-4:30|Build `ingest.py`: PDF -> Marker OCR -> Presidio PII -> chunking -> Cohere embed -> ChromaDB|`ingest_document()` works|Ingest 1 PDF successfully|Skip failing PDFs, log warning.|
|4:30-5:30|Build `pii.py`: Presidio wrapper with configurable thresholds|`redact_pii()` works|Detects PII in test string|Regex fallback.|
|5:30-6:30|Build `chunking.py`: 3 strategies (fixed, markdown, sentence)|All 3 work|Unit test each strategy|N/A (pure Python).|
|6:30-8:00|Ingest all docs. Build `config.py`.|>= 15 docs in ChromaDB|`make ingest` completes|Skip failures. Need >= 15.|

**Day 1 Gate:** `make validate` passes. >= 15 docs ingested.

### Day 2: Query Pipeline + Benchmarks

|Time|Task|Deliverable|Checkpoint|If Blocked|
|---|---|---|---|---|
|0:00-1:30|Build `retrieval.py`: embed -> ChromaDB search -> Cohere rerank|`retrieve()` returns chunks|Test with 5 queries|Skip rerank if Cohere fails.|
|1:30-3:00|Build `generate.py` + prompts (Tier 1, Tier 2, judge)|`generate_answer()` returns cited answers|5 test questions produce citations|Anthropic down -> block (LLM required).|
|3:00-4:00|Build `query.py`: wire retrieval + generation|End-to-end works|`python -m prototype.query "test"`|N/A|
|4:00-5:30|OCR benchmark (>= 2 doc types) + PII benchmark (>= 3 minutes)|`reports/ocr-benchmark.md`, `reports/presidio-benchmark.md`|Reports exist|Use synthetic docs for benchmark.|
|5:30-8:00|Chunking benchmark (>= 2 strategies), select winner|`reports/chunking-benchmark.md`|Report with comparison table|Test fixed + markdown only if time-constrained.|

**Day 2 Gate:** `python -m prototype.query "test question"` returns cited answer. All 3 benchmark reports exist.

### Day 3: Evaluation + UI + GO/NO-GO

|Time|Task|Deliverable|Checkpoint|If Blocked|
|---|---|---|---|---|
|0:00-1:00|Check/generate eval set (>= 30 pairs, >= 3 unanswerable)|Eval set ready|JSONL passes validation|`make generate-eval`|
|1:00-3:00|Build `eval.py`: claim-level scoring, LLM judge, JSON+MD output|`run_eval()` works|`make eval` produces reports|String overlap fallback if judge too expensive.|
|3:00-4:00|Re-ingest with best chunking strategy from benchmark|Updated ChromaDB|`make clean && make ingest`|Use fixed 512 if no clear winner.|
|4:00-5:00|Run baseline eval. Save as `eval-baseline.json`.|Baseline report|File exists|Skip failing questions (need >= 25 successful).|
|5:00-5:30|**Iteration loop** (if needed): up to 2 quick iterations|Improved metrics|Metrics improve or hold|Stop after 2 iterations on Day 3.|
|5:30-7:00|Build `app.py` (Streamlit chat UI) + `check_done.py`|Working UI + done checker|`make ui` works, `make check-done` reports status|CLI REPL fallback.|
|7:00-7:30|Write `docs/architecture-phase0.md`|Architecture doc|File exists|N/A|
|7:30-8:00|Final `make check-done`. Record GO*/GO/NO-GO.|Decision recorded|Exit code 0 or 1|If NO-GO: log root causes + iteration recommendations.|

**Day 3 Gate:** `make check-done` exits 0.

---

## 12. GO / NO-GO Criteria

### Two-Tier Decision

**GO* (Pipeline Mechanics):** Thresholds met on ALL eval data (real + synthetic). Proves the pipeline works.

**GO (Product Viability):** Thresholds met on REAL-DATA subset only. Proves the approach works on actual municipal documents.

|Decision|Condition|Action|
|---|---|---|
|**GO**|GO* passes AND real_data metrics all pass (>= 5 real-data eval questions)|Proceed to Phase 1 with confidence|
|**GO***|GO* passes BUT no real data available (or < 5 real eval questions)|Pipeline validated. Block Phase 1 until real data tested. Dan must provide docs.|
|**NO-GO**|GO* fails after 5 iterations|Document failures. Recommend: alt chunking, alt embeddings, alt LLM. Schedule Day 4 sprint.|

### Verification Command

bash

```bash
python -c "
import json, sys
r = json.load(open('reports/eval-latest.json'))
m = r['metrics']

# GO* check (all data)
go_star = (
    m['retrieval_recall_at_10'] >= 0.70 and
    m['citation_coverage'] >= 0.80 and
    m['hallucination_rate'] <= 0.05 and
    m.get('refusal_accuracy', 1.0) >= 0.66
)

# GO check (real data only)
has_real = r['num_real'] >= 5
go = False
if has_real:
    go = (
        (m.get('real_data_recall_at_10') or 0) >= 0.70 and
        (m.get('real_data_citation_coverage') or 0) >= 0.80 and
        (m.get('real_data_hallucination_rate') or 1) <= 0.05
    )

print(f'GO*  (pipeline):  {\"PASS\" if go_star else \"FAIL\"}')
print(f'GO   (viability): {\"PASS\" if go else \"N/A (need real data)\" if not has_real else \"FAIL\"}')
print(f'  Recall@10:          {m[\"retrieval_recall_at_10\"]:.1%}')
print(f'  Citation Coverage:  {m[\"citation_coverage\"]:.1%}')
print(f'  Hallucination Rate: {m[\"hallucination_rate\"]:.1%}')
if has_real:
    print(f'  [Real] Recall@10:   {m.get(\"real_data_recall_at_10\",0):.1%}')
    print(f'  [Real] Citation:    {m.get(\"real_data_citation_coverage\",0):.1%}')
    print(f'  [Real] Hallucinate: {m.get(\"real_data_hallucination_rate\",1):.1%}')
sys.exit(0 if go_star else 1)
"
```

---

## 13. Risks & Mitigations

|Risk|Likelihood|Impact|Mitigation|Fallback|
|---|---|---|---|---|
|Real Tofino PDFs not available|High|Medium|Check `data/raw/` first|Synthetic docs -> GO* only (not full GO). Dan must provide for GO.|
|Gold Q/A set not available|High|Medium|Check `data/eval/` first|Generate synthetic. Tag as synthetic.|
|Marker OCR fails on install|Medium|Medium|Try Marker first|`pymupdf4llm`. Document scanned-doc limitation.|
|Marker OCR produces garbled text|Medium|High|OCR benchmark catches this|Switch to `pymupdf4llm` for affected doc types. Log in benchmark.|
|Presidio misses PII|Low|Critical|PII benchmark with known PII|Lower confidence threshold. 100% human review on minutes during pilot.|
|PII leaks into vector store|Low|Critical|Verify: query ChromaDB for known PII patterns post-ingest|Purge and re-ingest. Add post-ingest PII scan as quality gate.|
|Anthropic API unavailable|Low|High|Retry 3x with backoff|Ollama local. Lower quality.|
|Cohere API unavailable|Low|Medium|Retry 3x|sentence-transformers local. Skip rerank.|
|Citation parsing fails (LLM doesn't follow format)|Medium|Medium|Few-shot examples in prompt|Regex fallback parser for common citation patterns.|
|Model drift between eval runs|Low|Low|Pin model IDs, temperature=0|Record model in config. Detect if model changes.|
|Eval metrics gameable with synthetic data|Medium|High|GO requires real-data gate|GO* without real data explicitly marked as insufficient for Phase 1.|
|Claim-level judge is inaccurate|Medium|Medium|Spot-check 10% of judge verdicts manually|Log all judge responses for human review.|
|API costs exceed $20|Low|Low|Track cumulative tokens|Stop eval early, report partial.|
|Windows path issues|Medium|Low|`pathlib.Path` everywhere|Agent must never use string concatenation for paths.|

---

## 14. Assumptions Log

|#|Assumption|Basis|Impact if Wrong|
|---|---|---|---|
|A1|Phase 0 runs on Windows|Working dir is `D:\My notes\Supersonic\Projects`|Makefile uses cross-platform Python commands, not shell. Agent provides `make` AND direct `python` equivalents.|
|A2|Real Tofino PDFs may not exist yet|Dan listed as provider, no confirmation|Synthetic data used. GO* only (not full GO).|
|A3|Gold Q/A eval set may not exist|Dan + Renaud co-create it|Synthetic eval set. Lower confidence.|
|A4|`uv` is installed|ACTION PLAN specifies it|If not: `pip install uv` first, or use `pip install -r requirements.txt`.|
|A5|Python >= 3.11 available|Modern dev machine|If not: agent must document minimum version clearly.|
|A6|Claude Sonnet 4.5 for generation, Haiku 4.5 for judging|Cost optimization|Configurable via env vars.|
|A7|Cohere trial key sufficient for Phase 0|~20 docs + ~60 API calls total|Switch to sentence-transformers if limits hit.|
|A8|Tier 2 = excerpt-only, no synthesis|SCHEMA: "Link-only search, no summarization"|Prompt edit if interpretation wrong.|
|A9|Page numbers extractable from Marker output|Marker inserts page breaks in markdown|Chunk index fallback if not.|
|A10|Prototype repo is standalone (not in Obsidian vault)|Code shouldn't pollute notes|Created as `govpal-phase0/` adjacent to notes.|
|A11|LLM-as-judge for faithfulness costs ~$3 for 30 questions|Haiku pricing * ~60 judge calls|String overlap fallback if over budget.|
|A12|`expected_claims` can be auto-generated from `expected_answer` by splitting on sentence boundaries|Claim extraction is sentence-level|If answers have multi-claim sentences, judge quality degrades. Acceptable for Phase 0.|

---

## 15. Definition of Done

`scripts/check_done.py` implements this checklist. Exit code 0 = DONE, 1 = NOT DONE.

python

```python
#!/usr/bin/env python3
"""Phase 0 Definition of Done checker."""
import json, glob, os, sys
from pathlib import Path

checks = []

def check(name, condition):
    checks.append((name, condition))

# Structure
check("Repo dirs exist", all(Path(d).is_dir() for d in ['prototype','prompts','scripts','data','reports','docs']))
check(".env.example exists", Path('.env.example').is_file())
check(".env in .gitignore", '.env' in Path('.gitignore').read_text().splitlines() if Path('.gitignore').exists() else False)
check("uv.lock exists", Path('uv.lock').is_file())
check("Tier 1 prompt", Path('prompts/system_prompt_tier1.txt').is_file())
check("Tier 2 prompt", Path('prompts/system_prompt_tier2.txt').is_file())
check("README.md", Path('README.md').is_file())
check("Streamlit app", Path('prototype/app.py').is_file())

# Benchmarks
check("OCR benchmark", Path('reports/ocr-benchmark.md').is_file())
check("PII benchmark", Path('reports/presidio-benchmark.md').is_file())
check("Chunking benchmark", Path('reports/chunking-benchmark.md').is_file())

# Eval report
eval_file = Path('reports/eval-latest.json')
check("Eval report exists", eval_file.is_file())

if eval_file.is_file():
    r = json.loads(eval_file.read_text())
    m = r['metrics']
    check("Eval: >= 25 questions", r['num_questions'] >= 25)
    check("Eval: recall@10 >= 70%", m['retrieval_recall_at_10'] >= 0.70)
    check("Eval: citation_coverage >= 80%", m['citation_coverage'] >= 0.80)
    check("Eval: hallucination <= 5%", m['hallucination_rate'] <= 0.05)
    check("Eval: refusal_accuracy >= 66%", m.get('refusal_accuracy', 0) >= 0.66)

    # GO* determination
    go_star = (m['retrieval_recall_at_10'] >= 0.70 and
               m['citation_coverage'] >= 0.80 and
               m['hallucination_rate'] <= 0.05)
    check("GO* (pipeline)", go_star)

    # GO determination (real data)
    if r['num_real'] >= 5:
        go = ((m.get('real_data_recall_at_10') or 0) >= 0.70 and
              (m.get('real_data_citation_coverage') or 0) >= 0.80 and
              (m.get('real_data_hallucination_rate') or 1) <= 0.05)
        check("GO (real data viability)", go)
    else:
        print("  INFO: < 5 real-data questions. GO requires real Tofino docs.")

all_pass = all(p for _, p in checks)
for name, passed in checks:
    print(f'  {"PASS" if passed else "FAIL"}: {name}')
print(f'\nDONE: {"YES" if all_pass else "NO"} ({sum(p for _,p in checks)}/{len(checks)} passed)')
sys.exit(0 if all_pass else 1)
```

**Stop Condition:** Agent work is complete when `python scripts/check_done.py` exits 0.
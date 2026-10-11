# Enterprise RAG Lab Workbook

**Project:** Production-grade Retrieval-Augmented Generation (RAG) for PDFs with text, tables, and diagrams  
**Mode:** Build it yourself, one lab at a time  
**Primary language:** Python  
**Initial assumption:** Start locally, then deploy when the quality and reliability evidence justifies it  
**Document profile:** PDFs averaging 10–20 pages; some text-heavy, table-heavy, scanned, or diagram-heavy

---

## How to use this workbook

Treat every lab as an engineering assignment, not a tutorial to copy verbatim.

For each lab:

1. Read only the linked documentation sections needed for the current task.
2. Write a short design note before coding. State your assumptions and alternatives.
3. Implement the smallest working version yourself.
4. Add tests and keep a record of failures.
5. Run the experiment and save its output.
6. Answer the review questions in your own words.
7. Do not move on until the acceptance criteria pass, or document why they do not.

When you get stuck, bring the **specific task, what you tried, the relevant code, the error/log, and your hypothesis**. The purpose is for you to understand the choices rather than receive a finished codebase.

### Rules for this project

- Do not begin by wiring every framework together. First implement and measure a small baseline.
- Keep interfaces around replaceable components: parser, chunker, embedder, retriever, reranker, generator, evaluator.
- Record dependency versions and configuration for every meaningful experiment.
- Never tune only on the final evaluation set. Maintain development and held-out test sets.
- Do not assume an LLM judge is ground truth. Manually inspect samples and disagreements.
- Treat source documents as untrusted input. A PDF can contain malicious prompt instructions, misleading tables, or sensitive information.
- Use synthetic or public documents until data-handling and access controls are ready.

## Reference architecture

```text
PDF upload / document source
        |
        v
Validation -> durable ingestion job -> parser/OCR/layout extraction
        |                                      |
        |                                extraction checks
        v                                      v
Document registry <- normalized document + page/element provenance
        |
        v
Structure-aware chunking -> chunk metadata -> embedding / lexical indexing
                                                   |
                                                   v
Question -> validation/auth -> query processing -> hybrid retrieval
                                                   |
                                                   v
                                         rerank/context selection
                                                   |
                                                   v
                                      answer generation + citations
                                                   |
                                                   v
                              evidence/citation/guardrail validation
                                                   |
                                                   v
                                             API response

Cross-cutting: evaluation, traces, structured logs, metrics, access control,
versioning, retries, audit events, secrets, tests, backups and runbooks.
```

This is a target architecture, not a requirement to build all components on day one.

---

# Phase 0 — Project setup and scope

## Lab 0: Define the problem before choosing the stack

**Goal:** Produce testable requirements and a small, representative PDF corpus.

### Task 0.1 — Write the requirements

Write `docs/requirements.md` covering:

- Who asks questions and who owns documents.
- Expected query types: lookup, summarization, multi-document questions, table questions, diagram questions, and unanswerable questions.
- Whether documents have different access permissions or tenants.
- Approximate number of documents now and expected growth.
- Freshness requirements: how quickly new or changed PDFs must become searchable.
- Expected concurrent users and rough request volume, even if initially estimated.
- Data sensitivity, retention/deletion requirements, residency, and whether external model APIs are allowed.
- Desired response style, citation granularity, and abstention behaviour.
- A first latency and cost budget. Label estimates as estimates, not measured SLOs.

**Read beside this task:**
- [FastAPI — first steps and application structure](https://fastapi.tiangolo.com/tutorial/)
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Google SRE Workbook — SLO-related concepts](https://sre.google/workbook/implementing-slos/)

### Task 0.2 — Build the evaluation corpus

Choose a small set of public or synthetic PDFs (for example, 15–30 initially). Include:

- A normal digitally generated PDF.
- A PDF with multi-page tables or repeated table headers.
- A scanned PDF requiring OCR.
- A document with charts, process diagrams, or architecture diagrams.
- A document with two-column or unusual reading order.
- At least one document whose relevant answer spans multiple pages.
- At least one document that does **not** contain the answer to a plausible question.

Create a `corpus_manifest.csv` with `document_id`, `path`, `sha256`, `page_count`, `document_type`, `expected_features`, `sensitivity`, and `version`.

**Read beside this task:**
- [PyMuPDF — document basics, text extraction and OCR](https://pymupdf.readthedocs.io/en/latest/the-basics.html)
- [Docling documentation](https://docling-project.github.io/docling/)

### Task 0.3 — Create engineering hygiene

Set up a Git repository, Python project metadata, environment configuration, lint/format tools of your choice, and a test command. Do not commit API keys or real confidential PDFs.

**Read beside this task:**
- [Python Packaging User Guide](https://packaging.python.org/en/latest/)
- [pytest getting started](https://docs.pytest.org/en/stable/getting-started.html)
- [GitHub Actions documentation](https://docs.github.com/en/actions)

### Questions to answer

1. Which requirements are hard constraints versus preferences?
2. What is the cost of a missing answer, a wrong answer, or an unauthorized answer? Are these equally severe?
3. Which PDFs represent the hardest realistic cases, and why?
4. What evidence would justify choosing one parser, vector store, or model over another?

### Deliverables

- `docs/requirements.md`
- `docs/threat-model.md` (initial draft; refine in Lab 7)
- `docs/architecture.md` (first diagram and assumptions)
- `data/corpus_manifest.csv`
- A reproducible development setup and one passing smoke test

### Acceptance criteria

- Another developer can set up the repository from the instructions.
- The corpus contains examples for the major layout types.
- Requirement statements are testable, or clearly labelled as decisions still needed.

---

# Phase 1 — Ingestion and document understanding

## Lab 1: Compare PDF parsers; do not trust extracted text blindly

**Goal:** Determine what information each parser preserves or loses.

### Task 1.1 — Implement a parser adapter

Create a small interface that accepts a PDF path and returns a normalized document object. Preserve at least:

- Stable document ID and content hash.
- Page number (choose and document zero- or one-based indexing).
- Element type: heading, paragraph, list, table, figure/image, caption, or unknown.
- Text or structured element content.
- Reading order where available.
- Page/element bounding box when available.
- Parser name and version.
- Warnings/errors and extraction metadata.

Start with one parser; keep the interface replaceable. Do not collapse all content to one string.

**Read beside this task:**
- [Docling documentation and examples](https://docling-project.github.io/docling/)
- [PyMuPDF text and page extraction](https://pymupdf.readthedocs.io/en/latest/the-basics.html)
- [PyMuPDF text recipes](https://pymupdf.readthedocs.io/en/latest/recipes-text.html)

### Task 1.2 — Run a parser bake-off

Select at least 5 PDFs that represent different failure modes. Run the same documents through your primary parser and one baseline/alternative. For each file, manually inspect selected pages and record:

- Text extraction completeness.
- Reading order and heading retention.
- Table row/column fidelity.
- OCR correctness on scanned pages.
- Figure/caption extraction and coordinates.
- Time, peak memory if practical, errors, and warnings.

Do not report one global extraction score without explaining how it was measured.

**Read beside this task:**
- [Docling — project documentation](https://docling-project.github.io/docling/)
- [PyMuPDF — OCR recipe](https://pymupdf.readthedocs.io/en/latest/recipes-ocr.html)

### Task 1.3 — Add ingestion failure handling

Distinguish unsupported/corrupt PDFs, encrypted documents, empty extraction, OCR-required pages, parser timeouts, and internal errors. Persist a status and reason; failed files should not quietly appear as successfully indexed.

**Read beside this task:**
- [Pydantic models and validation](https://docs.pydantic.dev/latest/concepts/models/)
- [Python logging cookbook](https://docs.python.org/3/howto/logging-cookbook.html)

### Questions to answer

1. How do you detect a scanned page when text extraction returns almost nothing?
2. What information is lost when a PDF is converted to Markdown or plain text?
3. How will you represent one table that continues onto the next page?
4. When should parsing fail the whole document, and when can it continue with a warning?
5. Which parts of parser output must be reproducible and versioned?

### Deliverables

- `src/ingestion/parsers/` with an adapter and a first parser.
- A normalized document schema.
- `experiments/parser_bakeoff.csv` plus notes/screenshots for manually inspected examples.
- Unit tests for corrupt, empty, scanned and normal PDFs.

### Acceptance criteria

- Every extracted element can be traced to a document and page.
- Failures are explicit and inspectable.
- You have documented a parser choice based on observed corpus behaviour, not popularity alone.

---

## Lab 2: Tables, diagrams, OCR and extraction quality

**Goal:** Keep structured visual evidence useful for retrieval and citation.

### Task 2.1 — Preserve tables as tables

For each table, store both a structured representation (rows/cells or an equivalent schema) and a retrieval-oriented textual rendering. Include the table title/caption, column headers, units, page number and source coordinates where available. Test repeated headers, merged cells, footnotes, negative numbers and units.

**Read beside this task:**
- [Docling documentation](https://docling-project.github.io/docling/)
- [PyMuPDF table extraction recipes](https://pymupdf.readthedocs.io/en/latest/recipes-text.html)

### Task 2.2 — Make diagrams discoverable

Start with a baseline: extract images/figures, captions, nearby headings, page number and a stable image reference. Then test one vision-capable approach on diagram-heavy PDFs. Store generated descriptions separately from source text and mark them as model-generated, with model/version metadata.

**Read beside this task:**
- [PyMuPDF — image extraction](https://pymupdf.readthedocs.io/en/latest/recipes-images.html)
- [Docling documentation](https://docling-project.github.io/docling/)

Do not assume a generated figure description is an authoritative source. Retrieval results should make clear whether the evidence came from document text, a table, OCR, or a model-generated image description.

### Task 2.3 — Add extraction-quality checks

Develop checks for blank pages, suspiciously low text density, duplicate content, invalid/unusual page references, malformed tables and figures without metadata. Produce a per-document extraction report.

**Read beside this task:**
- [pytest parameterization](https://docs.pytest.org/en/stable/how-to/parametrize.html)
- [Pydantic validators](https://docs.pydantic.dev/latest/concepts/validators/)

### Questions to answer

1. How could flattening a table into unstructured text change the meaning of a cell?
2. How would you test whether a chart description preserves its trend, axes and units correctly?
3. What will the system do when OCR or vision extraction has low confidence?
4. How do you avoid confusing original document content with machine-generated descriptions?

### Deliverables

- Table and figure data models.
- An extraction-quality report for the corpus.
- A set of regression PDFs with expected extraction observations.

### Acceptance criteria

- Table headers and units remain associated with the right values.
- Figures can be found by caption/context and linked back to the correct page.
- Low-quality extraction is flagged instead of silently treated as reliable evidence.

---

# Phase 2 — Normalized data, chunking and indexing

## Lab 3: Build structure-aware chunks

**Goal:** Create retrieval units that preserve meaning and provenance.

### Task 3.1 — Design a chunk schema

Each chunk should carry fields such as:

- `chunk_id`, `document_id`, content hash and document version.
- Text used for embedding/search and optional structured payload.
- Section path/headings and element type.
- Page range and source element IDs.
- Table/figure identifiers when relevant.
- Tenant/access metadata where applicable.
- Chunking strategy/version and token count.

Define stable IDs so re-running ingestion does not create duplicates. Decide what happens when the source document changes.

**Read beside this task:**
- [LlamaIndex — ingestion pipeline](https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/)
- [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/)

### Task 3.2 — Implement and compare chunking strategies

Implement at least two strategies:

1. Fixed-size chunking with a configurable overlap as a baseline.
2. Structure-aware chunking based on headings, paragraphs, lists, tables and captions.

Optional third experiment: parent-child chunks, where small retrieval units point to larger context units. Track the token count of every chunk and define safe handling for very large tables or sections.

**Read beside this task:**
- [LlamaIndex — node parsing and chunking](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)
- [Sentence Transformers — semantic search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)

### Task 3.3 — Build deterministic re-ingestion

Run ingestion twice on an unchanged document. Confirm that chunk IDs and index contents are stable. Then change the source and define how stale chunks are removed, replaced or retained for version history.

**Read beside this task:**
- [Qdrant — points and payloads](https://qdrant.tech/documentation/concepts/points/)
- [PostgreSQL — transactions](https://www.postgresql.org/docs/current/tutorial-transactions.html)

### Questions to answer

1. Why might a chunk with excellent semantic coherence still be a bad retrieval unit?
2. What is the tradeoff between chunk size, retrieval recall and generation context cost?
3. What should happen to citations when a document is re-indexed?
4. How will you represent a table too large to fit comfortably in one chunk?

### Deliverables

- `src/ingestion/chunking.py`.
- Chunk schema and ID/versioning policy.
- Chunk inspection tool or report.
- Experiment table comparing chunking strategies.

### Acceptance criteria

- Chunks preserve section context and source provenance.
- Re-ingestion is idempotent for unchanged files.
- You can compare chunk sizes and strategies against a fixed evaluation query set later.

---

## Lab 4: Add embeddings and build retrieval baselines

**Goal:** Measure retrieval before adding complexity.

### Task 4.1 — Establish lexical and dense baselines

Implement a lexical baseline (for example BM25) and a dense embedding baseline. Keep text normalization, embedding model ID, model revision, dimensions and normalization settings in configuration/metadata. Select a small embedding-model comparison set rather than trying many models without a hypothesis.

**Read beside this task:**
- [Sentence Transformers — semantic search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
- [Qdrant documentation](https://qdrant.tech/documentation/)
- [Elasticsearch — BM25 similarity reference](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html)

### Task 4.2 — Choose and operate a vector store

Use Qdrant as the first vector-store candidate. Learn collections, points, payload indexes, filtering, persistence and backups. Keep document metadata in a durable registry as well; do not make the vector index the only source of truth for document lifecycle.

**Read beside this task:**
- [Qdrant quickstart/documentation](https://qdrant.tech/documentation/)
- [Qdrant filtering](https://qdrant.tech/documentation/concepts/filtering/)
- [Qdrant snapshots](https://qdrant.tech/documentation/concepts/snapshots/)

### Task 4.3 — Create a retrieval benchmark

For each question, annotate the relevant chunk(s) or source elements. Measure at least Recall@k, Precision@k, MRR and nDCG@k. Explain the difference between “relevant document retrieved” and “required evidence retrieved.” Store actual result lists so errors can be inspected.

**Read beside this task:**
- [Stanford IR book — evaluation in information retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)
- [Qdrant hybrid search concepts](https://qdrant.tech/documentation/search/text-search/hybrid-search/)

### Questions to answer

1. Which query types favour lexical search (exact codes, names, model numbers)? Which favour semantic search?
2. Why can high Recall@k coexist with poor answer quality?
3. How will filters affect recall, and how will you test them?
4. What costs do index size, vector dimensions and reranking add?

### Deliverables

- Lexical and dense retrievers behind a common interface.
- A versioned retrieval evaluation dataset.
- Retrieval metric implementation and result inspection report.

### Acceptance criteria

- Every metric can be recomputed from saved query and relevance judgments.
- The benchmark includes exact-match queries, paraphrases, table questions and unanswerable questions.
- A reproducible baseline is available before hybrid retrieval is introduced.

---

# Phase 3 — Retrieval quality and grounded answers

## Lab 5: Hybrid retrieval, reranking and context construction

**Goal:** Improve evidence selection only when experiments show that it helps.

### Task 5.1 — Build hybrid search

Combine lexical and dense retrieval. Compare a rank-fusion method (for example Reciprocal Rank Fusion) with any score-weighted method you choose. Tune on development questions only. Measure recall, ranking quality, latency and cost.

**Read beside this task:**
- [Qdrant — hybrid search](https://qdrant.tech/documentation/search/text-search/hybrid-search/)
- [Qdrant — hybrid and multi-stage queries](https://qdrant.tech/documentation/search/hybrid-queries/)

### Task 5.2 — Add reranking as a controlled experiment

Rerank the top-N candidates with a cross-encoder or another explicit reranker. Compare no reranker versus reranker at the same candidate depth. Check whether improvements are consistent across query categories and whether the latency budget remains acceptable.

**Read beside this task:**
- [Sentence Transformers — retrieve and rerank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- [Qdrant — hybrid search](https://qdrant.tech/documentation/search/text-search/hybrid-search/)

### Task 5.3 — Build context assembly

Given a query and ranked chunks, choose evidence under a token budget. Preserve table headers, relevant parent context and source metadata. Avoid duplicates and avoid truncating the exact row or paragraph that contains the answer. Log which evidence was included and excluded.

**Read beside this task:**
- [LlamaIndex — retrievers](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/)
- [OpenAI API documentation](https://platform.openai.com/docs/overview) or the documentation for your chosen model provider

### Questions to answer

1. Does hybrid search improve which queries? Give concrete examples and metrics.
2. How do you choose candidate depth N and final top-k?
3. Can a reranker improve ranking but make end-to-end latency unacceptable?
4. How will you prevent duplicated chunks from consuming the context budget?

### Deliverables

- Hybrid retriever and optional reranker.
- Configuration-based experiment runner.
- Experiment report with quality/latency/cost comparison.

### Acceptance criteria

- You can roll back to the best baseline through configuration.
- Improvements are measured on the same development set.
- The final context is inspectable and every element has provenance.

---

## Lab 6: Answer generation, citations and abstention

**Goal:** Return verifiable answers, not fluent unsupported prose.

### Task 6.1 — Define the answer contract

Create a structured response schema with an answer, citations, an abstention/status field, and optional caveats. A citation should reference a known `document_id`, version, page and supporting chunk/element ID. Validate citations against evidence actually retrieved for that request.

**Read beside this task:**
- [Pydantic — models and validation](https://docs.pydantic.dev/latest/concepts/models/)
- [FastAPI — response models](https://fastapi.tiangolo.com/tutorial/response-model/)

### Task 6.2 — Write and test the generation prompt

Tell the model to answer using supplied evidence, distinguish conflicting sources, state when evidence is inadequate, and treat document content as data rather than instructions. Make the prompt versioned and testable. Prompts help behaviour but are not a security boundary.

**Read beside this task:**
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [OpenAI prompt engineering guide](https://platform.openai.com/docs/guides/prompt-engineering) or your provider's prompt guidance

### Task 6.3 — Implement citation checks and abstention

Test: answer present in one paragraph; answer in a table; answer across multiple pages; conflicting versions; missing answer; irrelevant retrieved context; and a retrieved chunk that contains an instruction attempting to override system behaviour. Define how citations are displayed to the user.

**Read beside this task:**
- [Ragas — available metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)
- [Phoenix — evaluation](https://arize.com/docs/phoenix/)

### Questions to answer

1. What counts as a supported claim, and how will you detect an unsupported claim within an otherwise correct answer?
2. Should the system answer partially when it has evidence for only part of the question?
3. How does it resolve conflicts between two document versions?
4. What should it do when retrieval is confident but the evidence does not support the proposed answer?

### Deliverables

- Generation module and response schema.
- Citation validator.
- Abstention and conflict test cases.
- A sample of manually reviewed answers with explanations.

### Acceptance criteria

- Unknown or invalid citation IDs cannot be returned as valid citations.
- The pipeline has explicit behaviour for insufficient or conflicting evidence.
- Answers can be traced to retrieved evidence and a specific prompt/model version.

---

# Phase 4 — Evaluation as an engineering discipline

## Lab 7: Build the evaluation and regression harness

**Goal:** Know whether the pipeline is improving and where it is failing.

### Task 7.1 — Create a gold dataset

Start with about 100–200 manually reviewed questions if feasible; scale the set according to document variety and risk. Each item should include:

- Question and category.
- Expected answer or acceptable answer points.
- Relevant document/page/chunk references.
- Whether abstention is expected.
- Difficulty and risk labels.
- Annotation notes and dataset version.

Include retrieval questions, multi-hop questions, exact values from tables, diagram interpretation, unanswerable questions, stale/conflicting versions and security/adversarial cases. Keep a held-out test split that you do not repeatedly tune against.

**Read beside this task:**
- [Ragas — metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)
- [Phoenix — datasets and experiments](https://arize.com/docs/phoenix/datasets-and-experiments/)
- [MLflow — RAG evaluation with judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/)

### Task 7.2 — Evaluate each stage independently

Measure at least:

- Parsing: extraction failures, manually assessed table fidelity, page/element provenance coverage.
- Retrieval: Recall@k, Precision@k, MRR and nDCG@k.
- Context: evidence coverage and irrelevant-context rate.
- Generation: factual correctness, groundedness/faithfulness, relevance and abstention correctness.
- Citation: citation coverage, support precision and page accuracy.
- Operations: latency, token use, model/API cost and error rate.

Define each metric in `docs/metrics.md`, including its unit, calculation, data needed, known failure modes and target/threshold if one is justified. Avoid assigning arbitrary targets without a baseline and use case.

**Read beside this task:**
- [Ragas metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)
- [MLflow RAG evaluation](https://mlflow.org/cookbook/rag-evaluation)
- [Stanford IR evaluation chapter](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html)

### Task 7.3 — Validate LLM-as-a-judge

For a stratified sample, compare automated scores with human judgments. Record disagreement, judge prompt/model version and rationale. Look for position bias, verbosity bias, and cases where the judge rewards a plausible but unsupported answer.

**Read beside this task:**
- [Phoenix evaluations and annotations](https://arize.com/docs/phoenix/)
- [Ragas — metrics and custom metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)

### Task 7.4 — Make every change a benchmarkable experiment

Record the corpus version, code commit, parser, chunker, embedding model, index configuration, retriever, reranker, prompt/model and metric versions. Compare changes on the same dataset. Save bad cases and add regression tests for failures that should not recur.

**Read beside this task:**
- [MLflow tracking and experiments](https://mlflow.org/docs/latest/ml/tracking/)
- [Phoenix datasets and experiments](https://arize.com/docs/phoenix/datasets-and-experiments/)

### Questions to answer

1. What evidence do you need to claim a retrieval change is better rather than merely different?
2. Why can aggregate scores hide a serious failure on table or diagram questions?
3. What happens when LLM judges disagree with domain experts?
4. How will you detect regression when the model provider changes its model version?

### Deliverables

- Versioned gold dataset and annotation instructions.
- Evaluation runner and metric definitions.
- Baseline report and per-category error analysis.
- Regression-test policy.

### Acceptance criteria

- A single command or documented procedure reproduces the evaluation.
- Results can be compared across configuration versions.
- Results include per-category analysis and reviewed examples, not only one aggregate score.

---

# Phase 5 — Security and enterprise access controls

## Lab 8: Threat model, permission-aware retrieval and adversarial tests

**Goal:** Protect private documents and make untrusted content safe to process.

### Task 8.1 — Refine the threat model

List assets, trust boundaries, actors, attack scenarios, impact and mitigations. Include unauthorized retrieval, cross-tenant access, prompt injection in document text/images, sensitive data in logs, malicious uploads, resource exhaustion and unsafe downstream handling of generated output.

**Read beside this task:**
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [OWASP GenAI Security Project](https://genai.owasp.org/)

### Task 8.2 — Enforce authorization before generation

Associate document/chunk metadata with an access policy. Apply server-side authorization filters to retrieval and verify the result set before sending evidence to the LLM. Do not rely on the prompt to prevent unauthorized disclosure. Test a user who attempts to request a document they are not authorized to see.

**Read beside this task:**
- [Qdrant payload filtering](https://qdrant.tech/documentation/concepts/filtering/)
- [FastAPI security tutorials](https://fastapi.tiangolo.com/tutorial/security/)

### Task 8.3 — Add adversarial and isolation tests

Create tests for malicious instructions in PDFs, hidden text, requests to ignore access policy, cross-tenant queries, forged citations, oversized uploads, malformed files, repeated requests and sensitive content appearing in traces. Document residual risk; no single prompt or classifier solves prompt injection completely.

**Read beside this task:**
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [pytest parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html)

### Task 8.4 — Define logging and retention policy

Specify which fields can be logged, which must be redacted, who can view traces, retention duration, deletion procedures and whether user prompts or source excerpts are stored. Test document deletion across the registry, search index, caches and derived artifacts.

**Read beside this task:**
- [OpenTelemetry — sensitive data handling guidance](https://opentelemetry.io/docs/security/)
- [OWASP Logging Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html)

### Questions to answer

1. Where exactly is authorization enforced, and can an untrusted query bypass it?
2. How will a document update or deletion propagate through embeddings, caches and backups?
3. Which logs contain content that may be confidential?
4. What threat remains even after prompt injection defenses are implemented?

### Deliverables

- Updated threat model and data-flow diagram.
- Permission-aware retrieval implementation.
- Security test suite and deletion/retention policy.
- Documented residual risks.

### Acceptance criteria

- Automated tests demonstrate that unauthorized content is never passed to the generator.
- Deletion behaviour is documented and tested across active stores and derived artifacts.
- Security controls are enforced outside the model prompt.

---

# Phase 6 — Observability, API and resilience

## Lab 9: Trace the full request and define service objectives

**Goal:** Diagnose quality, latency, errors and cost from evidence.

### Task 9.1 — Instrument a query trace

Create trace spans for request validation, authorization, query embedding, lexical search, dense search, fusion, reranking, context construction, generation and citation validation. Attach non-sensitive IDs and version/configuration metadata. Avoid storing document text or full prompts by default.

**Read beside this task:**
- [OpenTelemetry Python instrumentation](https://opentelemetry.io/docs/languages/python/)
- [Arize Phoenix tracing](https://arize.com/docs/phoenix/tracing/)
- [OpenInference instrumentation](https://github.com/Arize-ai/openinference)

### Task 9.2 — Add logs, metrics and dashboards

At minimum track request count, error rate, p50/p95 latency, model/API latency, tokens, estimated cost per query, retrieval candidate counts, citation-validation failures, ingestion queue depth, failed jobs, index freshness and document deletion completion. Use low-cardinality metric labels; don't put arbitrary questions or document IDs in metric labels.

**Read beside this task:**
- [OpenTelemetry — metrics](https://opentelemetry.io/docs/concepts/signals/metrics/)
- [Prometheus documentation](https://prometheus.io/docs/introduction/overview/)
- [Grafana documentation](https://grafana.com/docs/)

### Task 9.3 — Implement bounded resilience

Add explicit timeouts, bounded retries with backoff, rate-limit handling, idempotency for ingestion jobs and safe error responses. Decide which failures are retryable and which need operator intervention. Avoid retrying invalid requests forever.

**Read beside this task:**
- [Tenacity retry library](https://tenacity.readthedocs.io/en/latest/)
- [FastAPI error handling](https://fastapi.tiangolo.com/tutorial/handling-errors/)

### Task 9.4 — Define initial SLOs and alerts

Propose SLOs for latency, successful responses, ingestion completion and index freshness. Set thresholds after measuring a baseline. Define alerts with runbook links and explain the user impact of each threshold.

**Read beside this task:**
- [Google SRE Workbook — implementing SLOs](https://sre.google/workbook/implementing-slos/)
- [Prometheus alerting rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)

### Questions to answer

1. Can a trace explain a slow request end to end?
2. How do you distinguish a retrieval-quality issue from a model latency issue?
3. What should be retried, and what could a retry make worse?
4. Which dashboard metric is actionable, and what runbook follows when it breaches a threshold?

### Deliverables

- Traceable request pipeline.
- Structured logging and metrics dashboard.
- Initial SLO document and operational runbooks.
- Tests for timeout, provider failure and retry exhaustion.

### Acceptance criteria

- A request can be diagnosed without guessing which pipeline stage failed.
- Telemetry avoids unnecessary sensitive content.
- Retries are bounded and observable.

---

## Lab 10: API, reproducible deployment and testing pyramid

**Goal:** Package the pipeline as a maintainable service.

### Task 10.1 — Create a typed API

Define endpoints for health/readiness, document ingestion/status and querying. Validate requests/responses, map internal failures to suitable HTTP responses and keep ingestion work separate from latency-sensitive query handling. Add authentication before exposing real documents.

**Read beside this task:**
- [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/)
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Pydantic validation](https://docs.pydantic.dev/latest/concepts/validators/)

### Task 10.2 — Containerize the service

Use Docker to package the API and its dependencies. Keep configuration and secrets outside the image. Decide which services belong in local Compose (for example API, Qdrant, PostgreSQL, and Phoenix). Add health checks and persistent volumes where needed.

**Read beside this task:**
- [Docker documentation](https://docs.docker.com/)
- [Docker Compose documentation](https://docs.docker.com/compose/)
- [Twelve-Factor App — config](https://12factor.net/config)

### Task 10.3 — Add layered tests and CI

Include:

- Unit tests for chunking, filtering logic, citation validation and metric computation.
- Integration tests for parser, index and database interactions.
- End-to-end tests for known questions.
- Security tests for permission and prompt-injection cases.
- Regression evaluation on a stable subset in CI; run the full evaluation on demand or on a schedule if it is too costly.

**Read beside this task:**
- [pytest documentation](https://docs.pytest.org/en/stable/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)

### Task 10.4 — Write deployment and recovery instructions

Document environment variables, secrets setup, migrations, index rebuild, backup/restore, model/provider configuration, rollback, readiness checks and common failure diagnosis. Rehearse bringing the system up from a clean checkout.

**Read beside this task:**
- [Qdrant snapshots](https://qdrant.tech/documentation/concepts/snapshots/)
- [PostgreSQL backup and restore](https://www.postgresql.org/docs/current/backup.html)
- [Docker Compose](https://docs.docker.com/compose/)

### Questions to answer

1. What is the difference between liveness and readiness?
2. Which tests should block a pull request, and which should run less frequently?
3. How do you deploy an embedding-model change without mixing incompatible vectors?
4. Can you recover from a lost index using the source documents and registry?

### Deliverables

- FastAPI application and API contract.
- Docker/Compose configuration for local development.
- CI test workflow.
- Deployment, rollback and restore runbooks.

### Acceptance criteria

- A clean environment can run the documented smoke test.
- Required unit/integration/security tests pass.
- The service reports health/readiness correctly and fails safely when dependencies are unavailable.

---

# Phase 7 — Load testing and production readiness

## Lab 11: Load test, find bottlenecks and prepare a release checklist

**Goal:** Replace guesses about capacity with measurements.

### Task 11.1 — Design a load-test profile

Describe realistic traffic: concurrent users, query categories, document ingestion bursts, file size distribution, provider rate limits and expected peak period. Test warm/cold caches if relevant. Avoid measuring only an idle service with one request at a time.

**Read beside this task:**
- [Locust documentation](https://docs.locust.io/)
- [Prometheus metrics](https://prometheus.io/docs/introduction/overview/)

### Task 11.2 — Measure latency and cost by stage

Record p50/p95/p99 latency, throughput, errors, queue depth, token usage and estimated cost under load. Identify whether the bottleneck is parsing, database access, embedding, retrieval, reranking, provider limits or generation.

**Read beside this task:**
- [OpenTelemetry metrics concepts](https://opentelemetry.io/docs/concepts/signals/metrics/)
- [Phoenix tracing](https://arize.com/docs/phoenix/tracing/)

### Task 11.3 — Test failure and recovery

Test a model-provider outage, database/vector-store unavailability, malformed PDF, stuck ingestion job, rate limit, index rebuild and backup restore. Record expected behaviour, observed behaviour and recovery steps.

**Read beside this task:**
- [Docker Compose](https://docs.docker.com/compose/)
- [Qdrant snapshots](https://qdrant.tech/documentation/concepts/snapshots/)
- [PostgreSQL backup and restore](https://www.postgresql.org/docs/current/backup.html)

### Task 11.4 — Make the release decision

Create a release checklist covering quality gates, security, performance, operational readiness, data handling, cost, rollback and known limitations. A high average metric cannot override a critical security failure.

**Read beside this task:**
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Google SRE Workbook](https://sre.google/workbook/table-of-contents/)

### Questions to answer

1. At what request volume does latency or cost stop meeting your requirements?
2. Which component should be scaled first, based on evidence?
3. What does the user see when a dependency is unavailable?
4. What measurable conditions must be true before the system is called production-ready?

### Deliverables

- Load-test scripts and results.
- Bottleneck analysis and follow-up tasks.
- Backup/restore evidence.
- Production-readiness checklist and limitations document.

### Acceptance criteria

- Performance claims include a test profile, environment and measurements.
- The team can restore service from at least one rehearsed failure scenario.
- Release approval is based on explicit quality, security and reliability gates.

---

# Framework selection lab — learn the ecosystem without collecting frameworks

These tools overlap. Select one primary path and learn the alternatives by comparison, not by using everything in the production path.

| Area | First candidate | Alternative to investigate | Assignment question |
|---|---|---|---|
| PDF/layout extraction | [Docling](https://docling-project.github.io/docling/) | [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) | Which better preserves the structures your corpus actually contains? |
| RAG abstractions | [LlamaIndex](https://docs.llamaindex.ai/en/stable/) | [LangChain](https://python.langchain.com/docs/), [Haystack](https://docs.haystack.deepset.ai/) | Which abstraction helps without hiding debugging-critical behaviour? |
| Stateful workflow orchestration | Direct Python functions initially | [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) | Do you actually need branching, persisted workflow state or human approval? |
| Embeddings and reranking | [Sentence Transformers](https://www.sbert.net/) | Provider embedding APIs | Which has the best retrieval-quality/latency/cost tradeoff on the benchmark? |
| Vector database | [Qdrant](https://qdrant.tech/documentation/) | [pgvector](https://github.com/pgvector/pgvector) | What are the operational, filtering, search and backup tradeoffs? |
| RAG evaluation | [Ragas](https://docs.ragas.io/) | [DeepEval](https://docs.confident-ai.com/), [MLflow GenAI evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/) | Which evaluator is useful, reproducible and consistent with human review? |
| LLM observability | [Phoenix](https://arize.com/docs/phoenix/) | [MLflow tracing](https://mlflow.org/docs/latest/genai/tracing/) | Which gives the traces and experiments your debugging workflow needs? |
| API/schema | [FastAPI](https://fastapi.tiangolo.com/) + [Pydantic](https://docs.pydantic.dev/latest/) | Other Python API frameworks | Can the API contract be validated and tested independently of the model? |
| Telemetry | [OpenTelemetry](https://opentelemetry.io/docs/) | Vendor-specific instrumentation | Can telemetry remain portable and avoid exposing sensitive content? |
| Testing | [pytest](https://docs.pytest.org/en/stable/) | — | Are unit, integration, security and evaluation regression tests separated clearly? |
| Deployment | [Docker](https://docs.docker.com/) + Compose | Cloud container services | Can another developer reproduce the environment reliably? |

**Recommended initial stack:** Python, Pydantic, Docling plus a PyMuPDF comparison, Sentence Transformers or one hosted embedding API, Qdrant, a thin custom retrieval/generation layer, Ragas or a small custom evaluator plus human review, Phoenix, OpenTelemetry, FastAPI, pytest and Docker Compose. PostgreSQL becomes useful for the document registry and operational metadata. Add Redis/a job queue, orchestration frameworks, Kubernetes and advanced deployment services only when workload or requirements justify them.

---

# Project-wide scorecard

Maintain `docs/scorecard.md` and update it at the end of each lab. Do not fill in results before measuring them.

| Area | Baseline | Current | Target | Evidence / report | Status |
|---|---:|---:|---:|---|---|
| Parser success rate | TBD | TBD | TBD | Parser benchmark | Not measured |
| Table extraction fidelity | TBD | TBD | TBD | Manually inspected gold tables | Not measured |
| Figure/page provenance coverage | TBD | TBD | TBD | Extraction report | Not measured |
| Retrieval Recall@k | TBD | TBD | TBD | Retrieval evaluation | Not measured |
| Retrieval nDCG@k | TBD | TBD | TBD | Retrieval evaluation | Not measured |
| Faithfulness / groundedness | TBD | TBD | TBD | Human-reviewed answer evaluation | Not measured |
| Citation support precision | TBD | TBD | TBD | Citation audit | Not measured |
| Correct abstention rate | TBD | TBD | TBD | Unanswerable-question tests | Not measured |
| Unauthorized retrieval cases | TBD | TBD | 0 observed in test suite | Security tests | Not measured |
| Query p95 latency | TBD | TBD | TBD | Load-test report | Not measured |
| Estimated cost per query | TBD | TBD | TBD | Token/provider logs | Not measured |
| Ingestion failure visibility | TBD | TBD | All failures classified/logged | Ingestion tests | Not measured |
| Restore test | TBD | TBD | Successful rehearsal | Runbook evidence | Not measured |

Targets must be chosen from the actual use case, baseline, risk and available budget. “Zero unauthorized retrievals in tests” is necessary but does not prove that no unknown security issue exists.

---

# Final capstone submission

At completion, the repository should include:

- `README.md` with clean setup and demo instructions.
- Requirements, architecture, threat model, data model and architecture decision records.
- Parser benchmark and extraction-quality report.
- Versioned chunking/indexing and retrieval implementation.
- Versioned evaluation dataset, metric definitions, benchmark results and error analysis.
- Security and regression test suites.
- Traces, logs, metrics and a documented dashboard.
- Docker setup, CI configuration and API contract.
- Load-test results, backup/restore evidence, runbooks and production-readiness checklist.
- A limitations section describing unsupported PDF patterns and known failure modes.

## What to do first

Complete **Lab 0 only** first. Do not install every framework or build the full architecture yet. Create the requirements document, select representative PDFs, make the corpus manifest and get the development/test setup working. Then compare parsers in Lab 1. When those artifacts exist, use the results—not assumptions—to choose the next implementation step.

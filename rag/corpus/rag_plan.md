We are now ready to implement the RAG phase of the DEX IT-support LLM project.

Please inspect the existing repository before changing anything and reuse/refactor existing components where practical.

The goal is to build a clean, reproducible Retrieval-Augmented Generation pipeline using:

```text
1. Stack Overflow
   → broad technical troubleshooting knowledge

2. TechQA
   → IBM / enterprise technical support knowledge
```

The final system should support:

```text
Base Qwen3.5-9B
DEX fine-tuned Qwen3.5-9B

Base Qwen + RAG
DEX + RAG
```

and allow controlled benchmarking on our new held-out general IT-support benchmark.

Do NOT add additional documentation sources such as Microsoft, Red Hat, Cisco, Docker, Python, PostgreSQL, etc.

For this project, Stack Overflow + TechQA is enough.

Do NOT start any expensive full benchmark automatically.

Do NOT retrain the LLM.

You may run lightweight preprocessing, indexing, retrieval smoke tests, and unit tests.

---

# 1. Current Project Context

Base model:

```text
Qwen/Qwen3.5-9B
```

Use the same pinned base revision already defined by the project.

DEX is a QLoRA fine-tuned version of the same base model.

DEX was trained to behave as a general IT-support assistant.

The SFT dataset teaches behavior such as:

```text
troubleshooting
diagnosis
technical support responses
concise solutions
appropriate uncertainty / abstention
```

RAG should provide external technical knowledge that the model may not know from its weights.

The design principle is:

```text
SFT
→ teaches HOW DEX should respond

RAG
→ provides WHAT technical information DEX can use
```

Do not mix these responsibilities unnecessarily.

---

# 2. Evaluation Protocol Has Changed

TechQA is NO LONGER our primary benchmark.

We now have / are preparing a dedicated 1,000-example general IT-support benchmark drawn from the previous DEX validation pool.

Conceptually:

```text
27,000 train
2,000 validation
1,000 held-out general IT benchmark
```

The 1K benchmark must remain untouched by:

```text
training
validation
corpus construction
retrieval indexing
model selection
```

The new benchmark is intended to answer:

```text
Does DEX improve performance on unseen general IT-support questions?
```

and later:

```text
Does RAG improve performance further?
```

---

# 3. TechQA Has a New Role

TechQA should now be treated as a RAG KNOWLEDGE SOURCE.

Do NOT use TechQA as the primary benchmark after it has been indexed into RAG.

Existing historical TechQA benchmark results may remain in the repository for documentation, but future RAG results on TechQA must NOT be presented as independent held-out evaluation because the answers are now available to retrieval.

TechQA contains valuable IBM-specific information such as:

```text
WebSphere configuration
IBM Tivoli Monitoring
TIP / fix packs
SPSS
APAR information
IBM product versions
enterprise installation issues
```

This complements Stack Overflow well.

Conceptually:

```text
Stack Overflow
→ broad general technical knowledge

TechQA
→ IBM / enterprise-specific knowledge
```

---

# 4. Data Sources

## Stack Overflow

Use:

```text
krylodar/StackOverFlowQA
```

or the exact source already recorded/planned by this repository.

Pin and record the dataset revision when possible.

The dataset contains roughly one million Stack Overflow question/accepted-answer pairs.

We do NOT need the entire dataset.

Target:

```python
MAX_STACKOVERFLOW_RECORDS = 75_000
```

This is a maximum, not a quota.

If fewer than 75K high-quality relevant records survive filtering, keep fewer.

Do not lower quality simply to reach 75K.

---

## TechQA

Use the existing project TechQA data.

Current historical benchmark path is approximately:

```text
data/processed/evaluation/techqa_benchmark.jsonl
```

Reuse the actual existing path/configuration in the repository.

Only TechQA records with a meaningful non-empty reference answer should become RAG knowledge.

For example:

```json
{
  "question": "Does Linux KVM monitoring agent support CANDLEDATA?",
  "reference_answer": "The CANDLEDATA mechanism is available with ITM 6.30.05.00."
}
```

should be indexed.

A TechQA item with:

```text
reference_answer = null
```

contains no useful factual answer and should NOT be indexed as knowledge.

Report:

```text
total TechQA records
answerable records
indexed records
excluded null/unanswerable records
```

---

# Normalized Corpus Schema

Create a common internal schema for both sources.

Something conceptually like:

```json
{
  "id": "stackoverflow:123456",
  "source": "stackoverflow",
  "title": "SSH Permission denied after adding public key",
  "question": "...",
  "answer": "...",
  "retrieval_text": "...",
  "context_text": "...",
  "metadata": {
    "tags": ["linux", "ssh"],
    "question_score": 10,
    "answer_score": 18
  }
}
```

For TechQA:

```json
{
  "id": "techqa:TRAIN_Q046",
  "source": "techqa",
  "title": null,
  "question": "Does Linux KVM monitoring agent support CANDLEDATA?",
  "answer": "The CANDLEDATA mechanism is available with ITM 6.30.05.00.",
  "retrieval_text": "...",
  "context_text": "...",
  "metadata": {
    "product_family": "IBM"
  }
}
```

Adapt field names to the actual repository architecture.

IDs must be:

```text
stable
deterministic
unique across sources
```

Use source prefixes such as:

```text
stackoverflow:<source-id>
techqa:<source-id>
```

---

# 7. Stack Overflow Selection

We want IT-support / troubleshooting knowledge, not arbitrary Stack Overflow programming questions.

Prefer records relevant to areas such as:

```text
Linux
Ubuntu
Windows
PowerShell
Bash / shell
SSH
networking
DNS
HTTP / HTTPS
TLS / SSL
Nginx
Apache
servers
permissions
processes
system configuration
Python environment/package problems
pip
Git
databases
PostgreSQL
MySQL
SQL
web servers
deployment
software configuration
developer tooling
```

Use tags where available.

If tags are unavailable or incomplete, deterministic keyword matching is acceptable.

---

# 8. Stack Overflow Quality Filtering

Where source fields exist, prefer reasonable minimum-quality filters such as:

```text
question score >= 0
accepted answer score >= 1
```

Make these configurable.

Only keep records with:

```text
non-empty question
non-empty accepted answer
```

Prefer accepted-answer pairs.

Do not include questions with no useful answer.

Do not keep obvious spam/noise.

Do not perform overly aggressive filtering that removes legitimate troubleshooting questions.

---

# 9. Text Cleaning

Stack Overflow records may contain HTML.

Clean HTML deterministically while preserving important technical content.

Preserve:

```text
commands
code snippets
paths
error messages
configuration values
version numbers
stack traces when useful
```

Do not turn:

```text
sudo chmod 600 ~/.ssh/id_rsa
```

into broken prose.

Normalize excessive whitespace.

Do not aggressively strip punctuation or symbols because technical text depends on them.

For TechQA, perform only minimal whitespace normalization.

---

# 10. Deduplication

Perform exact deterministic deduplication.

At minimum check duplicates based on normalized:

```text
question/title + question
```

and optionally exact normalized:

```text
question + answer
```

Do not index the same Stack Overflow question multiple times.

Do not add expensive semantic-deduplication dependencies for this phase.

Lightweight near-duplicate detection is optional only if the repository already supports it cleanly.

---

# 11. Diversity

Do not simply take the 75,000 highest-scoring Stack Overflow records globally.

That could produce a corpus dominated by one topic.

Prefer a reasonably diverse selection across technical areas.

This does NOT need to be perfectly balanced.

A deterministic bucket/round-robin or weighted sampling approach is sufficient.

Examples of broad internal buckets could include:

```text
Linux / shell
Windows / PowerShell
networking / web protocols
server administration
databases
Python / packages
Git / developer tools
web servers
other relevant technical troubleshooting
```

Do not over-engineer this.

Record final topic/tag distributions in the manifest.

---

# 12. Do Not Apply a Hard Date Cutoff

The Stack Overflow snapshot may contain older material.

Do not reject records solely because they are old.

Old technical information can still be useful.

If there are extremely obvious obsolete-specific records and filtering them is simple, they may be excluded conservatively.

Do not build a complicated obsolescence classifier.

---

# 13. General-IT Benchmark Leakage Firewall

This is CRITICAL.

Our new:

```text
1,000-example general IT benchmark
```

must remain independent of RAG.

Do NOT remove examples because they merely cover the same topic.

For example:

```text
benchmark:
"Why does SSH say permission denied?"

corpus:
"SSH public key authentication fails after changing file permissions"
```

is legitimate.

Only actual duplicate/leakage should be removed.

Record these checks in the RAG manifest.

---

# 14. TechQA Is Allowed in RAG

Do NOT apply the old TechQA contamination firewall.

That rule belonged to our previous experimental setup where TechQA was the benchmark.

TechQA is now intentionally part of the knowledge corpus.

---

# 15. Processed Output Files

Prefer something conceptually like:

```text
data/processed/rag/
    stackoverflow_it_support.jsonl
    techqa_ibm.jsonl
    combined_corpus.jsonl
    manifest.json
    filter_stats.json
```

Exact names may follow repository conventions.

The manifest should contain:

```text
Stack Overflow source
Stack Overflow revision
TechQA source path/hash

raw counts
filtered counts
deduplicated counts

final Stack Overflow count
final TechQA count
final total count

filter configuration
quality thresholds
topic distributions

benchmark leakage checks

SHA256 for final processed files
```

All preprocessing must be reproducible.

---

# 16. Retrieval Text vs Returned Context

Do not blindly embed enormous answer text.

For Stack Overflow, use the question as the primary retrieval signal.

Preferred retrieval text:

```text
TITLE

QUESTION
```

Tags may optionally be included.

For example:

```text
Title: SSH Permission denied after copying public key

Question:
I copied my key to authorized_keys but SSH still returns
Permission denied (publickey)...
```

The accepted answer should primarily be returned as context AFTER retrieval.

Preferred context:

```text
Title: ...

Question:
...

Accepted answer:
...
```

For TechQA:

```text
Question:
...

IBM support answer:
...
```

may be used for both retrieval/context as appropriate.

Do not fabricate document titles for TechQA using an LLM.

---

# 17. Retrieval Unit

Initially treat each source Q&A pair as ONE retrieval unit.

Do NOT implement complicated chunking unless an individual record is too large to be useful.

Most support Q&A records should remain intact because:

```text
question + accepted answer
```

forms one coherent unit.

If exceptionally large records require truncation, implement deterministic token limits and report truncation statistics.

Do not split every answer into arbitrary chunks unless clearly necessary.

---

# 18. Embedding Model

Use:

```text
BAAI/bge-base-en-v1.5
```

unless the repository already contains a strong reason to use another embedding model.

Record the exact model ID and revision if practical.

Use normalized embeddings if supported.

Precompute embeddings explicitly rather than relying on opaque implicit embedding behavior inside Chroma.

This makes:

```text
batching
device selection
reproducibility
debugging
```

clearer.

Support GPU embedding when CUDA is available, with CPU fallback.

---

# 19. ChromaDB

Use ChromaDB as the vector database.

Do NOT replace it with:

```text
FAISS
BM25
Rank-BM25
Elasticsearch
```

for this version.

We deliberately want this project to demonstrate a real vector database.

Use persistent storage.

Conceptually:

```text
data/vectorstore/chroma/
```

or another repository-appropriate location.

Use cosine similarity / distance.

Store source metadata with every vector.

---

# 20. Prefer One Logical Knowledge Base

Prefer one unified Chroma collection containing both:

```text
Stack Overflow
TechQA
```

with metadata such as:

```text
source = stackoverflow
source = techqa
```

This keeps the initial architecture simple.

The retriever should still support optional metadata filtering if needed for debugging.

For example:

```python
source="techqa"
```

or:

```python
source="stackoverflow"
```

should be possible during diagnostics.

Do not force source filtering during normal retrieval.

Let similarity ranking select the appropriate source.

---

# 21. Stable Vector IDs

Chroma IDs must be deterministic.

Example:

```text
stackoverflow:12345
techqa:TRAIN_Q046
```

Running indexing twice should update/reuse the same records rather than create duplicates.

The indexing process should be safely repeatable.

---

# 22. Index Manifest / Fingerprint

Create metadata describing the built index.

Record at least:

```text
corpus SHA256
number of documents
embedding model
embedding model revision if available
embedding dimension
distance metric
Chroma collection name
creation configuration
source counts
```

If the processed corpus or embedding configuration changes, detect that the existing index is stale rather than silently using incompatible vectors.

A simple fingerprint is sufficient.

Do not over-engineer distributed index versioning.

---

# 23. Retriever

Implement a reusable retriever interface.

Conceptually:

```python
results = retriever.retrieve(
    query,
    top_k=3,
)
```

Return structured records such as:

```python
[
    {
        "id": ...,
        "source": ...,
        "score": ...,
        "title": ...,
        "question": ...,
        "answer": ...,
        "context_text": ...
    }
]
```

Do not return only raw strings.

This will make debugging and citations easier.

---

# 24. Initial Retrieval Configuration

Start simple.

Suggested initial value:

```python
RAG_TOP_K = 3
```

Make it configurable.

Do NOT implement yet:

```text
BM25 hybrid search
cross-encoder reranking
LLM reranking
query rewriting
multi-query retrieval
HyDE
agentic retrieval
```

Those can be future improvements.

For this CV project, a solid dense-retrieval pipeline is sufficient.

---

# 25. Context Construction

Build a context formatter that clearly separates retrieved records.

Conceptually:

```text
[Source 1: Stack Overflow]
Title: ...
Question: ...
Answer: ...

[Source 2: IBM TechQA]
Question: ...
Answer: ...

[Source 3: Stack Overflow]
...
```

Maintain deterministic order based on retrieval score.

Include source labels.

---

# 26. Context Token Budget

Do not allow retrieved context to overflow the model input.

The current benchmark/generation setup uses approximately:

```text
max_input_tokens = 4096
max_new_tokens = 1024
```

Reuse existing project configuration where appropriate.

Implement a context budget.

For example:

```python
RAG_MAX_CONTEXT_TOKENS = 1200
```

or another sensible value after accounting for:

```text
system prompt
user question
formatting
```

Use the actual Qwen tokenizer to measure context tokens when available.

Add highest-ranked records first until the budget is reached.

Do not truncate the user question.

If the final retrieved item only partially fits, either truncate it deterministically or skip it.

Record which policy is used.

---

# 27. RAG Prompt

Use a simple support-oriented RAG prompt.

The model should be told:

```text
Use the retrieved technical context when it is relevant.

Do not assume every retrieved passage is relevant.

If the context contains the answer, use it.

Do not invent version numbers, commands, configuration values,
product behavior, or other technical facts not supported by the
context or your reliable knowledge.

If the available information is insufficient, say so rather than
fabricating a solution.
```

Do not make the prompt extremely long.

Do not force the model to copy context verbatim.

Do not tell it that the reference answer exists.

---

# 28. Source Citations

Prefer supporting lightweight answer citations.

For example:

```text
DEX answer ...

Sources:
[1] Stack Overflow: SSH Permission denied...
[2] IBM TechQA: TRAIN_Q046
```

or inline references such as:

```text
... starting with ITM 6.30.05.00. [1]
```

Keep citations simple.

Do not require URL resolution if the source dataset does not contain a reliable URL.

If Stack Overflow URLs/IDs are available, preserve them in metadata.

The main goal is traceability:

```text
Which retrieved record caused this answer?
```

---

# 29. Base and DEX RAG Must Share the Same Retrieval

RAG comparison must remain controlled.

For a given query:

```text
Base + RAG
DEX + RAG
```

should receive the same retrieved documents and context construction rules.

Do NOT build separate indexes for base and DEX.

Do NOT use different retrieval parameters.

The only intended model difference should be:

```text
base weights only

vs.

base weights + DEX LoRA
```

---

# 30. RAG Generation Modes

The system should support at least:

```text
Base
DEX
Base + RAG
DEX + RAG
```

Prefer reusing existing model-loading code.

Do not duplicate the Qwen loading logic into multiple independent implementations if it can be shared cleanly.

Conceptually:

```python
generate(question, model="base", rag=False)

generate(question, model="dex", rag=False)

generate(question, model="base", rag=True)

generate(question, model="dex", rag=True)
```

Exact API is flexible.

---

# 31. CLI / Demo Interface

Provide a simple command for manual testing.

Something conceptually like:

```bash
python -m rag.query \
    --model dex \
    --question "Why does SSH keep returning permission denied?" \
    --top-k 3
```

and optionally:

```bash
python -m rag.query \
    --model base \
    --question "Does Linux KVM monitoring support CANDLEDATA?" \
    --top-k 3
```

Output should display:

```text
retrieved sources
similarity scores
context used
final model answer
```

A `--debug-retrieval` option would be useful.

Do not build a frontend in this phase.

---

# 32. Retrieval-Only CLI

Also provide a lightweight retrieval command that does NOT load Qwen.

Example:

```bash
python -m rag.retrieve \
    --query "CANDLEDATA Linux KVM monitoring"
    --top-k 3
```

This should display something like:

```text
1. techqa:TRAIN_Q046
   score: ...
   question: ...

2. stackoverflow:...
   score: ...
```

This is very useful for debugging retrieval independently from generation.

---

# 33. TechQA Smoke Tests

Create several deterministic retrieval smoke tests based on known TechQA facts.

Examples should include cases like:

```text
CANDLEDATA
ITM versions
WebSphere Linux ulimit
TIP fix packs
SPSS installation
```

The purpose is NOT to benchmark the model.

The purpose is to verify that IBM-specific queries retrieve IBM-specific knowledge.

For example:

```text
query:
"Which ITM version supports CANDLEDATA?"

expected:
the TechQA CANDLEDATA record appears within top-k
```

Do not hard-code every score.

Check retrieval presence/ranking reasonably.

---

# 34. Stack Overflow Smoke Tests

Also add general technical retrieval smoke tests.

Examples:

```text
SSH permission denied
Linux permissions
DNS resolution
PostgreSQL connection problems
Python package import problems
Git authentication
web server configuration
```

The expected result should be a relevant Stack Overflow record.

These may be implemented as lightweight scripts rather than brittle unit tests if exact rankings vary across library versions.

---

# 35. Retrieval Diagnostics

For debugging, report:

```text
query
document ID
source
similarity score/distance
title/question preview
```

Do not hide retrieval behavior behind the generation model.

We need to be able to distinguish:

```text
retrieval failure
```

from:

```text
generation failure despite good retrieval
```

---

# 36. Benchmark Integration

Integrate RAG with the new general-IT benchmark pipeline.

The benchmark should eventually support these four controlled experiments:

```text
1. Base Qwen
2. DEX
3. Base Qwen + RAG
4. DEX + RAG
```

All four must use exactly the same:

```text
1,000 benchmark questions
generation settings
temperature
max_new_tokens
random seed
judge model
judge prompt
metrics
```

For the two RAG variants, use the same:

```text
Chroma index
embedding model
top_k
context budget
retrieval code
```

---

# 37. Judge Prompt

Use the revised general-IT judge prompt currently defined by the project.

The judge should focus on:

```text
technical correctness
relevance
whether the answer actually solves/answers the problem
```

and should NOT penalize an answer merely for being shorter than the reference.

Do not modify the judge specifically to favor DEX or RAG.

The same judge must score all four model variants.

---

# 38. Benchmark Metrics

Preserve our current useful metrics:

```text
ROUGE-L F1
BERTScore F1
LLM-judged answer quality
abstention rate
```

If retrieval is enabled, also record lightweight RAG diagnostics such as:

```text
average retrieved documents used
average context tokens
source distribution:
    Stack Overflow
    TechQA
```

Do NOT mix these retrieval diagnostics into the LLM quality score.

---

# 39. Benchmark Result Paths

Keep experiment outputs separate.

Conceptually:

```text
results/
    baseline_general_it/
    dex_general_it/
    baseline_rag_general_it/
    dex_rag_general_it/
```

Follow existing repository conventions.

Do not overwrite old TechQA benchmark results.

---

# 40. Benchmark Resume / Fingerprinting

The existing benchmark supports checkpointing/resume.

Preserve it.

For RAG benchmarks, the fingerprint should additionally include:

```text
RAG enabled/disabled
corpus hash
Chroma/index fingerprint
embedding model
top_k
context token budget
RAG prompt hash
```

If any of those change, old partial benchmark results must not be silently resumed.

---

# 41. Retrieval Caching During Benchmarking

Do not recompute embeddings for the same benchmark questions unnecessarily.

If practical, cache retrieval results deterministically.

For example:

```text
benchmark query ID
→ retrieved document IDs
→ similarity scores
```

This also makes Base+RAG and DEX+RAG comparisons stronger because both can use exactly the same retrieved context.

A good workflow would be:

```text
1. retrieve contexts for all 1K questions once

2. save deterministic retrieval artifact

3. Base + RAG generation uses it

4. DEX + RAG generation uses the exact same artifact
```

This is preferred if it fits the current benchmark architecture cleanly.

Record the retrieval-artifact SHA256.

---

# 42. Important Experimental Control

For comparing:

```text
Base + RAG
vs
DEX + RAG
```

prefer using IDENTICAL precomputed retrieval results.

Do not independently retrieve once for Base and again for DEX if avoidable.

Retrieval does not depend on the generator model, so sharing retrieved contexts removes another source of variation.

---

# 43. Do Not Benchmark TechQA After Indexing It

After TechQA becomes part of Chroma:

```text
DEX + RAG on TechQA
```

is NOT a valid held-out evaluation.

Do not report such a number as benchmark performance.

Historical pre-RAG TechQA scores may remain documented:

```text
Base TechQA
V1 LoRA TechQA
DEX TechQA
```

but label them clearly as historical experiments performed before TechQA was incorporated into RAG.

---

# 44. README

Update the README so the architecture is understandable to someone who has never seen this project.

Explain simply:

```text
DEX
→ Qwen3.5-9B fine-tuned on general IT-support conversations

Stack Overflow RAG source
→ broad technical troubleshooting knowledge

TechQA RAG source
→ IBM / enterprise product knowledge

ChromaDB
→ vector storage and retrieval

General-IT held-out benchmark
→ compares Base, DEX, Base+RAG, DEX+RAG
```

Avoid overly academic wording.

---

# 45. Suggested Architecture

A reasonable structure would be something like:

```text
rag/
    config.py
    preprocess_stackoverflow.py
    preprocess_techqa.py
    build_corpus.py
    embeddings.py
    index.py
    retriever.py
    context.py
    pipeline.py
    query.py
    retrieve.py
```

and:

```text
data/processed/rag/
    stackoverflow_it_support.jsonl
    techqa_ibm.jsonl
    combined_corpus.jsonl
    manifest.json
    filter_stats.json
```

and a persistent Chroma directory.

This is only a suggestion.

Reuse existing project structure instead if cleaner.

Do not restructure the entire repository unnecessarily.

---

# 46. Configuration

Centralize important RAG configuration rather than scattering constants.

Include things such as:

```python
STACKOVERFLOW_MAX_RECORDS = 75_000

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

RAG_TOP_K = 3

RAG_MAX_CONTEXT_TOKENS = 1200

CHROMA_DISTANCE_METRIC = "cosine"
```

as well as:

```text
source paths
processed paths
index path
collection name
batch size
device
```

Do not hard-code machine-specific absolute paths.

---

# 47. Tests

Add tests covering at least:

```text
Stack Overflow preprocessing produces valid records

TechQA null reference answers are excluded

TechQA answerable records are preserved

stable IDs are unique

processed corpus contains both sources

exact duplicates are removed

general-IT benchmark exact prompt overlap = 0

Chroma index count matches manifest

re-indexing does not create duplicate documents

retriever returns structured results

source metadata survives retrieval

context builder respects token budget

highest-ranked contexts are used first

Base and DEX RAG use the same retrieval implementation

benchmark fingerprint includes RAG configuration

TechQA historical files/results are not overwritten
```

Tests should NOT load the full 9B model unless explicitly running an integration test.

---

# 48. Lightweight Integration Tests

Add a small integration/smoke mode that can verify:

```text
embedding model loads
Chroma opens
query embeds
retrieval returns records
context is built
```

without loading Qwen.

Optionally provide a separate generation smoke test that loads the model only when explicitly requested.

Do not make normal unit tests download multi-gigabyte models.

---

# 49. Failure Handling

Fail clearly if:

```text
processed corpus is missing
Chroma index is missing
index fingerprint does not match corpus
embedding dimension mismatches
benchmark file overlaps corpus
DEX adapter is missing for DEX mode
```

Do not silently rebuild expensive assets unless explicitly requested.

Provide useful error messages telling the user what command to run.

---

# 50. Performance

Batch Stack Overflow preprocessing and embedding efficiently.

The machine may have:

```text
RTX 5090 32 GB
```

during heavy project work.

Use GPU embedding efficiently when available.

Do not load Qwen while building embeddings.

Embedding and LLM generation are separate stages.

Avoid holding unnecessary models in VRAM simultaneously.

---

# 51. Dependency Changes

Add only necessary dependencies.

Likely examples:

```text
chromadb
sentence-transformers
```

and lightweight HTML-cleaning support if not already available.

Do not add an entire RAG framework solely for a simple retrieval pipeline.

We already use/know LangChain elsewhere, but this project does not need LangChain unless it genuinely simplifies existing code.

A direct implementation is acceptable and may be clearer for the CV project.

---

# 52. What NOT to Implement Yet

Do NOT implement:

```text
BM25
hybrid retrieval
reranking
cross encoders
query rewriting
HyDE
multi-query retrieval
agentic RAG
knowledge graphs
web search
document scraping
official vendor documentation ingestion
LLM-generated corpus summaries
```

We can add these later only if retrieval quality shows a clear need.

For the CV project, dense vector retrieval with ChromaDB is sufficient.

---

# 53. Scope Boundary

The intended final project scope is:

```text
QLoRA fine-tuned DEX
+
Stack Overflow knowledge
+
TechQA / IBM knowledge
+
ChromaDB RAG
+
controlled Base / DEX / RAG evaluation
```

Once this works well, we consider the main project complete.

Do not expand corpus collection indefinitely.

---

# 54. Final Experimental Matrix

Prepare the repository so we can eventually produce this table:

```text
                         General IT Benchmark

Base Qwen                       ?
DEX                             ?
Base Qwen + RAG                 ?
DEX + RAG                       ?
```

All four must use the exact same 1,000 held-out questions.

The most important comparisons are:

```text
Base → DEX
effect of fine-tuning

Base → Base + RAG
effect of retrieval on the base model

DEX → DEX + RAG
effect of retrieval on the specialized model

Base + RAG → DEX + RAG
whether specialization still adds value when both have knowledge access
```

#

Do not run the full 1,000-question RAG benchmark automatically.

Stop after preprocessing, indexing, implementation, smoke testing, documentation, and validation are complete.

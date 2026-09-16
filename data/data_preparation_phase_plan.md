# Dataset Downloading and Preprocessing Phase

## Important Correction — TechQA Usage

TechQA is used **only for final benchmarking**.

It is not used for:

* Fine-tuning
* Fine-tuning validation
* Hyperparameter selection
* Prompt development
* RAG development
* RAG indexing
* Any model weight updates

Therefore, **do not split TechQA into development, validation, or test subsets for this project**.

Combine all available labeled TechQA examples into one fixed benchmark dataset.

This exact full TechQA benchmark will later be passed through all four systems:

```text
1. Base model
2. Fine-tuned model
3. Base model + RAG
4. Fine-tuned model + RAG
```

All four systems must receive exactly the same TechQA examples.

---

## Agent Scope — Strict

Work **only on the dataset preparation phase described in this document**.

Do not modify or implement:

* Fine-tuning code
* Model downloading/loading
* Benchmark execution
* RAG pipeline
* Embeddings or vector databases
* Model inference
* Later project phases

Do **not** automatically continue to the next phase after completing this work.

When this phase is complete:

1. Stop all implementation work.
2. Run the relevant tests and validation.
3. Report what was completed.
4. Report any issues, assumptions, or remaining concerns.
5. Wait for explicit user permission before starting another phase.

### Code Organization

Do not implement the entire data pipeline in one large file.

Separate responsibilities into small, clear modules. For example:

```text
data/
├── download.py
├── inspect.py
├── preprocess_finetune.py
├── preprocess_techqa.py
├── preprocess_rag.py
├── validate.py
└── utils.py
```

Use only files that are actually useful. Do not create unnecessary abstractions or boilerplate.

---

## Goal

Prepare clean, validated, reproducible datasets for:

* QLoRA fine-tuning
* TechQA benchmarking
* Initial RAG testing

This phase should finish before renting GPU resources.

## 1. Download Raw Datasets

Download and preserve the original datasets without modification.

```text
data/raw/
├── finetune/
├── evaluation/
└── rag/
```

Datasets:

* `Tobi-Bueck/customer-support-tickets` → fine-tuning
* `TechQA` → benchmarking only
* `Console-AI/IT-helpdesk-synthetic-tickets` → initial RAG corpus

Never edit raw data in place.

## 2. Inspect Dataset Schemas

For each dataset:

* Print column names and data types.
* Inspect several random examples.
* Count rows.
* Check missing values.
* Check language/category distributions where available.
* Identify fields required by later stages.

Record any schema assumptions in preprocessing code.

## 3. Prepare Fine-Tuning Dataset

Filter Tobi-Bueck to useful IT-support examples.

Initial criteria:

* English only.
* Keep relevant queues such as `IT Support` and `Technical Support`.
* Require non-empty user problem and support answer.
* Remove malformed and duplicate examples.

Convert examples into:

```python
{
    "prompt": [
        {"role": "user", "content": "..."}
    ],
    "completion": [
        {"role": "assistant", "content": "..."}
    ]
}
```

Create:

```text
data/processed/finetune/
├── train.jsonl
└── validation.jsonl
```

Use a reproducible split with a fixed random seed.

The Tobi validation split is used to monitor fine-tuning behavior and generalization within the training distribution.

## 4. Prepare Full TechQA Benchmark

TechQA is an **independent benchmark only**.

Use **all available labeled TechQA examples**.

If TechQA provides separate source splits such as train/dev/eval, combine the labeled examples into one benchmark dataset because none of those examples will be used for training or development in this project.

Tasks:

* Load all available labeled TechQA examples.
* Normalize them into one common benchmark schema.
* Preserve question, expected answer, IDs, and useful metadata.
* Preserve original split metadata if available for traceability.
* Remove malformed records only when necessary.
* Deduplicate only true duplicate benchmark examples.
* Never include TechQA in fine-tuning data.
* Never include TechQA in the RAG corpus.

Output:

```text
data/processed/evaluation/
└── techqa_benchmark.jsonl
```

The processed benchmark should contain the full usable labeled TechQA corpus.

Example structure:

```python
{
    "id": "...",
    "question": "...",
    "answer": "...",
    "metadata": {
        "original_split": "..."
    }
}
```

This exact file will later be reused unchanged for all four experiments:

```text
Base
Fine-tuned
Base + RAG
Fine-tuned + RAG
```

Do not create separate TechQA validation/test sets unless the project requirements are explicitly changed later.

## 5. Prepare Initial RAG Corpus

Process `Console-AI/IT-helpdesk-synthetic-tickets`.

Tasks:

* Keep useful text and metadata.
* Remove empty or duplicate documents.
* Convert records into:

```python
{
    "id": "...",
    "text": "...",
    "metadata": {...}
}
```

Do not implement chunking, embeddings, retrieval, or a vector database in this phase.

Output:

```text
data/processed/rag/
└── documents.jsonl
```

This is only the initial RAG corpus. More technical documentation will be added later.

## 6. Validate Data

Check:

* Required fields exist.
* No empty prompt/completion pairs.
* No malformed structures.
* No unexpected duplicates.
* Tobi train and validation sets do not overlap.
* TechQA does not appear in Tobi training or validation data.
* TechQA does not appear in the RAG corpus.
* The TechQA benchmark contains all usable labeled examples.
* No TechQA examples were accidentally discarded because of original split names.
* Final dataset sizes are reasonable.

Produce a concise validation report.

For TechQA, explicitly report:

```text
Original labeled examples: ...
Final benchmark examples: ...
Removed malformed examples: ...
Removed duplicates: ...
Benchmark leakage: PASS/FAIL
```

## 7. Add Tests

Add focused preprocessing tests:

```text
tests/
├── test_finetune_data.py
├── test_evaluation_data.py
└── test_rag_data.py
```

Test:

* Filtering removes invalid rows.
* Output schemas are correct.
* Fine-tuning splits do not overlap.
* TechQA preprocessing combines all intended labeled examples.
* TechQA is not split for project benchmarking.
* TechQA does not leak into fine-tuning or RAG data.
* Preprocessing is deterministic.

## Phase Completion Criteria

```text
✓ All three raw datasets downloaded
✓ Schemas inspected
✓ Fine-tuning data filtered and split
✓ Full labeled TechQA benchmark prepared
✓ Initial RAG documents prepared
✓ Validation passes
✓ Preprocessing tests pass
```

If the previous implementation created a TechQA file containing only one original split, update the preprocessing and processed output so that the final benchmark contains **all available labeled TechQA examples**.

Once all criteria are satisfied:

**STOP. Report the results and wait for user approval. Do not begin the next phase.**

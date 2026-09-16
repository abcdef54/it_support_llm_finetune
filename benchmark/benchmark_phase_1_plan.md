Begin Phase 2: prepare the **baseline TechQA benchmarking code** for the base Qwen model.

Keep baseline-specific implementation inside:

```text
./benchmark/baseline/
```

Code that is clearly reusable by all four benchmark variants may be placed under:

```text
./benchmark/common/
```

Good shared candidates include TechQA loading and validation, metrics, judge logic, abstention detection, result schemas, and serialization. Do not extract speculative abstractions merely because they might be useful later.

Do not modify `benchmark/finetuned`, `benchmark/baseline_rag`, `benchmark/finetune_rag`, `finetune/`, or `rag/`.

## Scope

Your task is to **prepare and write the benchmark code only**.

Do **not run the actual benchmark** and do not load/download the full Qwen model on this machine. My local GPU does not have enough VRAM.

You may run lightweight checks that do not require the large model, including:

* syntax checks
* unit tests
* configuration validation
* loading a few TechQA records
* testing metric functions on synthetic examples

## Benchmark Goal

Prepare code to benchmark the **base Qwen3.5-9B model** using the complete processed TechQA benchmark:

```text
data/processed/evaluation/techqa_benchmark.jsonl
```

TechQA is benchmark-only.

Do not modify, split, sample, train on, or tune against TechQA.

The exact same TechQA benchmark and evaluation methodology must later be reused for:

```text
1. Base model
2. Fine-tuned model
3. Base model + RAG
4. Fine-tuned model + RAG
```

## Benchmark Metrics

Implement the following four metrics.

### 1. ROUGE-L F1

Purpose:

Measure lexical and structural similarity between the generated answer and the reference answer.

ROUGE-L finds the **Longest Common Subsequence (LCS)** between the generated answer and reference answer.

Calculate:

```text
precision = LCS_length / generated_answer_length

recall = LCS_length / reference_answer_length

ROUGE-L F1 =
    2 × precision × recall
    ───────────────────────
       precision + recall
```

Example:

```text
Reference:
Restart the Docker daemon and check the DNS configuration.

Prediction:
Check the DNS configuration and restart Docker.
```

The wording is not identical, but several important token sequences overlap.

Calculate ROUGE-L F1 for every TechQA example and report the mean across the entire benchmark.

Do not use ROUGE-L alone to determine answer correctness.

---

### 2. BERTScore F1

Purpose:

Measure **semantic similarity** between the generated answer and reference answer even when they use different wording.

Unlike ROUGE, BERTScore compares contextual token embeddings rather than exact token overlap.

Conceptually:

```text
generated tokens
      ↓ embeddings

reference tokens
      ↓ embeddings

cosine similarity between token embeddings
      ↓

semantic precision
semantic recall
      ↓

BERTScore F1
```

The final score is:

```text
BERTScore F1 =
    2 × semantic_precision × semantic_recall
    ─────────────────────────────────────────
       semantic_precision + semantic_recall
```

Calculate BERTScore F1 for every example and report the mean.

Use one fixed BERTScore model/configuration for all four experiments.

Do not silently change the semantic evaluation model between benchmark runs.

---

### 3. LLM-Judge Correctness

Purpose:

Measure whether the generated answer is actually **technically correct and useful**, which lexical metrics cannot reliably determine.

Use a fixed judge model and fixed evaluation prompt for all four experiments.

The judge receives:

```text
Question
Reference answer
Generated answer
```

It should evaluate technical correctness using this scale:

```text
0 = Completely incorrect or irrelevant
1 = Mostly incorrect; major technical problems
2 = Partially correct; important information missing or incorrect
3 = Mostly correct; minor problems only
4 = Fully correct and relevant
```

Save the raw score for every example.

Calculate:

```text
mean_correctness =
    sum(correctness_scores)
    ───────────────────────
       number_of_examples
```

Also calculate a normalized percentage:

```text
correctness_percentage =
    mean_correctness / 4 × 100
```

Example:

```text
Mean judge score: 3.20 / 4

Normalized correctness:
3.20 / 4 × 100 = 80%
```

The judge prompt, judge model, temperature, and scoring rubric must remain identical across all four experiments.

Save the judge's raw score and optionally a short justification for later error analysis.

Do not use benchmark results to modify TechQA itself.

---

### 4. Abstention Rate

Purpose:

Measure how often the model refuses, declines, or states that it cannot provide an answer.

Examples include responses such as:

```text
"I don't know."

"I don't have enough information to answer."

"I cannot determine the solution from the provided information."
```

For each generated answer classify:

```text
abstained = 1
answered  = 0
```

Then calculate:

```text
abstention_rate =
    number_of_abstentions
    ────────────────────── × 100
    total_examples
```

Example:

```text
70 abstentions
1400 TechQA examples

70 / 1400 × 100 = 5%
```

Keep the abstention detection logic consistent across all experiments.

Save the per-example abstention flag so results can later be inspected manually.

Abstention rate is not inherently good or bad. It helps explain model behavior alongside correctness.

---

## Final Metric Summary

The benchmark summary should contain at least:

```json
{
    "rouge_l_f1": 0.0,
    "bertscore_f1": 0.0,
    "llm_judge_mean": 0.0,
    "llm_judge_correctness_percent": 0.0,
    "abstention_rate": 0.0,
    "total_examples": 0
}
```

All aggregate metrics must be calculated over the same complete TechQA benchmark.

## Raw Predictions

Preserve per-example results.

Each result should contain enough information for later analysis:

```python
{
    "id": "...",
    "question": "...",
    "reference_answer": "...",
    "generated_answer": "...",

    "metrics": {
        "rouge_l_f1": 0.0,
        "bertscore_f1": 0.0,
        "judge_score": 0,
        "judge_reason": "...",
        "abstained": False
    }
}
```

Do not discard raw predictions after calculating aggregate metrics.

## Code Organization

Do not put everything into one file.

Use a small modular structure such as:

```text
benchmark/
├── common/
│   ├── dataset.py
│   ├── metrics.py
│   ├── judge.py
│   └── schemas.py
└── baseline/
    ├── config.py
    ├── model.py
    └── run_benchmark.py
```

Only create files that are actually useful.

Suggested responsibilities:

* `common/dataset.py` → TechQA loading and validation
* `common/metrics.py` → ROUGE-L, BERTScore, abstention and aggregation
* `common/judge.py` → fixed LLM-judge prompt and scoring
* `common/schemas.py` → shared result schemas and serialization, if needed
* `baseline/config.py` → base model, generation, metric and path configuration
* `baseline/model.py` → base model/processor loading and generation
* `baseline/run_benchmark.py` → baseline benchmark orchestration

## Reproducibility

Centralize and explicitly define:

* model identifier
* generation mode
* temperature
* `max_new_tokens`
* batch size
* random seed
* BERTScore model/config
* LLM judge model
* LLM judge prompt
* judge temperature

These settings must remain consistent across later experiments.

For baseline generation, prefer deterministic generation unless there is a documented reason otherwise.

## Outputs

Prepare output files under:

```text
results/base/
├── predictions.jsonl
└── metrics.json
```

Do not overwrite outputs belonging to other experiments.

## Tests

Add lightweight tests where useful.

Test:

* TechQA loading
* expected schema
* malformed/empty records
* ROUGE-L calculation on known examples
* BERTScore integration without running the full benchmark
* judge-output parsing
* abstention detection
* metric aggregation
* result serialization
* deterministic configuration

Tests must not load Qwen3.5-9B.

## Important Constraints

Do not:

* run the actual benchmark
* download/load the full Qwen model
* fine-tune anything
* modify TechQA
* split TechQA
* implement RAG
* begin another project phase

## Completion

When the benchmark implementation is ready:

1. Run only lightweight tests/checks.
2. Report files created or changed.
3. Explain the benchmark flow briefly.
4. Report assumptions or unresolved issues.
5. Give me the exact command to run later on the RTX 5090.
6. Stop.

Do not execute the actual benchmark and do not automatically continue to the next phase.

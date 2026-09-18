# IT Support LLM Fine-Tuning

DEX (Diagnostic EXpert) is Qwen3.5-9B fine-tuned on general IT-support
conversations. Fine-tuning teaches its support behavior; retrieval supplies
technical knowledge from Stack Overflow and IBM TechQA. A persistent ChromaDB
collection stores BGE embeddings for both sources.

The held-out 1,000-question general-IT benchmark compares Base, DEX,
Base + RAG, and DEX + RAG using the same questions, generation settings,
base-model judge, and metrics. The RAG variants consume one identical saved
retrieval/context artifact.

## TechQA data preparation

Activate the existing virtual environment, then run:

```bash
uv pip install -r requirements.txt
python -m data.prepare
python -m unittest discover -s tests -v
```

The command downloads pinned TechQA data, inspects its schema, prepares the
historical benchmark file, and validates its records. To reuse the raw files,
pass `--skip-download`.

Generated outputs:

```text
data/processed/
├── download_manifest.json
├── schema_report.json
├── validation_report.json
├── evaluation/techqa_benchmark.jsonl
```

All usable labeled TechQA examples from its original train and development
splits remain in one historical benchmark file; their original split is metadata
only. The RAG commands below build the current Stack Overflow + answerable
TechQA corpus. TechQA is never used for SFT.

## DEX fine-tuning data (V2)

The DEX (Diagnostic EXpert) experiment uses the pinned
`benjaminmacklin/IT_Support_V2` source. The preparation command downloads the
source through Hugging Face `datasets`, inspects its schema, filters and
deduplicates it, then writes 30,000 selected examples. The revised layout is
27,000 train, 2,000 validation for `eval_loss`/checkpoint selection, and 1,000
general-IT benchmark examples held out until final evaluation. It uses the
Qwen tokenizer for length checks but does not load model weights or train:

```bash
.venv/bin/python -m data.preprocess_finetune_v2 --project-root .
```

On a machine with the pinned source and tokenizer already cached under
`data/raw/finetune_v2/`, use `--skip-download` to regenerate the same output
offline. To reproduce only the 2K/1K split from the archived original 3K file,
without downloading or changing the 27K train file, run:

```bash
.venv/bin/python -m data.split_dex_validation --project-root .
```

The original 3K validation set is preserved as
`data/processed/finetune_v2/validation_original_3000.jsonl`. The revised
`validation.jsonl` contains only 2K; the 1K test is
`data/processed/benchmark/dex_general_it_benchmark.jsonl`. Their hashes,
topic distributions, overlap audit, source revision, and split seed are in the
DEX and benchmark manifests. Review the 100-record `audit_sample.jsonl` before
spending GPU time; deterministic filtering cannot certify every synthetic
answer's factual accuracy. TechQA is used only as an exact-contamination
firewall during DEX preprocessing.

The old DEX adapter was selected using all 3K original validation examples,
so its scores on the new 1K subset must **not** be called held-out. Retrain a
new adapter from scratch on the unchanged 27K train file and only the new 2K
validation file. Never use the 1K benchmark for training, checkpoint or prompt
selection, or tuning.

Validate the prepared data and configuration without loading Qwen:

```bash
.venv/bin/python -m finetune.train --project-root . --experiment dex_v2 --check
.venv/bin/python -m unittest data.test_finetune_v2 finetune.test_dex benchmark.test_general_it_protocol -q
```

When separately authorized and on a suitable GPU, the **retraining** command is:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune.train --project-root . --experiment dex_v2
```

It writes a **new** adapter directory at
`models/qwen3.5-9b-it-support-dex-v2-qlora/`. `--experiment dex` still identifies the earlier 3K
validation layout for historical reproducibility. QLoRA hyperparameters are
unchanged: one epoch, batch 8, accumulation 1, and the existing LoRA settings.
Training evaluates validation loss at each configured save interval, restores
the lowest-loss checkpoint at the end, and saves that adapter as the final
artifact. The selected checkpoint and loss are recorded in `run_metadata.json`.

## Baseline benchmark batching

On the RTX 5090, run the small 16-example throughput probe before
the complete benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmark.baseline.probe_batching --project-root .
```

It checks answer and judge outputs at batch sizes 1, 2, 4, 8, and 16, and prints
elapsed time, examples per second, and peak allocated VRAM. This historical
throughput probe uses TechQA prompts but does not write benchmark results. The
configured answer and judge batch sizes are in
`benchmark/baseline/config.py`: answers use 16, while judging uses 1. On the
16-example 5090 probe, answer batch 16 was fastest and used 19.33 GiB peak
allocated VRAM. Judge batches above 1 changed a score on one fixed judge
prompt, so judging remains unbatched to preserve the measured scoring behavior.
The probe still reports differences for every candidate size, but its final
configuration check applies only to the selected answer and judge sizes.

If outputs differ, inspect the first two examples without running the full
probe again:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmark.baseline.probe_batching --project-root . --count 2 --answer-token-limit 32 --diagnose
```

This reports the first changed text position, first generated token IDs,
first-token logit differences, and whether judge scores changed.

The primary general-IT baseline benchmark remains a separate command:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmark.baseline.run_benchmark --project-root .
```

It uses the new 1K benchmark and writes to `results/base_general_it/`.
For the historical 902-example TechQA benchmark, add `--benchmark techqa`;
those results remain under `results/base/`. Metrics JSON records the benchmark
path, SHA-256, and example count; resume rejects a changed benchmark file.

If a judge response is invalid, both benchmark variants retry that same prompt
once with a 1,024-token output limit (the usual limit remains 256). The metrics
JSON records `judge_retries` and both limits. If both outputs start with the
same valid score but the explanation never forms valid JSON, that score is
accepted and labeled `judge_recovered` in the prediction; `judge_recovered_scores`
counts these cases in the metrics JSON. Conflicting or missing scores still stop
the run with the benchmark item ID. No explanation is invented from truncated text.

Progress is saved after every answer batch and judge batch in the variant's
`results/.../progress.sqlite3` file. If a run stops, rerun the same command with
`--resume`; it checks the dataset and configuration before reusing saved work.
The progress file is removed after results are written successfully. Runs made
before this checkpoint change cannot be resumed. If a final output file was
partially written during a failure, use both `--resume --overwrite` to finish.
If BERTScore fails after judging, `--resume` skips both Qwen stages and retries
only metric calculation. BERTScore caps DeBERTa tokenization at the model's
supported position limit when its tokenizer reports an undefined huge limit.

## Fine-tuned benchmark (no RAG)

After DEX V2 retraining, the fine-tuned run uses the **same 1K general-IT
benchmark** as the baseline. It attaches the new adapter to the same BF16 base
checkpoint only while generating answers, then loads a fresh base checkpoint
for judging. Generation settings, judge rubric, ROUGE-L, BERTScore, abstention,
and retry behavior are unchanged. The judge score is reference-grounded answer
quality on a 0–4 scale, not a binary guarantee of factual correctness. Use a
PEFT version compatible with the saved adapter; inference disables automatic
FP32 adapter casting.

Run lightweight tests without loading Qwen:

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m unittest benchmark.finetuned.test_benchmark benchmark.test_general_it_protocol -q
```

On the RTX 5090, first verify the new adapter and fresh base judge with a
synthetic prompt. This does not read benchmark data or write results:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmark.finetuned.run_benchmark --project-root . --experiment dex_v2 --smoke
```

If that passes, the full fine-tuned benchmark command is:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmark.finetuned.run_benchmark --project-root . --experiment dex_v2
```

It writes to `results/finetuned_dex_v2/`, separate from the baseline and all
historical results. `--experiment dex` still selects the earlier DEX adapter
and TechQA results. Existing outputs are protected unless
`--overwrite` is supplied; do not mix results from different configurations.
The DEX V2 benchmark must wait until the new adapter exists.

## RAG: Stack Overflow + IBM TechQA

The current policy is defined in `rag/corpus/rag_plan.md`. TechQA is now
knowledge available to retrieval. Its historical files/results are preserved,
but TechQA + RAG is **not** held-out evaluation and the benchmark commands
reject that combination. The new general-IT benchmark remains held out.

The pipeline uses the pinned [Stack Overflow Q&A dataset](https://huggingface.co/datasets/krylodar/StackOverFlowQA)
and existing TechQA data. It scans the full Stack Overflow snapshot, keeps
nonempty accepted-answer pairs with question score >= 0 and answer score >= 1
when those fields exist, and selects up to 75,000 records across eight support
topics using deterministic hash sampling and round-robin selection. Each topic
has a cap; underfilled topics are not padded with lower-quality records.
Generic Python questions require package/environment/import relevance.
There is no date cutoff. Exact duplicate questions and exact matches to
general-IT benchmark prompts or reference answers are excluded. Reading the
benchmark for this exclusion audit never adds its content to the corpus.
This is an exact-match firewall, not a semantic near-duplicate guarantee.

HTML cleaning retains commands and code indentation. Each Q&A is one retrieval
unit. Stack Overflow retrieval text contains the title/question; its accepted
answer is returned after retrieval. TechQA uses its question for retrieval and
returns the IBM answer. Null answers are excluded. Normalized embeddings use
the pinned [BGE base English v1.5 model](https://huggingface.co/BAAI/bge-base-en-v1.5)
with its query instruction. BGE input is capped at 512 tokens; truncation counts
are reported, while the stored Q&A remains complete.

Install dependencies and build the corpus/index (no Qwen weights are loaded):

```bash
uv pip install -r requirements.txt
.venv/bin/python -m rag.build_corpus --project-root . --download
.venv/bin/python -m rag.index --project-root . --device cuda --batch-size 32
.venv/bin/python -m rag.smoke --project-root . --device cuda
```

Use `python -m rag.build_corpus --validate` and `python -m rag.index --check`
to validate existing assets without recomputing them or loading any model.

The raw Stack Overflow download is about 4.3 GB. Corpus preparation scans it
once; omit `--download` to reuse it. Embedding defaults to CUDA when available,
with CPU fallback (`--device cpu`). CUDA uses FP16 BGE inference, CPU uses FP32;
the output vectors are normalized/stored as FP32. Index metadata records the
precision policy, device, batch size, vector hash and library versions.
Only the small embedding model is loaded during indexing, never Qwen.
On another machine, regenerate the large corpus and index with these commands;
they are intentionally excluded from Git. Small manifests/reports are tracked.

Outputs:

```text
data/processed/rag/
    stackoverflow_it_support.jsonl
    techqa_ibm.jsonl
    combined_corpus.jsonl
    manifest.json
    filter_stats.json
    retrieval_smoke_report.json
    general_it_contexts.json       # created explicitly below; ignored by Git
data/vectorstore/chroma/
    manifest.json
    ... persistent Chroma files ...
```

The RAG manifest records source revisions,
hashes, filtering/deduplication counts, topic/tag distributions, and the leakage
audit. Indexing reuses a matching complete index; interrupted builds can be
rerun with the same command using stable IDs. A changed corpus or embedding
configuration fails clearly. Use `python -m rag.index --rebuild` only when you
intend to replace the derived RAG collection. Historical datasets/results are
not replaced by this operation.

Retrieval alone works without Qwen:

```bash
.venv/bin/python -m rag.retrieve --query "Which ITM version supports CANDLEDATA?" --top-k 5
.venv/bin/python -m rag.retrieve --query "SSH permission denied" --source stackoverflow
```

Results include source IDs, similarity scores, questions, answers and metadata.
The smoke command checks known IBM facts and general technical queries against
the real index, then verifies context construction using the Qwen **tokenizer
only**. These are retrieval diagnostics, not benchmark quality scores.

Manual generation on a machine suitable for Qwen:

```bash
.venv/bin/python -m rag.query --model base --question "Why does SSH return permission denied?" --debug-retrieval
.venv/bin/python -m rag.query --model dex --question "Which ITM version supports CANDLEDATA?"
```

Both default to RAG. Add `--no-rag` for the corresponding plain Base/DEX mode.
DEX requires the `dex_v2` adapter. BGE is released before Qwen loads.
Answers may cite numbered sources; saved retrieval metadata records the actual
passages supplied, but citations do not prove that a passage caused an answer.

## Controlled RAG benchmarks

Prepare the shared contexts once, using only retrieval and the Qwen tokenizer:

```bash
.venv/bin/python -m rag.prepare_benchmark --project-root . --device cuda
```

This saves top-5 retrieval results and final formatted contexts for all 1,000
questions. It does not generate or judge answers. Both RAG models use this
exact file. Context is capped at 2,400 Qwen tokens and the complete chat at
4,096 tokens. Higher-ranked records are considered first; whole records that
do not fit are skipped. User questions are never truncated: an overlong
question fails before generation. Sources are labeled and the prompt tells
the model to ignore irrelevant passages and abstain when information is
insufficient. No reference answers are placed in the retrieval artifact.

Only when ready to run the full GPU experiments:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmark.baseline.run_benchmark --project-root . --rag
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmark.finetuned.run_benchmark --project-root . --experiment dex_v2 --rag
```

| Experiment | Result directory |
| --- | --- |
| Base | `results/base_general_it/` |
| DEX | `results/finetuned_dex_v2/` |
| Base + RAG | `results/base_rag_general_it/` |
| DEX + RAG | `results/finetuned_dex_v2_rag/` |

The existing `--resume` behavior applies. RAG fingerprints include corpus,
index/vector configuration, embedding revision, top-k, context budget, prompt,
benchmark hash, and the shared artifact hash. Changed inputs reject resume.
Regenerate a stale artifact explicitly using `rag.prepare_benchmark --overwrite`.
Retrieval does not run again during generation or judging. The same untouched
base judge and metrics evaluate all four variants. Predictions also store
retrieved/used IDs, source scores, context and token counts; metrics include
average documents used, average context tokens, and source counts separately
from answer quality.

Lightweight tests (no model downloads or Qwen loading):

```bash
.venv/bin/python -m unittest rag.test_rag benchmark.finetuned.test_benchmark benchmark.test_general_it_protocol -q
```

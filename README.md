# IT Support LLM Fine-Tuning

The project includes dataset preparation, QLoRA fine-tuning setup, and baseline
and fine-tuned benchmark pipelines.

## Prepare the datasets

Activate the existing virtual environment, then run:

```bash
uv pip install -r requirements.txt
python -m data.prepare
python -m unittest discover -s tests -v
```

The first command downloads pinned Hugging Face revisions, preserves them under
`data/raw/`, inspects the source schemas, creates the processed JSONL files, and
validates duplicates, split overlap, schemas, and exact TechQA leakage. To reuse
raw files already downloaded, pass `--skip-download`.

Generated outputs:

```text
data/processed/
├── download_manifest.json
├── schema_report.json
├── validation_report.json
├── finetune/train.jsonl
├── finetune/validation.jsonl
├── evaluation/techqa_benchmark.jsonl
└── rag/documents.jsonl
```

All usable labeled TechQA examples from its original train and development
splits remain in one historical benchmark file; their original split is metadata
only. TechQA is not used for SFT or indexed by RAG in this phase. Any later use
as a RAG source requires a separate implementation and leakage review.

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
`models/qwen3.5-9b-it-support-dex-v2-qlora/`. The earlier DEX adapter and V1
adapter are preserved. `--experiment dex` still identifies the earlier 3K
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
historical results. `--experiment v1` and `--experiment dex` still select their
old adapters and TechQA results. Existing outputs are protected unless
`--overwrite` is supplied; do not mix results from different configurations.
The DEX V2 benchmark must wait until the new adapter exists.

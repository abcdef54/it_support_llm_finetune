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
splits are combined into one benchmark-only file. The original split is retained
only as metadata. TechQA is never used for training, tuning, or RAG indexing.

## DEX fine-tuning data (V2)

The DEX (Diagnostic EXpert) experiment uses the pinned
`benjaminmacklin/IT_Support_V2` source. The preparation command downloads the
source through Hugging Face `datasets`, inspects its schema, filters and
deduplicates it, then writes 30,000 selected examples (27,000 train and 3,000
validation). It uses the Qwen tokenizer for length checks but does not load
model weights or start training:

```bash
.venv/bin/python -m data.preprocess_finetune_v2 --project-root .
```

On a machine with the pinned source and tokenizer already cached under
`data/raw/finetune_v2/`, use `--skip-download` to regenerate the same output
offline. The generated files are under `data/processed/finetune_v2/`:
`train.jsonl`, `validation.jsonl`, `manifest.json`, `filter_stats.json`,
`source_inspection.json`, `audit_sample.jsonl`, `rejected_sample.jsonl`, and
`SOURCE_LICENSE.txt`. Review the 100-record audit sample before spending GPU
time; deterministic filtering cannot certify the factual accuracy of every
synthetic troubleshooting answer. TechQA is used only for an exact-overlap
firewall, never to choose or tune training examples.

Validate the prepared data and configuration without loading Qwen:

```bash
.venv/bin/python -m finetune.train --project-root . --experiment dex --check
.venv/bin/python -m unittest data.test_finetune_v2 finetune.test_dex -q
```

When separately authorized and on a suitable GPU, the DEX training command is:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune.train --project-root . --experiment dex
```

It writes a **new** adapter directory at
`models/qwen3.5-9b-it-support-dex-qlora/`; V1 data and the existing
`models/qwen3.5-9b-it-support-qlora/` adapter are preserved. The existing QLoRA
hyperparameters are unchanged. Use `--experiment v1` to select V1 explicitly.
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
elapsed time, examples per second, and peak allocated VRAM. It does not write
benchmark results. The configured answer and judge batch sizes are in
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

The complete benchmark remains a separate command:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmark.baseline.run_benchmark --project-root .
```

If a judge response is invalid, both benchmark variants retry that same prompt
once with a 1,024-token output limit (the usual limit remains 256). The metrics
JSON records `judge_retries` and both limits. If both outputs start with the
same valid score but the explanation never forms valid JSON, that score is
accepted and labeled `judge_recovered` in the prediction; `judge_recovered_scores`
counts these cases in the metrics JSON. Conflicting or missing scores still stop
the run with the TechQA ID. No explanation is invented from truncated text.

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

The fine-tuned run uses the same 902-example TechQA file, prompt, BF16 base
checkpoint, batched generation, judge, and metrics as the baseline. It attaches
`models/qwen3.5-9b-it-support-qlora` only while generating answers, then unloads
that model and loads a fresh base checkpoint for judging. The adapter was saved
with PEFT 0.21.0; use PEFT 0.21.0 or newer in the benchmark environment. Its
saved BF16 weights are loaded without PEFT's automatic FP32 adapter cast.

Run lightweight tests without loading Qwen:

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m unittest benchmark.finetuned.test_benchmark -q
```

On the RTX 5090, first verify the real adapter and fresh base judge with a
synthetic prompt. This does not read TechQA or write results:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmark.finetuned.run_benchmark --project-root . --smoke
```

If that passes, the full fine-tuned benchmark command is:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmark.finetuned.run_benchmark --project-root .
```

It writes `results/finetuned/predictions.jsonl` and
`results/finetuned/metrics.json` without replacing `results/base/`. Existing
fine-tuned results are protected unless `--overwrite` is supplied. Do not mix
results from different batch-size settings in a controlled comparison.

After DEX training is complete, choose its adapter with `--experiment dex` on
the same fine-tuned benchmark command (or synthetic `--smoke` check). DEX writes
to `results/finetuned_dex/` and leaves V1 results untouched. The TechQA data,
BF16 base checkpoint, answer-generation settings, base-model judge, prompts,
metrics, and resume behavior remain shared and unchanged. Do not run the DEX
benchmark until its adapter exists.

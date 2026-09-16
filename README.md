# IT Support LLM Fine-Tuning

The project includes dataset preparation and the baseline benchmark pipeline.

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
JSON records `judge_retries` and both limits. A second invalid response stops
the run with the TechQA ID and an output preview including its end.

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

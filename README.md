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

On the RTX 5090, run the small 8-example equivalence and throughput probe before
the complete benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmark.baseline.probe_batching --project-root .
```

It checks answer and judge outputs at batch sizes 1, 2, 4, and 8, and prints
elapsed time, examples per second, and peak allocated VRAM. It does not write
benchmark results. The configured answer and judge batch sizes are in
`benchmark/baseline/config.py`; their initial values of 4 are provisional until
the RTX 5090 probe is run. The complete benchmark remains a separate command:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmark.baseline.run_benchmark --project-root .
```

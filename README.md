# IT Support LLM Fine-Tuning

The current implementation covers only the dataset-preparation phase in
[`data/data_preparation_phase_plan.md`](data/data_preparation_phase_plan.md).

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

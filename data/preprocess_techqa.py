from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.utils import clean_text, join_question, text_key, write_json, write_jsonl


SOURCE_FILES = {"train": "training_Q_A.json", "dev": "dev_Q_A.json"}


def normalize_record(row: dict, original_split: str) -> dict:
    answerable = row.get("ANSWERABLE") == "Y"
    return {
        "id": clean_text(row.get("QUESTION_ID")),
        "question": join_question(row.get("QUESTION_TITLE"), row.get("QUESTION_TEXT")),
        "answer": clean_text(row.get("ANSWER")) if answerable else None,
        "metadata": {
            "original_split": original_split,
            "answerable": answerable,
            "question_title": clean_text(row.get("QUESTION_TITLE")),
            "question_text": clean_text(row.get("QUESTION_TEXT")),
            "document_id": clean_text(row.get("DOCUMENT")) if answerable else None,
            "start_offset": int(row["START_OFFSET"]) if answerable else None,
            "end_offset": int(row["END_OFFSET"]) if answerable else None,
            "candidate_document_ids": row.get("DOC_IDS") or [],
        },
    }


def prepare_rows(source_rows: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    records, seen = [], {}
    malformed, duplicates = [], []
    for original_split, rows in source_rows.items():
        for row in rows:
            record = normalize_record(row, original_split)
            if (
                not record["id"]
                or not record["question"]
                or (record["metadata"]["answerable"] and not record["answer"])
            ):
                malformed.append(record["id"] or "<missing-id>")
                continue
            key = (text_key(record["question"]), text_key(record["answer"]))
            if key in seen:
                duplicates.append({"removed_id": record["id"], "retained_id": seen[key]})
                continue
            seen[key] = record["id"]
            records.append(record)

    stats = {
        "policy": "benchmark_only",
        "original_labeled_examples": sum(map(len, source_rows.values())),
        "source_counts": {split: len(rows) for split, rows in source_rows.items()},
        "final_benchmark_examples": len(records),
        "final_by_original_split": {
            split: sum(record["metadata"]["original_split"] == split for record in records)
            for split in source_rows
        },
        "removed_malformed": len(malformed),
        "removed_malformed_ids": malformed,
        "removed_duplicates": len(duplicates),
        "removed_duplicate_ids": duplicates,
        "answerable": sum(record["metadata"]["answerable"] for record in records),
        "unanswerable": sum(not record["metadata"]["answerable"] for record in records),
        "deduplication_key": "normalized question and answer",
    }
    return records, stats


def run(project_root: Path) -> dict:
    source_dir = project_root / "data" / "raw" / "evaluation" / "TechQA" / "training_and_dev"
    source_rows = {
        split: json.loads((source_dir / filename).read_text(encoding="utf-8"))
        for split, filename in SOURCE_FILES.items()
    }
    records, stats = prepare_rows(source_rows)
    output = project_root / "data" / "processed" / "evaluation" / "techqa_benchmark.jsonl"
    write_jsonl(output, records)
    stats["source_files"] = {
        split: str((source_dir / filename).relative_to(project_root))
        for split, filename in SOURCE_FILES.items()
    }
    stats["output_file"] = str(output.relative_to(project_root))
    stats["usage_restrictions"] = [
        "No fine-tuning or weight updates",
        "No hyperparameter or prompt tuning",
        "No RAG development or indexing",
        "Use unchanged for all four final experiments",
    ]
    write_json(output.parent / "preprocessing_report.json", stats)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare one full benchmark-only TechQA dataset.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), indent=2))

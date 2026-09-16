from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from data.utils import clean_text, join_question, text_key, write_json, write_jsonl


SOURCE_FILE = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
ALLOWED_QUEUES = {"IT Support", "Technical Support", "Service Outages and Maintenance"}


def prepare_rows(rows: list[dict], seed: int = 42, validation_ratio: float = 0.1) -> tuple[list[dict], list[dict], dict]:
    records, seen = [], set()
    stats = {
        "input": len(rows),
        "filtered": 0,
        "duplicates": 0,
        "exclusion_reasons": {
            "non_english": 0,
            "queue_not_allowed": 0,
            "prompt_or_answer_under_20_characters": 0,
            "duplicate_prompt": 0,
        },
    }
    for row in rows:
        if clean_text(row.get("language")).casefold() != "en":
            stats["filtered"] += 1
            stats["exclusion_reasons"]["non_english"] += 1
            continue
        if clean_text(row.get("queue")) not in ALLOWED_QUEUES:
            stats["filtered"] += 1
            stats["exclusion_reasons"]["queue_not_allowed"] += 1
            continue
        prompt = join_question(row.get("subject"), row.get("body"))
        answer = clean_text(row.get("answer"))
        if len(prompt) < 20 or len(answer) < 20:
            stats["filtered"] += 1
            stats["exclusion_reasons"]["prompt_or_answer_under_20_characters"] += 1
            continue
        key = text_key(prompt)
        if key in seen:
            stats["duplicates"] += 1
            stats["exclusion_reasons"]["duplicate_prompt"] += 1
            continue
        seen.add(key)
        records.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "completion": [{"role": "assistant", "content": answer}],
            }
        )

    random.Random(seed).shuffle(records)
    validation_size = max(1, round(len(records) * validation_ratio)) if records else 0
    validation = records[:validation_size]
    train = records[validation_size:]
    stats.update(
        {
            "kept": len(records),
            "train": len(train),
            "validation": len(validation),
            "validation_ratio": validation_ratio,
            "seed": seed,
            "filter_criteria": {
                "language": "en",
                "allowed_queues": sorted(ALLOWED_QUEUES),
                "minimum_prompt_characters": 20,
                "minimum_answer_characters": 20,
                "deduplication_key": "normalized prompt text",
            },
        }
    )
    return train, validation, stats


def run(project_root: Path, seed: int = 42) -> dict:
    source = project_root / "data" / "raw" / "finetune" / SOURCE_FILE
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train, validation, stats = prepare_rows(rows, seed=seed)
    output = project_root / "data" / "processed" / "finetune"
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", validation)
    stats["source_file"] = str(source.relative_to(project_root))
    stats["output_files"] = [
        str((output / "train.jsonl").relative_to(project_root)),
        str((output / "validation.jsonl").relative_to(project_root)),
    ]
    write_json(output / "preprocessing_report.json", stats)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare IT-support fine-tuning data.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve(), args.seed), indent=2))

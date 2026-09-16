from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from data.utils import clean_text, join_question, text_key, write_json, write_jsonl


def prepare_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    records, seen_ids, seen_text = [], set(), set()
    stats = {
        "input": len(rows),
        "invalid": 0,
        "duplicates": 0,
        "exclusion_reasons": {"missing_id_or_text_under_20_characters": 0, "duplicate_id": 0, "duplicate_text": 0},
    }
    for row in rows:
        record_id = clean_text(row.get("id"))
        text = join_question(row.get("subject"), row.get("description"))
        key = text_key(text)
        if not record_id or len(text) < 20:
            stats["invalid"] += 1
            stats["exclusion_reasons"]["missing_id_or_text_under_20_characters"] += 1
            continue
        if record_id in seen_ids:
            stats["duplicates"] += 1
            stats["exclusion_reasons"]["duplicate_id"] += 1
            continue
        if key in seen_text:
            stats["duplicates"] += 1
            stats["exclusion_reasons"]["duplicate_text"] += 1
            continue
        seen_ids.add(record_id)
        seen_text.add(key)
        records.append(
            {
                "id": record_id,
                "text": text,
                "metadata": {
                    "subject": clean_text(row.get("subject")),
                    "category": clean_text(row.get("category")),
                    "priority": clean_text(row.get("priority")),
                    "created_at": clean_text(row.get("createdAt")),
                },
            }
        )
    stats["documents"] = len(records)
    return records, stats


def run(project_root: Path) -> dict:
    source = project_root / "data" / "raw" / "rag" / "tickets.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records, stats = prepare_rows(rows)
    output = project_root / "data" / "processed" / "rag" / "documents.jsonl"
    write_jsonl(output, records)
    stats["source_file"] = str(source.relative_to(project_root))
    stats["output_file"] = str(output.relative_to(project_root))
    stats["metadata_fields"] = ["subject", "category", "priority", "created_at"]
    stats["excluded_metadata_fields"] = ["requesterEmail"]
    write_json(output.parent / "preprocessing_report.json", stats)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare initial RAG documents without chunking or embeddings.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), indent=2))

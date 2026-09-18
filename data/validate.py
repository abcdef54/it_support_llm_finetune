"""Validate the prepared TechQA file used by RAG and historical benchmarks.

Current Stack Overflow + TechQA corpus validation lives in rag.build_corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.utils import read_jsonl, text_key, write_json


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def output_details(path: Path, records: int, project_root: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.relative_to(project_root)),
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def validate_datasets(project_root: Path) -> dict:
    processed = project_root / "data" / "processed"
    preprocessing = {"evaluation": load_json(processed / "evaluation" / "preprocessing_report.json")}
    techqa = read_jsonl(processed / "evaluation" / "techqa_benchmark.jsonl")
    errors = []

    valid_techqa = [
        row for row in techqa
        if isinstance(row, dict)
        and row.get("id")
        and row.get("question")
        and "answer" in row
        and isinstance(row.get("metadata"), dict)
        and isinstance(row["metadata"].get("answerable"), bool)
        and row["metadata"].get("original_split") in {"train", "dev"}
        and (not row["metadata"]["answerable"] or bool(row["answer"]))
    ]
    if len(valid_techqa) != len(techqa):
        errors.append("TechQA: malformed record")
    duplicates = {
        "techqa_ids": len(valid_techqa) - len({row["id"] for row in valid_techqa}),
        "techqa_question_answer_pairs": len(valid_techqa) - len({
            (text_key(row["question"]), text_key(row["answer"])) for row in valid_techqa
        }),
    }
    duplicates_passed = not any(duplicates.values())
    if not duplicates_passed:
        errors.append("unexpected duplicates remain")

    techqa_stats = preprocessing.get("evaluation", {})
    sizes = {"techqa_benchmark": len(techqa)}
    expected_techqa = techqa_stats.get("final_benchmark_examples")
    sizes_passed = len(techqa) == expected_techqa
    if not sizes_passed:
        errors.append("one or more final dataset sizes are unreasonable")

    schema_passed = len(valid_techqa) == len(techqa)
    output_paths = {
        "TechQA benchmark": processed / "evaluation" / "techqa_benchmark.jsonl",
    }
    output_counts = {"TechQA benchmark": len(techqa)}
    actual_split_counts = {
        split: sum(row["metadata"]["original_split"] == split for row in valid_techqa)
        for split in ("train", "dev")
    }
    techqa_reconciled = (
        techqa_stats.get("policy") == "historical_benchmark_and_rag_knowledge"
        and techqa_stats.get("original_labeled_examples")
        == len(techqa) + techqa_stats.get("removed_malformed", -1) + techqa_stats.get("removed_duplicates", -1)
        and techqa_stats.get("source_counts") == {"train": 600, "dev": 310}
        and techqa_stats.get("final_by_original_split") == actual_split_counts
    )
    if not techqa_reconciled:
        errors.append("TechQA source splits do not reconcile to the full benchmark")
    legacy_outputs = [
        processed / "evaluation" / name
        for name in ("techqa_test.jsonl", "techqa_train.jsonl", "techqa_dev.jsonl", "techqa_validation.jsonl")
    ]
    single_benchmark_file = not any(path.exists() for path in legacy_outputs)
    if not single_benchmark_file:
        errors.append("legacy split-specific TechQA output still exists")
    report = {
        "title": "TechQA Preparation Validation Report",
        "status": "PASS" if not errors else "FAIL",
        "passed": not errors,
        "sizes": sizes,
        "preprocessing": preprocessing,
        "sources": load_json(processed / "download_manifest.json").get("datasets", {}),
        "outputs": {
            name: output_details(path, output_counts[name], project_root)
            for name, path in output_paths.items()
        },
        "checks": {
            "required_schemas": {"passed": schema_passed, "details": "All required fields and structures are valid." if schema_passed else "Malformed records were found."},
            "duplicates_removed": {"passed": duplicates_passed, "details": f"Remaining duplicates: {sum(duplicates.values())} across all outputs."},
            "reasonable_dataset_sizes": {"passed": sizes_passed, "details": f"{len(techqa):,} TechQA records."},
            "all_techqa_source_splits_combined": {"passed": techqa_reconciled, "details": f"Original splits retained only as metadata: {actual_split_counts}."},
            "single_techqa_benchmark_file": {"passed": single_benchmark_file, "details": "No split-specific TechQA processed outputs exist." if single_benchmark_file else "Legacy split-specific output exists."},
        },
        "duplicates": duplicates,
        "notes": [
            "Raw source files are preserved under data/raw and excluded from version control.",
            "All usable labeled TechQA train and dev source records are combined into one historical benchmark file; original split is traceability metadata only.",
            "TechQA is excluded from fine-tuning and tuning; it is now also a RAG knowledge source. Historical TechQA benchmark results predate this role change.",
            "DEX split and leakage checks are recorded in the separate finetune_v2 manifest.",
            "The current RAG corpus and index are validated separately by rag.build_corpus and rag.index.",
        ],
        "errors": errors,
    }
    output = processed / "validation_report.json"
    write_json(output, report)
    if errors:
        raise ValueError("; ".join(errors))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the processed TechQA data.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(validate_datasets(args.project_root.resolve()), indent=2))

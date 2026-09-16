from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.utils import read_jsonl, text_key, write_json


def duplicate_count(values: list[str]) -> int:
    keys = [text_key(value) for value in values]
    return len(keys) - len(set(keys))


def exact_leakage(techqa: list[dict], finetune: list[dict], rag: list[dict]) -> dict:
    train_prompts = {text_key(row["prompt"][0]["content"]) for row in finetune}
    train_answers = {text_key(row["completion"][0]["content"]) for row in finetune}
    rag_text = {text_key(row["text"]) for row in rag}
    questions = {text_key(row["question"]) for row in techqa}
    answers = {text_key(row["answer"]) for row in techqa if row["answer"]}
    return {
        "techqa_questions_in_finetune": len(questions & train_prompts),
        "techqa_answers_in_finetune": len(answers & train_answers),
        "techqa_questions_in_rag": len(questions & rag_text),
        "techqa_answers_in_rag": len(answers & rag_text),
    }


def valid_finetune_record(row: dict) -> bool:
    try:
        prompt, completion = row["prompt"], row["completion"]
        return (
            set(row) == {"prompt", "completion"}
            and len(prompt) == len(completion) == 1
            and prompt[0].get("role") == "user"
            and completion[0].get("role") == "assistant"
            and bool(text_key(prompt[0].get("content")))
            and bool(text_key(completion[0].get("content")))
        )
    except (AttributeError, KeyError, TypeError):
        return False


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
    preprocessing = {
        "finetune": load_json(processed / "finetune" / "preprocessing_report.json"),
        "evaluation": load_json(processed / "evaluation" / "preprocessing_report.json"),
        "rag": load_json(processed / "rag" / "preprocessing_report.json"),
    }
    train = read_jsonl(processed / "finetune" / "train.jsonl")
    validation = read_jsonl(processed / "finetune" / "validation.jsonl")
    techqa = read_jsonl(processed / "evaluation" / "techqa_benchmark.jsonl")
    rag = read_jsonl(processed / "rag" / "documents.jsonl")
    errors = []

    valid_finetune = {}
    for name, rows in (("train", train), ("validation", validation)):
        valid_finetune[name] = [row for row in rows if valid_finetune_record(row)]
        if len(valid_finetune[name]) != len(rows):
            errors.append(f"{name}: malformed prompt/completion record")

    train_keys = {text_key(row["prompt"][0]["content"]) for row in valid_finetune["train"]}
    validation_keys = {text_key(row["prompt"][0]["content"]) for row in valid_finetune["validation"]}
    split_overlap = len(train_keys & validation_keys)
    if split_overlap:
        errors.append("fine-tuning train and validation prompts overlap")

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
    valid_rag = [
        row for row in rag
        if isinstance(row, dict)
        and set(row) == {"id", "text", "metadata"}
        and row["id"]
        and text_key(row["text"])
        and isinstance(row["metadata"], dict)
    ]
    if len(valid_rag) != len(rag):
        errors.append("RAG: malformed record")

    duplicates = {
        "train_prompts": duplicate_count([row["prompt"][0]["content"] for row in valid_finetune["train"]]),
        "validation_prompts": duplicate_count([row["prompt"][0]["content"] for row in valid_finetune["validation"]]),
        "techqa_ids": len(valid_techqa) - len({row["id"] for row in valid_techqa}),
        "rag_ids": len(valid_rag) - len({row["id"] for row in valid_rag}),
        "rag_texts": duplicate_count([row["text"] for row in valid_rag]),
    }
    duplicates_passed = not any(duplicates.values())
    if not duplicates_passed:
        errors.append("unexpected duplicates remain")

    leakage = exact_leakage(valid_techqa, valid_finetune["train"] + valid_finetune["validation"], valid_rag)
    leakage_passed = not any(leakage.values())
    if not leakage_passed:
        errors.append("exact TechQA content leakage detected")

    techqa_stats = preprocessing.get("evaluation", {})
    sizes = {"train": len(train), "validation": len(validation), "techqa_benchmark": len(techqa), "rag": len(rag)}
    expected_techqa = techqa_stats.get("final_benchmark_examples")
    sizes_passed = sizes["train"] >= 100 and sizes["validation"] >= 10 and sizes["techqa_benchmark"] == expected_techqa and sizes["rag"] >= 100
    if not sizes_passed:
        errors.append("one or more final dataset sizes are unreasonable")

    schema_passed = len(valid_finetune["train"]) == len(train) and len(valid_finetune["validation"]) == len(validation) and len(valid_techqa) == len(techqa) and len(valid_rag) == len(rag)
    output_paths = {
        "fine-tuning train": processed / "finetune" / "train.jsonl",
        "fine-tuning validation": processed / "finetune" / "validation.jsonl",
        "TechQA benchmark": processed / "evaluation" / "techqa_benchmark.jsonl",
        "initial RAG documents": processed / "rag" / "documents.jsonl",
    }
    output_counts = dict(zip(output_paths, (len(train), len(validation), len(techqa), len(rag))))
    deterministic_passed = preprocessing.get("finetune", {}).get("seed") == 42
    if not deterministic_passed:
        errors.append("fine-tuning split does not record the required seed 42")
    actual_split_counts = {
        split: sum(row["metadata"]["original_split"] == split for row in valid_techqa)
        for split in ("train", "dev")
    }
    techqa_reconciled = (
        techqa_stats.get("policy") == "benchmark_only"
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
        "title": "Dataset Preparation Validation Report",
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
            "train_validation_disjoint": {"passed": split_overlap == 0, "details": f"Overlapping normalized prompts: {split_overlap}"},
            "no_exact_techqa_leakage": {"passed": leakage_passed, "details": f"Exact normalized TechQA matches: {sum(leakage.values())}."},
            "reasonable_dataset_sizes": {"passed": sizes_passed, "details": f"{len(train):,} train, {len(validation):,} validation, {len(techqa):,} TechQA, and {len(rag):,} RAG records."},
            "deterministic_split_configuration": {"passed": deterministic_passed, "details": f"Fixed random seed: {preprocessing.get('finetune', {}).get('seed', 'missing')}"},
            "all_techqa_source_splits_combined": {"passed": techqa_reconciled, "details": f"Original splits retained only as metadata: {actual_split_counts}."},
            "single_techqa_benchmark_file": {"passed": single_benchmark_file, "details": "No split-specific TechQA processed outputs exist." if single_benchmark_file else "Legacy split-specific output exists."},
        },
        "duplicates": duplicates,
        "train_validation_prompt_overlap": split_overlap,
        "exact_leakage": leakage,
        "notes": [
            "Raw source files are preserved under data/raw and excluded from version control.",
            "All usable labeled TechQA train and dev source records are combined into one benchmark-only dataset; original split is traceability metadata only.",
            "The TechQA benchmark is excluded from fine-tuning, validation, prompt or hyperparameter tuning, RAG development, indexing, and weight updates.",
            "The same unchanged TechQA benchmark must be used for all four later experiments.",
            "Leakage detection checks exact normalized questions and answers; it is not a semantic-similarity audit.",
            "RAG preprocessing creates whole documents only; chunking, embeddings, retrieval, and vector storage are intentionally out of scope.",
        ],
        "errors": errors,
    }
    output = processed / "validation_report.json"
    write_json(output, report)
    if errors:
        raise ValueError("; ".join(errors))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate all processed datasets.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(validate_datasets(args.project_root.resolve()), indent=2))

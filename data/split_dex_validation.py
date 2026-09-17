"""Repartition the original DEX validation set without touching its 27K train set."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from benchmark.common.dataset import file_sha256
from data.preprocess_finetune_v2 import normalized, prompt_key, topic_for
from data.utils import read_jsonl, write_json, write_jsonl

TRAIN_SHA256 = "f9b8d4e4af26263ad879b0ac90f0ce4fb4b2fcad7c51dfea88fccf7caf144c2a"
ORIGINAL_VALIDATION_SHA256 = "f046451381792a0be7715830ba233a073ac8db90667005c8f042b92efec4fa3d"
SEED = 42
BENCHMARK_SIZE = 1000
SPLIT_METHOD = "Proportional keyword-topic stratification; seeded shuffle within topic, largest remainder for quotas"


def _keys(rows: list[dict]) -> tuple[set[str], set[tuple[str, str]]]:
    prompts = [prompt_key(row["prompt"]) for row in rows]
    pairs = [(prompt, normalized(row["completion"][0]["content"])) for prompt, row in zip(prompts, rows, strict=True)]
    if len(set(prompts)) != len(rows) or len(set(pairs)) != len(rows):
        raise ValueError("A DEX split contains duplicate prompts or conversations")
    return set(prompts), set(pairs)


def split_records(original: list[dict], *, seed: int = SEED) -> tuple[list[dict], list[dict], dict]:
    if len(original) != 3000:
        raise ValueError(f"Expected 3,000 original validation examples, found {len(original)}")
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(original):
        if (len(row["prompt"]) != 1 or row["prompt"][0]["role"] != "user"
                or len(row["completion"]) != 1 or row["completion"][0]["role"] != "assistant"):
            raise ValueError(f"Original validation example {index} is not a single-turn SFT record")
        question, answer = row["prompt"][0]["content"], row["completion"][0]["content"]
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Original validation example {index} has an empty question or answer")
        buckets[topic_for(question)].append(index)

    quotas = {topic: len(indices) * BENCHMARK_SIZE // len(original) for topic, indices in buckets.items()}
    remainder = BENCHMARK_SIZE - sum(quotas.values())
    ranked = sorted(buckets, key=lambda topic: (-(len(buckets[topic]) * BENCHMARK_SIZE % len(original)), topic))
    for topic in ranked[:remainder]:
        quotas[topic] += 1
    rng = random.Random(seed)
    selected = set()
    for topic in sorted(buckets):
        indices = buckets[topic][:]
        rng.shuffle(indices)
        selected.update(indices[:quotas[topic]])
    if len(selected) != BENCHMARK_SIZE:
        raise ValueError("Benchmark stratification did not select exactly 1,000 records")

    validation = [row for index, row in enumerate(original) if index not in selected]
    benchmark = [
        {"id": f"DEX_GENERAL_IT_{index + 1:06d}",
         "question": original[index]["prompt"][0]["content"],
         "reference_answer": original[index]["completion"][0]["content"],
         "metadata": {"topic": topic_for(original[index]["prompt"][0]["content"]),
                      "original_validation_index": index}}
        for index in sorted(selected)
    ]
    return validation, benchmark, {
        "validation": dict(sorted(Counter(topic_for(row["prompt"][0]["content"]) for row in validation).items())),
        "benchmark": dict(sorted(Counter(row["metadata"]["topic"] for row in benchmark).items())),
    }


def run(root: Path) -> dict:
    processed = root / "data/processed/finetune_v2"
    benchmark_dir = root / "data/processed/benchmark"
    train_path = processed / "train.jsonl"
    validation_path = processed / "validation.jsonl"
    original_path = processed / "validation_original_3000.jsonl"
    benchmark_path = benchmark_dir / "dex_general_it_benchmark.jsonl"
    benchmark_manifest_path = benchmark_dir / "dex_general_it_manifest.json"

    if file_sha256(train_path) != TRAIN_SHA256:
        raise ValueError("DEX training data changed; refusing to create a benchmark")
    manifest_path = processed / "manifest.json"
    stats_path = processed / "filter_stats.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if manifest["validation_count"] == 3000 and file_sha256(validation_path) != ORIGINAL_VALIDATION_SHA256:
        raise ValueError("Regenerated original validation data differs from the archived 3K split")
    if not original_path.exists():
        if file_sha256(validation_path) != ORIGINAL_VALIDATION_SHA256:
            raise ValueError("Original 3K validation file has an unexpected hash; refusing to overwrite it")
        shutil.copyfile(validation_path, original_path)
    if file_sha256(original_path) != ORIGINAL_VALIDATION_SHA256:
        raise ValueError("Archived original 3K validation file has an unexpected hash")
    original = read_jsonl(original_path)
    validation, benchmark, topics = split_records(original)
    train = read_jsonl(train_path)
    benchmark_sft = [{"prompt": [{"role": "user", "content": row["question"]}],
                      "completion": [{"role": "assistant", "content": row["reference_answer"]}]}
                     for row in benchmark]
    splits = {"train": train, "validation": validation, "benchmark": benchmark_sft}
    checked = {name: _keys(rows) for name, rows in splits.items()}
    overlaps = {}
    for left, right in (("train", "validation"), ("train", "benchmark"), ("validation", "benchmark")):
        prompt_overlap = len(checked[left][0] & checked[right][0])
        conversation_overlap = len(checked[left][1] & checked[right][1])
        overlaps[f"{left}_{right}"] = {"prompt": prompt_overlap, "conversation": conversation_overlap}
    if any(count for pair in overlaps.values() for count in pair.values()):
        raise ValueError(f"DEX split overlap detected: {overlaps}")

    write_jsonl(validation_path, validation)
    write_jsonl(benchmark_path, benchmark)
    validation_sha = file_sha256(validation_path)
    benchmark_sha = file_sha256(benchmark_path)
    layout = {"train": 27000, "validation": len(validation), "benchmark": len(benchmark)}
    if layout != {"train": 27000, "validation": 2000, "benchmark": 1000}:
        raise ValueError(f"Unexpected revised DEX layout: {layout}")

    revision = {
        "original_validation_count": 3000,
        "original_validation_sha256": ORIGINAL_VALIDATION_SHA256,
        "validation_count": 2000,
        "validation_sha256": validation_sha,
        "benchmark_count": 1000,
        "benchmark_path": str(benchmark_path.relative_to(root)),
        "benchmark_sha256": benchmark_sha,
        "split_seed": SEED,
        "split_method": SPLIT_METHOD,
        "split_topic_distribution": topics,
        "split_overlap_checks": overlaps,
        "original_processed_layout": {"train": 27000, "validation": 3000},
        "revised_evaluation_layout": layout,
    }
    manifest.update(revision)
    stats.update(revision)
    manifest["output_files"] = sorted(set(manifest["output_files"]) | {
        str(original_path.relative_to(root)), str(benchmark_path.relative_to(root)),
        str(benchmark_manifest_path.relative_to(root)),
    })
    write_json(manifest_path, manifest)
    write_json(stats_path, stats)
    write_json(benchmark_manifest_path, {
        "benchmark_name": "DEX general IT support",
        "source_dataset": manifest["source_dataset"],
        "source_revision": manifest["source_revision"],
        "source_relationship": "Subset of the original 3K DEX validation set; previously used for the first DEX adapter's checkpoint selection. Held out only for models retrained after this split.",
        "benchmark_path": str(benchmark_path.relative_to(root)),
        "benchmark_count": 1000,
        "benchmark_sha256": benchmark_sha,
        "train_sha256": TRAIN_SHA256,
        "original_validation_sha256": ORIGINAL_VALIDATION_SHA256,
        "validation_sha256": validation_sha,
        "split_seed": SEED,
        "split_method": SPLIT_METHOD,
        "topic_distribution": topics,
        "split_overlap_checks": overlaps,
    })
    return revision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), indent=2))

"""Small RTX 5090-only batching check; does not write benchmark results."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmark.baseline.config import (
    ANSWER_SYSTEM_PROMPT,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    DATASET_PATH,
    MAX_NEW_TOKENS,
    RANDOM_SEED,
)
from benchmark.baseline.model import QwenGenerator
from benchmark.baseline.run_benchmark import _set_seed
from benchmark.common.config import JUDGE_MAX_NEW_TOKENS
from benchmark.common.dataset import load_techqa
from benchmark.common.generation import batches
from benchmark.common.judge import build_judge_messages


def _measure(generator: QwenGenerator, prompts: list[list[dict[str, str]]], size: int, token_limit: int) -> tuple[list[str], dict]:
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    outputs = [output for batch in batches(prompts, size) for output in generator.generate_batch(batch, token_limit)]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return outputs, {
        "elapsed_seconds": round(elapsed, 2),
        "examples_per_second": round(len(prompts) / elapsed, 3),
        "peak_allocated_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }


def run(project_root: Path, count: int = 8) -> dict:
    import torch

    if not 8 <= count <= 16:
        raise ValueError("Probe count must be between 8 and 16")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; no model was loaded")
    name = torch.cuda.get_device_name(0)
    capacity = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if "5090" not in name or capacity < 24:
        raise RuntimeError(f"This probe requires the RTX 5090; found {name} ({capacity:.1f} GiB). No model was loaded")

    examples = load_techqa(project_root / DATASET_PATH)[:count]
    answer_prompts = [
        [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}, {"role": "user", "content": example.question}]
        for example in examples
    ]
    _set_seed(RANDOM_SEED)
    generator = QwenGenerator.load(BASE_MODEL_ID, BASE_MODEL_REVISION)
    report = {"device": name, "examples": count, "batch_sizes": {}}
    try:
        reference_answers = None
        reference_judgments = None
        for size in (1, 2, 4, 8):
            try:
                answers, answer_stats = _measure(generator, answer_prompts, size, MAX_NEW_TOKENS)
                if reference_answers is None:
                    reference_answers = answers
                # Hold judge inputs fixed at the batch-1 answers so only judge batching changes.
                judge_prompts = [build_judge_messages(example, answer) for example, answer in zip(examples, reference_answers, strict=True)]
                judgments, judge_stats = _measure(generator, judge_prompts, size, JUDGE_MAX_NEW_TOKENS)
                if reference_judgments is None:
                    reference_judgments = judgments
                report["batch_sizes"][str(size)] = {
                    "answers": answer_stats,
                    "judgments": judge_stats,
                    "answer_mismatch_indices": [i for i, (a, b) in enumerate(zip(reference_answers, answers, strict=True)) if a != b],
                    "judge_mismatch_indices": [i for i, (a, b) in enumerate(zip(reference_judgments, judgments, strict=True)) if a != b],
                }
            except RuntimeError as exc:
                if not isinstance(exc.__cause__, torch.cuda.OutOfMemoryError):
                    raise
                report["batch_sizes"][str(size)] = {"error": str(exc)}
                torch.cuda.empty_cache()
                if size == 1:
                    raise RuntimeError("Batch size 1 did not fit; no equivalence baseline is available") from exc
    finally:
        generator.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check small batched Qwen inference on an RTX 5090 without running the full benchmark.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    result = run(args.project_root.resolve(), args.count)
    print(json.dumps(result, indent=2))
    if any(data.get("answer_mismatch_indices") or data.get("judge_mismatch_indices") for data in result["batch_sizes"].values()):
        raise SystemExit("Batch outputs differed from batch size 1; investigate before the full benchmark")

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
    ENABLE_THINKING,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    RANDOM_SEED,
)
from benchmark.baseline.model import QwenGenerator
from benchmark.baseline.run_benchmark import _set_seed
from benchmark.common.config import JUDGE_MAX_NEW_TOKENS
from benchmark.common.dataset import load_techqa
from benchmark.common.generation import batches
from benchmark.common.judge import build_judge_messages, parse_judge_output

PROBE_BATCH_SIZES = (1, 2, 4, 8, 16)


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


def _differences(reference: list[str], batched: list[str]) -> list[dict]:
    differences = []
    for index, (expected, actual) in enumerate(zip(reference, batched, strict=True)):
        if expected != actual:
            first = next((i for i, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]), min(len(expected), len(actual)))
            differences.append({
                "index": index,
                "first_different_character": first,
                "batch_1_excerpt": expected[max(0, first - 40):first + 80],
                "batched_excerpt": actual[max(0, first - 40):first + 80],
            })
    return differences


def _judge_score_changes(reference: list[str], batched: list[str]) -> list[dict]:
    changes = []
    for index, (expected, actual) in enumerate(zip(reference, batched, strict=True)):
        try:
            before = parse_judge_output(expected).score
            after = parse_judge_output(actual).score
        except ValueError as exc:
            changes.append({"index": index, "error": str(exc)})
            continue
        if before != after:
            changes.append({"index": index, "batch_1_score": before, "batched_score": after})
    return changes


def _first_token_diagnostic(generator: QwenGenerator, prompts: list[list[dict[str, str]]]) -> dict:
    import torch

    results = {}
    baseline_logits = []
    baseline_ids = []
    for size in (1, 2):
        rows = []
        for batch in batches(prompts[:2], size):
            inputs = generator.tokenizer.apply_chat_template(
                batch, add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt", padding=True, truncation=True,
                max_length=MAX_INPUT_TOKENS, enable_thinking=ENABLE_THINKING,
            ).to(generator.model.device)
            with torch.inference_mode():
                output = generator.model.generate(
                    **inputs, max_new_tokens=1, do_sample=False,
                    pad_token_id=generator.tokenizer.pad_token_id,
                    return_dict_in_generate=True, output_logits=True,
                )
            for index, logits in enumerate(output.logits[0]):
                top = torch.topk(logits.float(), 2)
                prompt_ids = inputs["input_ids"][index][inputs["attention_mask"][index].bool()].tolist()
                row = {
                    "prompt_tokens": int(inputs["attention_mask"][index].sum()),
                    "left_padding_tokens": int((inputs["attention_mask"][index] == 0).sum()),
                    "first_generated_token_id": int(output.sequences[index, inputs["input_ids"].shape[1]]),
                    "top_logit_margin": round(float(top.values[0] - top.values[1]), 4),
                }
                if size == 1:
                    baseline_ids.append(prompt_ids)
                    baseline_logits.append(logits.float().cpu())
                else:
                    row["same_prompt_tokens_as_batch_1"] = prompt_ids == baseline_ids[len(rows)]
                    row["max_abs_first_logit_difference"] = round(
                        float((logits.float().cpu() - baseline_logits[len(rows)]).abs().max()), 4
                    )
                rows.append(row)
        results[str(size)] = rows
    return results


def run(project_root: Path, count: int = 8, answer_token_limit: int = MAX_NEW_TOKENS, diagnose: bool = False) -> dict:
    import torch

    if not 2 <= count <= 16:
        raise ValueError("Probe count must be between 2 and 16")
    if not 1 <= answer_token_limit <= MAX_NEW_TOKENS:
        raise ValueError(f"Answer token limit must be between 1 and {MAX_NEW_TOKENS}")
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
    report = {"device": name, "examples": count, "answer_token_limit": answer_token_limit, "batch_sizes": {}}
    try:
        if diagnose:
            report["first_token_diagnostic"] = _first_token_diagnostic(generator, answer_prompts)
        reference_answers = None
        reference_judgments = None
        for size in (size for size in PROBE_BATCH_SIZES if size <= count):
            try:
                answers, answer_stats = _measure(generator, answer_prompts, size, answer_token_limit)
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
                    "answer_differences": _differences(reference_answers, answers),
                    "judge_differences": _differences(reference_judgments, judgments),
                    "judge_score_changes": _judge_score_changes(reference_judgments, judgments),
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
    parser.add_argument("--answer-token-limit", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--diagnose", action="store_true", help="Compare first generated tokens and logit margins for the first two prompts")
    args = parser.parse_args()
    result = run(args.project_root.resolve(), args.count, args.answer_token_limit, args.diagnose)
    print(json.dumps(result, indent=2))
    if any("error" in data for data in result["batch_sizes"].values()):
        raise SystemExit("A batch ran out of memory; lower the affected batch size before the full benchmark")
    if any(data.get("judge_score_changes") for data in result["batch_sizes"].values()):
        raise SystemExit("Judge scores changed or a response was invalid; investigate before the full benchmark")

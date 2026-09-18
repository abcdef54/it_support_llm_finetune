from __future__ import annotations

from types import SimpleNamespace

BASE_MODEL_ID = "Qwen/Qwen3.5-9B"
BASE_MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL_DTYPE = "bfloat16"
GENERATION_MODE = "deterministic_greedy"
GENERATION_TEMPERATURE = 0.0
MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKENS = 4096
ANSWER_BATCH_SIZE = 12
JUDGE_BATCH_SIZE = 1
RANDOM_SEED = 42
ENABLE_THINKING = False

DATASET_PATH = "data/processed/evaluation/techqa_benchmark.jsonl"
PREDICTIONS_PATH = "results/base/predictions.jsonl"
METRICS_PATH = "results/base/metrics.json"

ANSWER_SYSTEM_PROMPT = """You are an IT support assistant. Give a concise, technically useful answer.
If the question cannot be answered reliably from the information given, say that you do not know."""


def for_benchmark(benchmark: str = "general_it", *, answer_batch_size: int | None = None) -> SimpleNamespace:
    if benchmark not in {"techqa", "general_it"}:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    values = {name: value for name, value in globals().items() if name.isupper()}
    if benchmark == "general_it":
        values.update(
            DATASET_KIND="general_it",
            DATASET_PATH="data/processed/benchmark/dex_general_it_benchmark.jsonl",
            DATASET_MANIFEST_PATH="data/processed/benchmark/dex_general_it_manifest.json",
            PREDICTIONS_PATH="results/base_general_it/predictions.jsonl",
            METRICS_PATH="results/base_general_it/metrics.json",
        )
    if answer_batch_size is not None:
        if answer_batch_size < 1:
            raise ValueError("Answer batch size must be positive")
        values["ANSWER_BATCH_SIZE"] = answer_batch_size
    return SimpleNamespace(**values)

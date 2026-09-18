from __future__ import annotations

from types import SimpleNamespace
from benchmark.baseline import config as baseline

# Source the controlled experiment settings from the baseline, not a copy.
BASE_MODEL_ID = baseline.BASE_MODEL_ID
BASE_MODEL_REVISION = baseline.BASE_MODEL_REVISION
MODEL_DTYPE = baseline.MODEL_DTYPE
GENERATION_MODE = baseline.GENERATION_MODE
GENERATION_TEMPERATURE = baseline.GENERATION_TEMPERATURE
MAX_NEW_TOKENS = baseline.MAX_NEW_TOKENS
MAX_INPUT_TOKENS = baseline.MAX_INPUT_TOKENS
RANDOM_SEED = baseline.RANDOM_SEED
ENABLE_THINKING = baseline.ENABLE_THINKING
DATASET_PATH = baseline.DATASET_PATH
ANSWER_SYSTEM_PROMPT = baseline.ANSWER_SYSTEM_PROMPT

# These may be lowered independently only if the adapter makes a batch too large.
ANSWER_BATCH_SIZE = baseline.ANSWER_BATCH_SIZE
JUDGE_BATCH_SIZE = baseline.JUDGE_BATCH_SIZE

ADAPTER_PATH = "models/qwen3.5-9b-it-support-dex-qlora"
ADAPTER_AUTOCAST_DTYPE = False  # Keep the saved BF16 LoRA weights in BF16 for inference.
PREDICTIONS_PATH = "results/finetuned_dex/predictions.jsonl"
METRICS_PATH = "results/finetuned_dex/metrics.json"


def for_experiment(experiment: str = "dex_v2") -> SimpleNamespace:
    if experiment not in {"dex", "dex_v2"}:
        raise ValueError(f"Unknown fine-tuned benchmark experiment: {experiment}")
    values = {name: value for name, value in globals().items() if name.isupper()}
    if experiment == "dex_v2":
        general = baseline.for_benchmark("general_it")
        values.update(
            DATASET_KIND=general.DATASET_KIND,
            DATASET_PATH=general.DATASET_PATH,
            DATASET_MANIFEST_PATH=general.DATASET_MANIFEST_PATH,
            ADAPTER_PATH="models/qwen3.5-9b-it-support-dex-v2-qlora",
            PREDICTIONS_PATH="results/finetuned_dex_v2/predictions.jsonl",
            METRICS_PATH="results/finetuned_dex_v2/metrics.json",
        )
    return SimpleNamespace(**values)

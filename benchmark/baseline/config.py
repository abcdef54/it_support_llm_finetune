from __future__ import annotations

BASE_MODEL_ID = "Qwen/Qwen3.5-9B"
BASE_MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL_DTYPE = "bfloat16"
GENERATION_MODE = "deterministic_greedy"
GENERATION_TEMPERATURE = 0.0
MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKENS = 4096
ANSWER_BATCH_SIZE = 4
JUDGE_BATCH_SIZE = 4
RANDOM_SEED = 42
ENABLE_THINKING = False

DATASET_PATH = "data/processed/evaluation/techqa_benchmark.jsonl"
PREDICTIONS_PATH = "results/base/predictions.jsonl"
METRICS_PATH = "results/base/metrics.json"

ANSWER_SYSTEM_PROMPT = """You are an IT support assistant. Give a concise, technically useful answer.
If the question cannot be answered reliably from the information given, say that you do not know."""

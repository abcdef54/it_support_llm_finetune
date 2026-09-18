from __future__ import annotations

from benchmark.baseline.config import BASE_MODEL_ID, BASE_MODEL_REVISION

EXPERIMENTS = {
    "dex": {
        "train_dataset": "data/processed/finetune_v2/train.jsonl",
        "validation_dataset": "data/processed/finetune_v2/validation_original_3000.jsonl",
        "output_dir": "models/qwen3.5-9b-it-support-dex-qlora",
    },
    "dex_v2": {
        "train_dataset": "data/processed/finetune_v2/train.jsonl",
        "validation_dataset": "data/processed/finetune_v2/validation.jsonl",
        "output_dir": "models/qwen3.5-9b-it-support-dex-v2-qlora",
    },
}

SEED = 42
MIN_GPU_MEMORY_GIB = 28
MAX_SEQUENCE_LENGTH = 1024
NUM_TRAIN_EPOCHS = 1.0
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
EFFECTIVE_BATCH_SIZE = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE = 1e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_STEPS = 100
OPTIMIZER = "paged_adamw_8bit"
LOGGING_STEPS = 10
EVAL_STEPS = 500
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 2
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "eval_loss"
GREATER_IS_BETTER = False

QUANTIZATION_BITS = 4
QUANTIZATION_TYPE = "nf4"
DOUBLE_QUANTIZATION = True
COMPUTE_DTYPE = "bfloat16"

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def as_dict(experiment: str = "dex_v2") -> dict:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown training experiment: {experiment}")
    return {
        **{
            name.lower(): value
            for name, value in globals().items()
            if name.isupper() and isinstance(value, (str, int, float, bool, tuple))
        },
        **EXPERIMENTS[experiment],
    }

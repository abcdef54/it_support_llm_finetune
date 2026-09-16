from __future__ import annotations

from benchmark.baseline.config import BASE_MODEL_ID, BASE_MODEL_REVISION

TRAIN_DATASET = "data/processed/finetune/train.jsonl"
VALIDATION_DATASET = "data/processed/finetune/validation.jsonl"
OUTPUT_DIR = "models/qwen3.5-9b-it-support-qlora"

SEED = 42
MIN_GPU_MEMORY_GIB = 28
MAX_SEQUENCE_LENGTH = 1024
NUM_TRAIN_EPOCHS = 2.0
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
EFFECTIVE_BATCH_SIZE = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE = 1e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_STEPS = 100
OPTIMIZER = "paged_adamw_8bit"
LOGGING_STEPS = 10
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2

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


def as_dict() -> dict:
    return {
        name.lower(): value
        for name, value in globals().items()
        if name.isupper() and isinstance(value, (str, int, float, bool, tuple))
    }

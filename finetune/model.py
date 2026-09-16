from __future__ import annotations

import os

from finetune.config import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    COMPUTE_DTYPE,
    DOUBLE_QUANTIZATION,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MIN_GPU_MEMORY_GIB,
    QUANTIZATION_TYPE,
)


def require_training_hardware() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU")
    device = int(os.environ.get("LOCAL_RANK", "0"))
    properties = torch.cuda.get_device_properties(device)
    memory_gib = properties.total_memory / 1024**3
    if memory_gib < MIN_GPU_MEMORY_GIB:
        raise RuntimeError(
            f"Refusing to load Qwen3.5-9B on {properties.name} ({memory_gib:.1f} GiB); "
            f"at least {MIN_GPU_MEMORY_GIB} GiB is required"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The training GPU must support bfloat16")
    return {"device": device, "name": properties.name, "memory_gib": round(memory_gib, 2)}


def build_quantization_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=QUANTIZATION_TYPE,
        bnb_4bit_use_double_quant=DOUBLE_QUANTIZATION,
        bnb_4bit_compute_dtype=getattr(torch, COMPUTE_DTYPE),
    )


def build_lora_config():
    from peft import LoraConfig

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )


def validate_target_modules(model) -> dict[str, int]:
    matches = {target: [] for target in LORA_TARGET_MODULES}
    for name, module in model.named_modules():
        target = name.rsplit(".", 1)[-1]
        if target in matches:
            matches[target].append(name)
    missing = [target for target, names in matches.items() if not names]
    if missing:
        raise ValueError(f"LoRA target modules missing from Qwen3.5: {missing}")
    vision_matches = [name for names in matches.values() for name in names if ".visual." in name]
    if vision_matches:
        raise ValueError(f"LoRA targets unexpectedly matched the vision tower: {vision_matches[:3]}")
    return {target: len(names) for target, names in matches.items()}


def parameter_counts(model) -> dict[str, float | int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": trainable / total * 100 if total else 0.0,
    }


def load_qlora_model():
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    hardware = require_training_hardware()
    device = hardware["device"]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION)
    tokenizer.padding_side = "right"
    model = AutoModelForMultimodalLM.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        quantization_config=build_quantization_config(),
        dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    targets = validate_target_modules(model)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(model, build_lora_config())
    return model, tokenizer, hardware, targets, parameter_counts(model)

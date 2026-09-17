from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

from packaging.version import Version

from benchmark.baseline.model import QwenGenerator
from benchmark.common.dataset import file_sha256


def inspect_adapter(adapter_path: Path, model_id: str, revision: str, *, expected_experiment: str | None = None) -> dict:
    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.safetensors"
    training_path = adapter_path / "run_metadata.json"
    for path in (config_path, weights_path, training_path):
        if not path.is_file():
            raise FileNotFoundError(f"Fine-tuned adapter artifact not found: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(training, dict):
        raise ValueError("Adapter configuration and training metadata must be JSON objects")
    if config.get("base_model_name_or_path") != model_id or training.get("base_model_id") != model_id:
        raise ValueError("Adapter was not trained from the configured benchmark base model")
    if training.get("base_model_revision") != revision:
        raise ValueError("Adapter training revision differs from the pinned benchmark base revision")
    # if expected_experiment and training.get("experiment") != expected_experiment:
    #     raise ValueError(f"Adapter metadata does not identify the {expected_experiment} training experiment")
    targets = config.get("target_modules")
    if (config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM"
            or not isinstance(config.get("r"), int) or config["r"] < 1
            or not isinstance(targets, list) or not targets
            or not all(isinstance(target, str) and target for target in targets)):
        raise ValueError("Adapter must contain a causal-LM LoRA configuration with target modules")
    saved_peft_version = config.get("peft_version")
    if saved_peft_version and Version(version("peft")) < Version(saved_peft_version):
        raise RuntimeError(f"Adapter requires PEFT >= {saved_peft_version}; installed version is {version('peft')}")
    return {
        "path": str(adapter_path),
        "configuration": config,
        "artifact_sha256": {
            "adapter_config.json": file_sha256(config_path),
            "adapter_model.safetensors": file_sha256(weights_path),
        },
    }


def load_finetuned_generator(
    model_id: str, revision: str, adapter_path: Path, expected_config: dict, *, autocast_adapter_dtype: bool
) -> QwenGenerator:
    from peft import PeftModel

    generator = QwenGenerator.load(model_id, revision)
    try:
        model = PeftModel.from_pretrained(
            generator.model, str(adapter_path), is_trainable=False, autocast_adapter_dtype=autocast_adapter_dtype
        )
        model.eval()  # LoRA dropout must be disabled for inference.
        if not isinstance(model, PeftModel) or "default" not in model.peft_config:
            raise RuntimeError("PEFT did not attach the default LoRA adapter")
        loaded_config = model.peft_config["default"]
        if loaded_config.r != expected_config["r"] or set(loaded_config.target_modules) != set(expected_config["target_modules"]):
            raise RuntimeError("Loaded LoRA configuration does not match the adapter artifacts")
        if "default" not in model.active_adapters or model.training:
            raise RuntimeError("The LoRA adapter is not active in evaluation mode")
        targets = {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if hasattr(module, "lora_A") and hasattr(module, "lora_B")
        }
        missing = set(expected_config["target_modules"]) - targets
        if missing:
            raise RuntimeError(f"LoRA modules are missing from the loaded answer model: {sorted(missing)}")
        generator.model = model
        return generator
    except Exception:
        generator.close()
        raise

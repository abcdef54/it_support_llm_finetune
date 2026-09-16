from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from data.utils import write_json
from finetune import config
from finetune.dataset import inspect_chat_format, load_datasets, load_split


def build_training_arguments(output_dir: Path, *, use_cpu: bool = False):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        warmup_steps=config.WARMUP_STEPS,
        optim=config.OPTIMIZER,
        max_length=config.MAX_SEQUENCE_LENGTH,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        bf16=not use_cpu,
        tf32=False if use_cpu else True,
        use_cpu=use_cpu,
        completion_only_loss=True,
        packing=False,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        logging_steps=config.LOGGING_STEPS,
        report_to="none",
        seed=config.SEED,
        data_seed=config.SEED,
    )


def check(project_root: Path) -> dict:
    train_path = project_root / config.TRAIN_DATASET
    validation_path = project_root / config.VALIDATION_DATASET
    train, validation = load_datasets(train_path, validation_path)
    result = {
        "status": "ready",
        "model_id": config.BASE_MODEL_ID,
        "model_revision": config.BASE_MODEL_REVISION,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "training_configuration": config.as_dict(),
    }
    json.dumps(result)
    return result


def train(project_root: Path, resume_from_checkpoint: str | None = None) -> dict:
    from trl import SFTTrainer

    from finetune.model import load_qlora_model

    train_path = project_root / config.TRAIN_DATASET
    validation_path = project_root / config.VALIDATION_DATASET
    output_dir = project_root / config.OUTPUT_DIR
    train_dataset, validation_dataset = load_datasets(train_path, validation_path)
    model, tokenizer, hardware, targets, parameters = load_qlora_model()
    format_report = inspect_chat_format(tokenizer, load_split(train_path), config.MAX_SEQUENCE_LENGTH)
    print(
        json.dumps(
            {
                "hardware": hardware,
                "configuration": config.as_dict(),
                "lora_target_matches": targets,
                "parameters": parameters,
                "formatting": {key: value for key, value in format_report.items() if key != "previews"},
            },
            indent=2,
        )
    )

    trainer = SFTTrainer(
        model=model,
        args=build_training_arguments(output_dir),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    validation_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    trainer.save_state()
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model_id": config.BASE_MODEL_ID,
        "base_model_revision": config.BASE_MODEL_REVISION,
        "dataset": {
            "train_path": config.TRAIN_DATASET,
            "validation_path": config.VALIDATION_DATASET,
            "train_examples": len(train_dataset),
            "validation_examples": len(validation_dataset),
        },
        "configuration": config.as_dict(),
        "hardware": hardware,
        "lora_target_matches": targets,
        "parameters": parameters,
        "formatting": {key: value for key, value in format_report.items() if key != "previews"},
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_metrics,
        "library_versions": {
            package: version(package)
            for package in ("torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes", "datasets")
        },
    }
    write_json(output_dir / "run_metadata.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Qwen3.5-9B IT-support QLoRA adapter.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true", help="Validate local data/config without loading Qwen")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = check(root) if args.check else train(root, args.resume_from_checkpoint)
    print(json.dumps(result, indent=2, default=str))

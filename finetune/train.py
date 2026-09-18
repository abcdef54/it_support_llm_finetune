from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from math import isfinite
from pathlib import Path

from benchmark.common.dataset import file_sha256
from data.utils import write_json
from finetune import config
from finetune.dataset import inspect_chat_format, load_datasets, load_split


def dataset_provenance(project_root: Path, settings: dict, experiment: str) -> dict:
    hashes = {f"{split}_sha256": file_sha256(project_root / settings[f"{split}_dataset"])
              for split in ("train", "validation")}
    manifest_path = (project_root / settings["train_dataset"]).parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not manifest["audit"]["passed"] or manifest["train_sha256"] != hashes["train_sha256"]
            or manifest["validation_sha256"] != hashes["validation_sha256"]):
        raise ValueError("DEX data differs from the validated preprocessing manifest")
    if experiment == "dex_v2":
        benchmark_path = project_root / manifest["benchmark_path"]
        if benchmark_path.resolve() in {(project_root / settings[f"{split}_dataset"]).resolve()
                                        for split in ("train", "validation")}:
            raise ValueError("The DEX benchmark cannot be a training or validation file")
        if (file_sha256(benchmark_path) != manifest["benchmark_sha256"]
                or any(count for pair in manifest["split_overlap_checks"].values() for count in pair.values())):
            raise ValueError("DEX benchmark integrity or split-overlap audit failed")
        hashes["benchmark_sha256"] = manifest["benchmark_sha256"]
    hashes["manifest_path"] = str(manifest_path.relative_to(project_root))
    hashes["source_revision"] = manifest["source_revision"]
    return hashes


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
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=config.GREATER_IS_BETTER,
        logging_steps=config.LOGGING_STEPS,
        report_to="none",
        seed=config.SEED,
        data_seed=config.SEED,
    )


def check(project_root: Path, experiment: str = "dex_v2") -> dict:
    settings = config.as_dict(experiment)
    train_path = project_root / settings["train_dataset"]
    validation_path = project_root / settings["validation_dataset"]
    train, validation = load_datasets(train_path, validation_path)
    result = {
        "status": "ready",
        "model_id": config.BASE_MODEL_ID,
        "model_revision": config.BASE_MODEL_REVISION,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "experiment": experiment,
        "training_configuration": settings,
        "dataset_provenance": dataset_provenance(project_root, settings, experiment),
    }
    json.dumps(result)
    return result


def train(project_root: Path, resume_from_checkpoint: str | None = None, experiment: str = "dex_v2") -> dict:
    from trl import SFTTrainer

    from finetune.model import load_qlora_model

    settings = config.as_dict(experiment)
    train_path = project_root / settings["train_dataset"]
    validation_path = project_root / settings["validation_dataset"]
    output_dir = project_root / settings["output_dir"]
    if output_dir.exists() and any(output_dir.iterdir()) and not resume_from_checkpoint:
        raise FileExistsError(f"Training output already exists: {output_dir}; use an explicit checkpoint to resume")
    if resume_from_checkpoint and not Path(resume_from_checkpoint).resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("Resume checkpoint must belong to the selected experiment's output directory")
    train_dataset, validation_dataset = load_datasets(train_path, validation_path)
    provenance = dataset_provenance(project_root, settings, experiment)
    model, tokenizer, hardware, targets, parameters = load_qlora_model()
    format_report = inspect_chat_format(tokenizer, load_split(train_path), config.MAX_SEQUENCE_LENGTH)
    print(
        json.dumps(
            {
                "hardware": hardware,
                "configuration": settings,
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
    best_checkpoint = trainer.state.best_model_checkpoint
    if (not best_checkpoint or trainer.state.best_metric is None or not isfinite(trainer.state.best_metric)
            or not any((Path(best_checkpoint) / name).is_file()
                       for name in ("adapter_model.safetensors", "adapter_model.bin"))):
        raise RuntimeError("No valid best adapter checkpoint was selected; refusing to save the last adapter")
    validation_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    trainer.save_state()
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model_id": config.BASE_MODEL_ID,
        "base_model_revision": config.BASE_MODEL_REVISION,
        "dataset": {
            "train_path": settings["train_dataset"],
            "validation_path": settings["validation_dataset"],
            "train_examples": len(train_dataset),
            "validation_examples": len(validation_dataset),
            **provenance,
        },
        "experiment": experiment,
        "configuration": settings,
        "hardware": hardware,
        "lora_target_matches": targets,
        "parameters": parameters,
        "formatting": {key: value for key, value in format_report.items() if key != "previews"},
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_metrics,
        "best_model_checkpoint": best_checkpoint,
        "best_metric": trainer.state.best_metric,
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
    parser.add_argument("--experiment", choices=tuple(config.EXPERIMENTS), default="dex_v2",
                        help="Training dataset/output selection; defaults to the revised held-out DEX experiment")
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = check(root, args.experiment) if args.check else train(root, args.resume_from_checkpoint, args.experiment)
    print(json.dumps(result, indent=2, default=str))

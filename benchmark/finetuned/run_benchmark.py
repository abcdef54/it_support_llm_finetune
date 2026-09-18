from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

from benchmark.baseline.model import QwenGenerator
from benchmark.common.config import JUDGE_MAX_NEW_TOKENS, JUDGE_MODEL_ID, JUDGE_MODEL_REVISION
from benchmark.common.judge import build_judge_messages, parse_or_retry_judge_output
from benchmark.common.runner import run_benchmark
from benchmark.common.schemas import BenchmarkExample
from benchmark.finetuned import config as benchmark_config
from benchmark.finetuned.model import inspect_adapter, load_finetuned_generator


def run(project_root: Path, overwrite: bool = False, resume: bool = False, experiment: str = "dex_v2", rag: bool = False) -> dict:
    config = benchmark_config.for_experiment(experiment)
    rag_records, rag_metadata = None, None
    if rag:
        if experiment != "dex_v2":
            raise ValueError("RAG evaluation requires dex_v2 and the held-out general-IT benchmark")
        from rag.prepare_benchmark import load_artifact
        rag_records, rag_metadata = load_artifact(project_root)
        config.PREDICTIONS_PATH = "results/finetuned_dex_v2_rag/predictions.jsonl"
        config.METRICS_PATH = "results/finetuned_dex_v2_rag/metrics.json"
    adapter_path = project_root / config.ADAPTER_PATH
    adapter = inspect_adapter(adapter_path, config.BASE_MODEL_ID, config.BASE_MODEL_REVISION,
                              expected_experiment=experiment)
    return run_benchmark(
        project_root,
        settings=config,
        load_answer_model=lambda: load_finetuned_generator(
            config.BASE_MODEL_ID, config.BASE_MODEL_REVISION, adapter_path, adapter["configuration"],
            autocast_adapter_dtype=config.ADAPTER_AUTOCAST_DTYPE,
        ),
        load_judge_model=lambda: QwenGenerator.load(JUDGE_MODEL_ID, JUDGE_MODEL_REVISION),
        reuse_answer_as_judge=False,
        overwrite=overwrite,
        resume=resume,
        rag_records=rag_records,
        extra_metadata={
            **({"rag": rag_metadata} if rag else {}),
            "adapter": {**adapter, "path": config.ADAPTER_PATH, "autocast_adapter_dtype": config.ADAPTER_AUTOCAST_DTYPE},
            "library_versions": {
                package: version(package) for package in ("torch", "transformers", "peft", "accelerate", "bert-score")
            },
        },
    )


def smoke(project_root: Path, experiment: str = "dex_v2") -> dict:
    """Check real adapter and base-judge inference without reading benchmark data or writing results."""
    config = benchmark_config.for_experiment(experiment)
    adapter_path = project_root / config.ADAPTER_PATH
    adapter = inspect_adapter(adapter_path, config.BASE_MODEL_ID, config.BASE_MODEL_REVISION,
                              expected_experiment=experiment)
    prompts = [
        [
            {"role": "system", "content": config.ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": "How can I check a Linux service status? " + "Include the command. " * (index % 4)},
        ]
        for index in range(config.ANSWER_BATCH_SIZE)
    ]
    answer_model = load_finetuned_generator(
        config.BASE_MODEL_ID, config.BASE_MODEL_REVISION, adapter_path, adapter["configuration"],
        autocast_adapter_dtype=config.ADAPTER_AUTOCAST_DTYPE,
    )
    try:
        answers = answer_model.generate_batch(prompts, min(64, config.MAX_NEW_TOKENS))
        if len(answers) != len(prompts) or any(not answer for answer in answers):
            raise RuntimeError("Adapter smoke test returned missing or empty answers")
    finally:
        answer_model.close()

    judge = QwenGenerator.load(JUDGE_MODEL_ID, JUDGE_MODEL_REVISION)
    try:
        example = BenchmarkExample("synthetic-smoke", prompts[0][1]["content"],
                                   "Use systemctl status SERVICE.", "synthetic", True)
        judge_messages = build_judge_messages(example, answers[0])
        judgment = judge.generate_batch([judge_messages], JUDGE_MAX_NEW_TOKENS)
        if len(judgment) != 1:
            raise RuntimeError("Base judge smoke test returned an unexpected number of decisions")
        parse_or_retry_judge_output(judge, judge_messages, judgment[0])
    finally:
        judge.close()
    return {"passed": True, "synthetic_answers": len(answers), "fresh_base_judge": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fine-tuned benchmark with optional shared RAG contexts.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue a matching saved benchmark run")
    parser.add_argument("--rag", action="store_true", help="Use the same saved contexts as Base + RAG")
    parser.add_argument("--smoke", action="store_true", help="Run a synthetic GPU smoke test without benchmark data or results")
    parser.add_argument("--experiment", choices=("dex", "dex_v2"), default="dex_v2",
                        help="Select adapter, benchmark, and result directory; dex_v2 uses the new general-IT test")
    args = parser.parse_args()
    if args.smoke and args.rag:
        parser.error("Use python -m rag.smoke for retrieval checks or python -m rag.query for explicit RAG generation")
    print(smoke(args.project_root.resolve(), args.experiment) if args.smoke else run(
        args.project_root.resolve(), overwrite=args.overwrite, resume=args.resume, experiment=args.experiment, rag=args.rag))

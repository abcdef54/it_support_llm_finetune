from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.baseline import config
from benchmark.baseline.model import QwenGenerator
from benchmark.common.config import JUDGE_MODEL_ID, JUDGE_MODEL_REVISION
from benchmark.common.runner import _set_seed, run_benchmark


def run(project_root: Path, overwrite: bool = False, resume: bool = False, benchmark: str = "general_it", rag: bool = False,
        answer_batch_size: int | None = None) -> dict:
    settings = config.for_benchmark(benchmark, answer_batch_size=answer_batch_size)
    rag_records, rag_metadata = None, None
    if rag:
        if benchmark != "general_it":
            raise ValueError("TechQA is indexed knowledge; use the general_it benchmark with RAG")
        from benchmark.common.rag import load_artifact
        rag_records, rag_metadata = load_artifact(project_root)
        settings.PREDICTIONS_PATH = "results/base_rag_general_it/predictions.jsonl"
        settings.METRICS_PATH = "results/base_rag_general_it/metrics.json"
    return run_benchmark(
        project_root,
        settings=settings,
        load_answer_model=lambda: QwenGenerator.load(settings.BASE_MODEL_ID, settings.BASE_MODEL_REVISION),
        load_judge_model=lambda: QwenGenerator.load(JUDGE_MODEL_ID, JUDGE_MODEL_REVISION),
        reuse_answer_as_judge=(settings.BASE_MODEL_ID, settings.BASE_MODEL_REVISION) == (JUDGE_MODEL_ID, JUDGE_MODEL_REVISION),
        overwrite=overwrite,
        resume=resume,
        rag_records=rag_records,
        extra_metadata={"rag": rag_metadata} if rag else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the base-model benchmark.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue a matching saved benchmark run")
    parser.add_argument("--rag", action="store_true", help="Use shared precomputed general-IT retrieval contexts")
    parser.add_argument("--answer-batch-size", type=int, help="Positive answer batch size (default: 12)")
    parser.add_argument("--benchmark", choices=("general_it", "techqa"), default="general_it",
                        help="General IT is primary; TechQA remains available for historical comparison")
    args = parser.parse_args()
    print(run(args.project_root.resolve(), overwrite=args.overwrite, resume=args.resume, benchmark=args.benchmark,
              rag=args.rag, answer_batch_size=args.answer_batch_size))

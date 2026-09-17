from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.baseline import config
from benchmark.baseline.model import QwenGenerator
from benchmark.common.config import JUDGE_MODEL_ID, JUDGE_MODEL_REVISION
from benchmark.common.runner import _set_seed, run_benchmark


def run(project_root: Path, overwrite: bool = False, resume: bool = False, benchmark: str = "general_it") -> dict:
    settings = config.for_benchmark(benchmark)
    return run_benchmark(
        project_root,
        settings=settings,
        load_answer_model=lambda: QwenGenerator.load(settings.BASE_MODEL_ID, settings.BASE_MODEL_REVISION),
        load_judge_model=lambda: QwenGenerator.load(JUDGE_MODEL_ID, JUDGE_MODEL_REVISION),
        reuse_answer_as_judge=(settings.BASE_MODEL_ID, settings.BASE_MODEL_REVISION) == (JUDGE_MODEL_ID, JUDGE_MODEL_REVISION),
        overwrite=overwrite,
        resume=resume,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the base-model benchmark.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue a matching saved benchmark run")
    parser.add_argument("--benchmark", choices=("general_it", "techqa"), default="general_it",
                        help="General IT is primary; TechQA remains available for historical comparison")
    args = parser.parse_args()
    print(run(args.project_root.resolve(), overwrite=args.overwrite, resume=args.resume, benchmark=args.benchmark))

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.baseline import config
from benchmark.baseline.model import QwenGenerator
from benchmark.common.config import JUDGE_MODEL_ID, JUDGE_MODEL_REVISION
from benchmark.common.runner import _set_seed, run_benchmark


def run(project_root: Path, overwrite: bool = False) -> dict:
    return run_benchmark(
        project_root,
        settings=config,
        load_answer_model=lambda: QwenGenerator.load(config.BASE_MODEL_ID, config.BASE_MODEL_REVISION),
        load_judge_model=lambda: QwenGenerator.load(JUDGE_MODEL_ID, JUDGE_MODEL_REVISION),
        reuse_answer_as_judge=(config.BASE_MODEL_ID, config.BASE_MODEL_REVISION) == (JUDGE_MODEL_ID, JUDGE_MODEL_REVISION),
        overwrite=overwrite,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete base-model TechQA benchmark.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(run(args.project_root.resolve(), overwrite=args.overwrite))

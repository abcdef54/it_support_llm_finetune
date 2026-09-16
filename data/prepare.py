from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.download import download_all
from data.inspect_datasets import create_report
from data.preprocess_finetune import run as prepare_finetune
from data.preprocess_rag import run as prepare_rag
from data.preprocess_techqa import run as prepare_techqa
from data.validate import validate_datasets


def run(project_root: Path, download: bool = True) -> dict:
    results = {}
    if download:
        results["download"] = download_all(project_root)
    results["inspection"] = {name: report["rows"] for name, report in create_report(project_root).items()}
    results["finetune"] = prepare_finetune(project_root)
    results["evaluation"] = prepare_techqa(project_root)
    results["rag"] = prepare_rag(project_root)
    results["validation"] = validate_datasets(project_root)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run only the dataset-preparation phase.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-download", action="store_true", help="Reuse raw datasets already on disk.")
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve(), not args.skip_download), indent=2))

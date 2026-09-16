from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download


DATASETS = {
    "finetune": (
        "Tobi-Bueck/customer-support-tickets",
        "ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb",
    ),
    "evaluation": (
        "PrimeQA/TechQA",
        "60437bc79ab217679682217598a3693cab78365b",
    ),
    "rag": (
        "Console-AI/IT-helpdesk-synthetic-tickets",
        "165693849d955c6da4c00bc8be8e541afec0514e",
    ),
}

LICENSES = {
    "finetune": "CC-BY-NC-4.0",
    "evaluation": "CDLA-Permissive-1.0",
    "rag": "MIT",
}

SELECTED_INPUTS = {
    "finetune": "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
    "evaluation": [
        "TechQA/training_and_dev/training_Q_A.json",
        "TechQA/training_and_dev/dev_Q_A.json",
    ],
    "rag": "tickets.csv",
}


def download_all(project_root: Path) -> dict:
    raw_root = project_root / "data" / "raw"
    manifest = {"datasets": {}}
    for purpose, (repo_id, revision) in DATASETS.items():
        destination = raw_root / purpose
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=destination,
        )
        manifest["datasets"][purpose] = {
            "repo_id": repo_id,
            "revision": revision,
            "license": LICENSES[purpose],
            "directory": str(destination.relative_to(project_root)),
            "selected_input": SELECTED_INPUTS[purpose],
        }

    archive = raw_root / "evaluation" / "TechQA.tar.gz"
    extracted = raw_root / "evaluation" / "TechQA" / "README.txt"
    if not extracted.exists():
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(raw_root / "evaluation", filter="data")

    manifest_path = project_root / "data" / "processed" / "download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pinned raw datasets.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(download_all(args.project_root.resolve()), indent=2))

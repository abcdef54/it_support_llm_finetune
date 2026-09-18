from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def inspect_techqa(paths: dict[str, Path], seed: int = 42) -> dict:
    rows = []
    for split, path in paths.items():
        rows.extend({**row, "ORIGINAL_SPLIT": split} for row in json.loads(path.read_text(encoding="utf-8")))
    columns = list(rows[0]) if rows else []
    return {
        "source": {split: str(path) for split, path in paths.items()},
        "rows": len(rows),
        "columns": columns,
        "types": {column: type(rows[0][column]).__name__ for column in columns},
        "missing": {column: sum(row.get(column) in (None, "", []) for row in rows) for column in columns},
        "distributions": {
            "ORIGINAL_SPLIT": dict(Counter(row["ORIGINAL_SPLIT"] for row in rows)),
            "ANSWERABLE": dict(Counter(row["ANSWERABLE"] for row in rows)),
        },
        "samples": random.Random(seed).sample(rows, min(3, len(rows))),
    }


def create_report(project_root: Path) -> dict:
    raw = project_root / "data" / "raw"
    report = {
        "evaluation": inspect_techqa({
            "train": raw / "evaluation" / "TechQA" / "training_and_dev" / "training_Q_A.json",
            "dev": raw / "evaluation" / "TechQA" / "training_and_dev" / "dev_Q_A.json",
        }),
    }
    output = project_root / "data" / "processed" / "schema_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect raw dataset schemas and samples.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(create_report(args.project_root.resolve()), ensure_ascii=False, indent=2))

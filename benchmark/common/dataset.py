from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmark.common.config import TECHQA_RECORDS, TECHQA_SHA256
from benchmark.common.schemas import BenchmarkExample, TechQAExample


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_techqa(
    path: Path,
    *,
    expected_count: int | None = TECHQA_RECORDS,
    expected_sha256: str | None = TECHQA_SHA256,
) -> list[TechQAExample]:
    if not path.is_file():
        raise FileNotFoundError(f"TechQA benchmark not found: {path}")
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ValueError("TechQA benchmark SHA-256 does not match the approved data-preparation output")

    examples: list[TechQAExample] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on TechQA line {line_number}: {exc.msg}") from exc
            examples.append(_validate_row(row, line_number, seen_ids))

    if expected_count is not None and len(examples) != expected_count:
        raise ValueError(f"Expected {expected_count} TechQA records, found {len(examples)}")
    return examples


def _validate_row(row: object, line_number: int, seen_ids: set[str]) -> TechQAExample:
    if not isinstance(row, dict):
        raise ValueError(f"TechQA line {line_number} must be a JSON object")

    record_id, question, answer, metadata = (
        row.get("id"), row.get("question"), row.get("answer"), row.get("metadata")
    )
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError(f"TechQA line {line_number} has an invalid id")
    if record_id in seen_ids:
        raise ValueError(f"Duplicate TechQA id: {record_id}")
    seen_ids.add(record_id)
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"TechQA record {record_id} has an empty question")
    if not isinstance(metadata, dict):
        raise ValueError(f"TechQA record {record_id} has invalid metadata")

    answerable = metadata.get("answerable")
    original_split = metadata.get("original_split")
    if not isinstance(answerable, bool):
        raise ValueError(f"TechQA record {record_id} has invalid answerable metadata")
    if original_split not in {"train", "dev"}:
        raise ValueError(f"TechQA record {record_id} has invalid original_split metadata")
    if answerable and (not isinstance(answer, str) or not answer.strip()):
        raise ValueError(f"Answerable TechQA record {record_id} has no reference answer")
    if not answerable and answer is not None:
        raise ValueError(f"Unanswerable TechQA record {record_id} must have a null answer")

    return TechQAExample(record_id, question.strip(), answer.strip() if answerable else None, original_split, answerable)


def load_general_it(
    path: Path, *, expected_count: int = 1000, expected_sha256: str | None = None,
) -> list[BenchmarkExample]:
    if not path.is_file():
        raise FileNotFoundError(f"General-IT benchmark not found: {path}")
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ValueError("General-IT benchmark SHA-256 differs from its manifest")
    examples: list[BenchmarkExample] = []
    seen_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on general-IT line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"General-IT line {line_number} must be a JSON object")
            record_id, question, answer = row.get("id"), row.get("question"), row.get("reference_answer")
            if (not isinstance(record_id, str) or not record_id.strip() or record_id in seen_ids
                    or not isinstance(question, str) or not question.strip()
                    or not isinstance(answer, str) or not answer.strip()):
                raise ValueError(f"General-IT line {line_number} has a duplicate ID or missing question/reference")
            if not isinstance(row.get("metadata"), dict):
                raise ValueError(f"General-IT line {line_number} has invalid metadata")
            seen_ids.add(record_id)
            examples.append(BenchmarkExample(record_id, question, answer, "heldout", True))
    if len(examples) != expected_count:
        raise ValueError(f"Expected {expected_count} general-IT records, found {len(examples)}")
    return examples

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\\n", "\n").replace("\r\n", "\n").strip()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def text_key(value: object) -> str:
    """Return a normalized text key for duplicate detection (case-folded, whitespace-normalized)."""
    return " ".join(clean_text(value).casefold().split())


def join_question(title: object, body: object) -> str:
    """Join title and body into a single text, preserving meaningful structure and detecting duplication."""
    title_text, body_text = clean_text(title), clean_text(body)
    if title_text and body_text and text_key(title_text) != text_key(body_text):
        return f"{title_text}\n\n{body_text}"
    return body_text or title_text


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    """Write records to a JSONL file, creating the parent directory if needed and handling encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, value: object) -> None:
    """Write a Python object to a JSON file with pretty-printing and proper encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    """Read records from a JSONL file, handling empty lines and ensuring proper encoding."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

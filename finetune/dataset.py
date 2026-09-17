from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from data.utils import text_key


def load_split(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Fine-tuning dataset not found: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc.msg}") from exc
            _validate_record(row, path, line_number)
            records.append({**row, "chat_template_kwargs": {"enable_thinking": False}})
    if not records:
        raise ValueError(f"Fine-tuning dataset is empty: {path}")
    return records


def load_datasets(train_path: Path, validation_path: Path):
    from datasets import Dataset

    train, validation = load_split(train_path), load_split(validation_path)
    key = lambda row: tuple((message["role"], text_key(message["content"])) for message in row["prompt"])
    train_prompts = {key(row) for row in train}
    validation_prompts = {key(row) for row in validation}
    if len(train_prompts) != len(train) or len(validation_prompts) != len(validation):
        raise ValueError("Fine-tuning split contains duplicate prompts")
    overlap = train_prompts & validation_prompts
    if overlap:
        raise ValueError(f"Train and validation contain {len(overlap)} overlapping prompts")
    return Dataset.from_list(train), Dataset.from_list(validation)


def inspect_chat_format(processor, records: list[dict], max_length: int) -> dict:
    longest = 0
    over_limit = 0
    previews = []
    for index, record in enumerate(records):
        messages = record["prompt"] + record["completion"]
        kwargs = record["chat_template_kwargs"]
        rendered = processor.apply_chat_template(messages, tokenize=False, **kwargs)
        token_ids = processor.apply_chat_template(messages, tokenize=True, **kwargs)
        if isinstance(token_ids, Mapping):
            token_ids = token_ids["input_ids"]
        length = len(token_ids)
        longest = max(longest, length)
        over_limit += length > max_length
        if index < 3:
            if any(message["content"] not in rendered for message in messages):
                raise ValueError("Qwen chat template dropped prompt or completion content")
            if "[INST]" in rendered:
                raise ValueError("Incompatible Llama-style chat token found")
            previews.append(rendered)
    if over_limit:
        raise ValueError(f"{over_limit} examples exceed max sequence length {max_length}")
    return {"max_formatted_tokens": longest, "over_length_examples": over_limit, "previews": previews}


def _validate_record(row: object, path: Path, line_number: int) -> None:
    if not isinstance(row, dict) or set(row) != {"prompt", "completion"}:
        raise ValueError(f"Invalid schema in {path} line {line_number}")
    prompt, completion = row["prompt"], row["completion"]
    if not isinstance(prompt, list) or not prompt or not isinstance(completion, list) or len(completion) != 1:
        raise ValueError(f"Expected a nonempty prompt and one assistant completion in {path} line {line_number}")
    previous = None
    for index, message in enumerate(prompt + completion):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(f"Invalid message in {path} line {line_number}")
        expected = "assistant" if previous == "user" else "user"
        role = message["role"]
        if (role != expected and not (index == 0 and role == "system")
                or not isinstance(message["content"], str) or not message["content"].strip()):
            raise ValueError(f"Invalid conversation role or content in {path} line {line_number}")
        previous = role
    if previous != "assistant":
        raise ValueError(f"Final completion must be assistant in {path} line {line_number}")

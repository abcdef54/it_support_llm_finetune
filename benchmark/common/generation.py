from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def batches(items: Sequence[T], size: int):
    if size < 1:
        raise ValueError("Batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def generate_batch(
    tokenizer,
    model,
    batch_messages: list[list[dict[str, str]]],
    *,
    max_input_tokens: int,
    max_new_tokens: int,
    enable_thinking: bool,
) -> list[str]:
    import torch

    if not batch_messages or any(
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(message, dict)
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
            for message in messages
        )
        for messages in batch_messages
    ):
        raise ValueError("Each batch item must contain nonempty text chat messages")
    if max_input_tokens < 1 or max_new_tokens < 1:
        raise ValueError("Token limits must be positive")
    if tokenizer.padding_side != "left":
        raise ValueError("Batched causal generation requires left-padded tokenizer inputs")

    inputs = tokenizer.apply_chat_template(
        batch_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        enable_thinking=enable_thinking,
    )
    try:
        inputs = inputs.to(model.device)
        if "attention_mask" not in inputs:
            raise ValueError("Tokenizer did not return an attention mask")
        padded_prompt_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            f"CUDA OOM generating batch of {len(batch_messages)}; reduce the configured inference batch size"
        ) from exc
    if outputs.shape[0] != len(batch_messages):
        raise ValueError("Model returned a different number of sequences than prompts")
    # Transformers returns the entire left-padded input prefix for every output row.
    # Slice at the shared padded width, not each item's unpadded attention-mask length.
    return [
        tokenizer.decode(output_ids[padded_prompt_length:], skip_special_tokens=True).strip()
        for output_ids in outputs
    ]

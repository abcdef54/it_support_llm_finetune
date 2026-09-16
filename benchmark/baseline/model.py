from __future__ import annotations

from dataclasses import dataclass

from benchmark.baseline.config import ENABLE_THINKING, MAX_INPUT_TOKENS, MODEL_DTYPE
from benchmark.common.generation import generate_batch


@dataclass
class QwenGenerator:
    tokenizer: object
    model: object

    @classmethod
    def load(cls, model_id: str, revision: str) -> "QwenGenerator":
        import torch
        from transformers import AutoModelForMultimodalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3.5-9B benchmark generation requires a CUDA GPU")
        dtype = getattr(torch, MODEL_DTYPE)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        tokenizer.padding_side = "left"  # Decoder-only batched generation must continue from real tokens.
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype,
            device_map="auto",
        )
        model.eval()
        return cls(tokenizer, model)

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        return self.generate_batch([messages], max_new_tokens)[0]

    def generate_batch(self, batch_messages: list[list[dict[str, str]]], max_new_tokens: int) -> list[str]:
        return generate_batch(
            self.tokenizer,
            self.model,
            batch_messages,
            max_input_tokens=MAX_INPUT_TOKENS,
            max_new_tokens=max_new_tokens,
            enable_thinking=ENABLE_THINKING,
        )

    def close(self) -> None:
        import gc
        import torch

        self.tokenizer = None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

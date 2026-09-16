from __future__ import annotations

from dataclasses import dataclass

from benchmark.baseline.config import ENABLE_THINKING, MAX_INPUT_TOKENS, MODEL_DTYPE


@dataclass
class QwenGenerator:
    processor: object
    model: object

    @classmethod
    def load(cls, model_id: str, revision: str) -> "QwenGenerator":
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3.5-9B benchmark generation requires a CUDA GPU")
        dtype = getattr(torch, MODEL_DTYPE)
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype,
            device_map="auto",
        )
        model.eval()
        return cls(processor, model)

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        import torch

        multimodal_messages = [
            {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}
            for message in messages
        ]
        inputs = self.processor.apply_chat_template(
            multimodal_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            enable_thinking=ENABLE_THINKING,
        ).to(self.model.device)
        prompt_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return self.processor.decode(output_ids[0][prompt_length:], skip_special_tokens=True).strip()

    def close(self) -> None:
        import gc
        import torch

        self.processor = None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

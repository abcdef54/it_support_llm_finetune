"""Manual base/DEX generation with optional shared RAG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.baseline.config import (
    ANSWER_SYSTEM_PROMPT,
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    MAX_NEW_TOKENS,
    RANDOM_SEED,
)
from benchmark.baseline.model import QwenGenerator
from benchmark.common.runner import _set_seed
from benchmark.finetuned.config import for_experiment
from benchmark.finetuned.model import inspect_adapter, load_finetuned_generator
from rag import config as C
from rag.context import build_context
from rag.retrieve import Retriever


def load_tokenizer(root):
    from transformers import AutoTokenizer
    cache = root / "data/raw/finetune_v2/tokenizer_cache"
    return AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION,
                                         cache_dir=str(cache) if cache.exists() else None)


def generate(root, question, *, model="dex", rag=True, top_k=C.RAG_TOP_K, device=None):
    if model not in {"base", "dex"}:
        raise ValueError("Model must be base or dex")
    settings = for_experiment("dex_v2")
    adapter_path = root / settings.ADAPTER_PATH
    adapter = inspect_adapter(adapter_path, settings.BASE_MODEL_ID, settings.BASE_MODEL_REVISION,
                              expected_experiment="dex_v2") if model == "dex" else None
    retrieved, context = [], None
    if rag:
        retriever = Retriever(root, device=device)
        try:
            retrieved = retriever.retrieve(question, top_k)
            context = build_context(question, retrieved, load_tokenizer(root))
        finally:
            retriever.close()  # Free BGE VRAM before Qwen is loaded.
    _set_seed(RANDOM_SEED)
    generator = (load_finetuned_generator(settings.BASE_MODEL_ID, settings.BASE_MODEL_REVISION, adapter_path,
                                         adapter["configuration"], autocast_adapter_dtype=settings.ADAPTER_AUTOCAST_DTYPE)
                 if adapter else QwenGenerator.load(settings.BASE_MODEL_ID, settings.BASE_MODEL_REVISION))
    try:
        messages = context["messages"] if context else [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT}, {"role": "user", "content": question}]
        answer = generator.generate(messages, MAX_NEW_TOKENS)
        return {"model": model, "rag": rag, "question": question, "retrieved_sources": retrieved,
                "context_used": context, "answer": answer}
    finally:
        generator.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=C.PROJECT_ROOT)
    parser.add_argument("--model", choices=("base", "dex"), default="dex")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=C.RAG_TOP_K)
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--debug-retrieval", action="store_true", help="Print full retrieved Q&A records")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Embedding device only")
    args = parser.parse_args()
    result = generate(args.project_root.resolve(), args.question, model=args.model, rag=not args.no_rag,
                      top_k=args.top_k, device=args.device)
    if not args.debug_retrieval:
        result["retrieved_sources"] = [{key: row[key] for key in ("id", "source", "score", "title", "question")}
                                       for row in result["retrieved_sources"]]
    print(json.dumps(result, indent=2))

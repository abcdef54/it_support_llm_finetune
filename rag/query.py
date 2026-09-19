"""LangChain-based base/DEX generation with optional RAG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

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
    return AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        cache_dir=str(cache) if cache.exists() else None,
    )


def _messages_to_dicts(prompt_val):
    if hasattr(prompt_val, "to_messages"):
        messages = []
        for m in prompt_val.to_messages():
            role = "system" if m.type == "system" else "user" if m.type in ("human", "user") else "assistant"
            messages.append({"role": role, "content": m.content})
        return messages
    return prompt_val


def generate(root, question, *, model="dex", rag=True, top_k=C.RAG_TOP_K, device=None):
    if model not in {"base", "dex"}:
        raise ValueError("Model must be base or dex")

    settings = for_experiment("dex_v2")
    adapter_path = root / settings.ADAPTER_PATH
    adapter = (
        inspect_adapter(
            adapter_path,
            settings.BASE_MODEL_ID,
            settings.BASE_MODEL_REVISION,
            expected_experiment="dex_v2",
        )
        if model == "dex"
        else None
    )

    retrieved, context = [], None
    if rag:
        retriever = Retriever(root, device=device)
        retrieved = retriever.retrieve(question, top_k)
        context = build_context(question, retrieved, load_tokenizer(root))

    _set_seed(RANDOM_SEED)
    generator = (
        load_finetuned_generator(
            settings.BASE_MODEL_ID,
            settings.BASE_MODEL_REVISION,
            adapter_path,
            adapter["configuration"],
            autocast_adapter_dtype=settings.ADAPTER_AUTOCAST_DTYPE,
        )
        if adapter
        else QwenGenerator.load(settings.BASE_MODEL_ID, settings.BASE_MODEL_REVISION)
    )

    llm = RunnableLambda(lambda p: generator.generate(_messages_to_dicts(p), MAX_NEW_TOKENS)) | StrOutputParser()

    try:
        if rag and context:
            prompt = ChatPromptTemplate.from_messages([
                ("system", C.RAG_SYSTEM_PROMPT),
                ("human", "Retrieved technical context:\n{context}\n\nUser question:\n{question}"),
            ])
            chain = prompt | llm
            answer = chain.invoke({"context": context["context"] or "(none fits)", "question": question})
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", ANSWER_SYSTEM_PROMPT),
                ("human", "{question}"),
            ])
            chain = prompt | llm
            answer = chain.invoke({"question": question})

        return {
            "model": model,
            "rag": rag,
            "question": question,
            "retrieved_sources": retrieved,
            "context_used": context,
            "answer": answer,
        }
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
    result = generate(
        args.project_root.resolve(),
        args.question,
        model=args.model,
        rag=not args.no_rag,
        top_k=args.top_k,
        device=args.device,
    )
    if not args.debug_retrieval:
        result["retrieved_sources"] = [
            {key: row[key] for key in ("id", "source", "score", "title", "question")}
            for row in result["retrieved_sources"]
        ]
    print(json.dumps(result, indent=2))


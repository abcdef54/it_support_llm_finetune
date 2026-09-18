from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

from benchmark.baseline.config import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    ENABLE_THINKING,
    MAX_INPUT_TOKENS,
)
from benchmark.common.dataset import file_sha256, load_general_it
from rag import config as C
from rag.index import fingerprint, inspect_index


def retrieval_configuration(root: Path, index_manifest: dict) -> dict:
    return {
        "enabled": True,
        "corpus_sha256": index_manifest["configuration"]["corpus_sha256"],
        "index_fingerprint": fingerprint(index_manifest),
        "embedding_model": C.EMBEDDING_MODEL,
        "embedding_revision": C.EMBEDDING_REVISION,
        "benchmark_sha256": file_sha256(root / C.BENCHMARK_PATH),
        "top_k": C.RAG_TOP_K,
        "max_context_tokens": C.RAG_MAX_CONTEXT_TOKENS,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "context_policy": C.CONTEXT_POLICY,
        "rag_prompt_sha256": fingerprint(C.RAG_SYSTEM_PROMPT),
        "tokenizer": BASE_MODEL_ID,
        "tokenizer_revision": BASE_MODEL_REVISION,
        "transformers_version": version("transformers"),
        "tokenizers_version": version("tokenizers"),
        "enable_thinking": ENABLE_THINKING,
        "format_version": 2,
    }


def benchmark_examples(root: Path):
    manifest = json.loads((root / C.BENCHMARK_MANIFEST).read_text(encoding="utf-8"))
    return load_general_it(root / C.BENCHMARK_PATH, expected_sha256=manifest["benchmark_sha256"])


def load_artifact(root: Path) -> tuple[dict, dict]:
    _, _, index = inspect_index(root)
    path = root / C.RETRIEVAL_ARTIFACT
    if not path.exists():
        raise FileNotFoundError(f"Shared RAG contexts missing at {path}")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    expected = retrieval_configuration(root, index)
    if artifact["configuration"] != expected or artifact["records_sha256"] != fingerprint(artifact["records"]):
        raise ValueError("Shared RAG contexts are stale or changed")
    examples = benchmark_examples(root)
    records = artifact["records"]
    if (len(records) != len(examples) or any(row["id"] != example.id or row["question"] != example.question
                                          for row, example in zip(records, examples, strict=True))):
        raise ValueError("RAG retrieval artifact does not match benchmark questions/order")
    return {row["id"]: row for row in records}, {**expected, "retrieval_artifact_sha256": file_sha256(path)}

"""Save identical retrieval/context inputs for both RAG benchmark variants; no Qwen weights."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

from benchmark.baseline.config import BASE_MODEL_ID, BASE_MODEL_REVISION, MAX_INPUT_TOKENS, ENABLE_THINKING
from benchmark.common.dataset import file_sha256, load_general_it
from data.utils import write_json
from rag import config as C
from rag.context import build_context
from rag.index import fingerprint, inspect_index
from rag.retrieve import Retriever


def load_tokenizer(root):
    from transformers import AutoTokenizer
    cache = root / "data/raw/finetune_v2/tokenizer_cache"
    return AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION,
                                         cache_dir=str(cache) if cache.exists() else None)


def retrieval_configuration(root, index_manifest):
    return {"enabled": True, "corpus_sha256": index_manifest["configuration"]["corpus_sha256"],
            "index_fingerprint": fingerprint(index_manifest),
            "embedding_model": C.EMBEDDING_MODEL, "embedding_revision": C.EMBEDDING_REVISION,
            "benchmark_sha256": file_sha256(root / C.BENCHMARK_PATH),
            "top_k": C.RAG_TOP_K, "max_context_tokens": C.RAG_MAX_CONTEXT_TOKENS,
            "max_input_tokens": MAX_INPUT_TOKENS, "context_policy": C.CONTEXT_POLICY,
            "rag_prompt_sha256": fingerprint(C.RAG_SYSTEM_PROMPT),
            "tokenizer": BASE_MODEL_ID, "tokenizer_revision": BASE_MODEL_REVISION,
            "transformers_version": version("transformers"), "tokenizers_version": version("tokenizers"),
            "enable_thinking": ENABLE_THINKING, "format_version": 2}


def benchmark_examples(root):
    manifest = json.loads((root / C.BENCHMARK_MANIFEST).read_text())
    return load_general_it(root / C.BENCHMARK_PATH, expected_sha256=manifest["benchmark_sha256"])


def load_artifact(root):
    _, _, index = inspect_index(root)
    path = root / C.RETRIEVAL_ARTIFACT
    if not path.exists():
        raise FileNotFoundError("Shared RAG contexts missing; run python -m rag.prepare_benchmark")
    artifact = json.loads(path.read_text())
    expected = retrieval_configuration(root, index)
    if artifact["configuration"] != expected or artifact["records_sha256"] != fingerprint(artifact["records"]):
        raise ValueError("Shared RAG contexts are stale or changed; rerun python -m rag.prepare_benchmark --overwrite")
    examples = benchmark_examples(root)
    records = artifact["records"]
    if (len(records) != len(examples) or any(row["id"] != example.id or row["question"] != example.question
                                          for row, example in zip(records, examples, strict=True))):
        raise ValueError("RAG retrieval artifact does not match benchmark questions/order")
    return {row["id"]: row for row in records}, {**expected, "retrieval_artifact_sha256": file_sha256(path)}


def prepare(root, *, device=None, overwrite=False):
    path = root / C.RETRIEVAL_ARTIFACT
    if path.exists() and not overwrite:
        records, metadata = load_artifact(root)
        return {"examples": len(records), **metadata}
    retriever = Retriever(root, device=device)
    try:
        from tqdm import tqdm
        tokenizer = load_tokenizer(root)
        config = retrieval_configuration(root, retriever.manifest)
        records = []
        for example in tqdm(benchmark_examples(root), desc="Preparing shared retrieval contexts"):
            retrieved = retriever.retrieve(example.question)
            context = build_context(example.question, retrieved, tokenizer)
            records.append({"id": example.id, "question": example.question, **context,
                            "retrieved_documents": [{"id": r["id"], "source": r["source"], "score": r["score"]}
                                                    for r in retrieved]})
        artifact = {"configuration": config, "records_sha256": fingerprint(records), "records": records,
                    "embedding_truncated_queries": retriever.embedder.truncated_queries}
        temporary = path.with_suffix(".tmp")
        write_json(temporary, artifact)
        temporary.replace(path)
        return {"examples": len(records), "path": C.RETRIEVAL_ARTIFACT, "sha256": file_sha256(path)}
    finally:
        retriever.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=C.PROJECT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.project_root.resolve(), device=args.device, overwrite=args.overwrite), indent=2))

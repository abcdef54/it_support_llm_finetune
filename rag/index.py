"""Explicit BGE embedding and repeatable persistent Chroma indexing."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from benchmark.common.generation import batches
from data.utils import write_json
from rag import config as C
from rag.build_corpus import validate_corpus
from rag.embeddings import Embedder


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def index_configuration(manifest):
    return {"schema_version": 1, "corpus_sha256": manifest["files"]["combined_corpus.jsonl"]["sha256"],
            "benchmark_sha256": manifest["leakage_checks"]["benchmark_sha256"],
            "embedding_model": C.EMBEDDING_MODEL, "embedding_revision": C.EMBEDDING_REVISION,
            "max_embedding_tokens": C.EMBEDDING_MAX_TOKENS, "normalize_embeddings": True,
            "precision_policy": C.EMBEDDING_PRECISION, "insertion_order": "retrieval_text_length_then_id",
            "query_instruction": C.QUERY_INSTRUCTION, "distance_metric": C.CHROMA_DISTANCE_METRIC,
            "collection": C.COLLECTION_NAME, "source_counts": manifest["source_counts"],
            "documents": manifest["total_records"], "chromadb_version": version("chromadb"),
            "sentence_transformers_version": version("sentence-transformers"),
            "hnsw": {"space": "cosine", "num_threads": 1, "ef_construction": 100, "ef_search": 100}}


def client_for(root):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(root / C.INDEX_PATH), settings=Settings(anonymized_telemetry=False))


def inspect_index(root):
    records, corpus = validate_corpus(root)
    config = index_configuration(corpus)
    path = root / C.INDEX_PATH / "manifest.json"
    if not path.exists():
        raise FileNotFoundError("RAG index missing or incomplete; run python -m rag.index")
    manifest = json.loads(path.read_text())
    if manifest.get("fingerprint") != fingerprint(config) or manifest.get("configuration") != config:
        raise ValueError("RAG index is stale; run python -m rag.index --rebuild")
    client = client_for(root)
    if C.COLLECTION_NAME not in {c.name for c in client.list_collections()}:
        raise FileNotFoundError("Chroma collection missing; run python -m rag.index")
    collection = client.get_collection(C.COLLECTION_NAME, embedding_function=None)
    if collection.metadata.get("fingerprint") != manifest["fingerprint"] or collection.count() != len(records):
        raise ValueError("Chroma index count/fingerprint differs from manifest; run python -m rag.index")
    return client, collection, manifest


def build_index(root, *, device=None, batch_size=C.EMBEDDING_BATCH_SIZE, rebuild=False, embedder=None):
    if batch_size < 1:
        raise ValueError("Embedding batch size must be positive")
    records, corpus = validate_corpus(root)
    config = index_configuration(corpus)
    identity = fingerprint(config)
    client = client_for(root)
    existing = {collection.name for collection in client.list_collections()}
    if C.COLLECTION_NAME in existing:
        collection = client.get_collection(C.COLLECTION_NAME, embedding_function=None)
        if rebuild:
            client.delete_collection(C.COLLECTION_NAME)
        elif collection.metadata.get("fingerprint") != identity:
            raise ValueError("Existing index is stale; use --rebuild to replace only the derived RAG collection")
        elif (root / C.INDEX_PATH / "manifest.json").exists():
            return inspect_index(root)[2]
    manifest_path = root / C.INDEX_PATH / "manifest.json"
    manifest_path.unlink(missing_ok=True)  # An interrupted build must never look complete.
    collection = client.get_or_create_collection(C.COLLECTION_NAME, embedding_function=None,
                                                 metadata={"fingerprint": identity},
                                                 configuration={"hnsw": config["hnsw"]})
    owned = embedder is None
    embedder = embedder or Embedder(root, device=device, batch_size=batch_size)
    try:
        import numpy as np
        from tqdm import tqdm
        records.sort(key=lambda r: (len(r["retrieval_text"]), r["id"]))
        vector_digest = hashlib.sha256()
        truncated_documents = 0
        for batch in tqdm(list(batches(records, batch_size)), desc="Embedding RAG corpus", mininterval=10):
            # For a fixed corpus fingerprint existing IDs are immutable. Reuse
            # their actual vectors, including when recovering an interrupted build.
            saved = collection.get(ids=[r["id"] for r in batch], include=["embeddings", "metadatas"])
            vectors_by_id = dict(zip(saved["ids"], saved["embeddings"], strict=True))
            truncated_documents += sum(m.get("embedding_truncated", False) for m in saved["metadatas"])
            missing = [r for r in batch if r["id"] not in vectors_by_id]
            if missing:
                new_vectors = embedder.encode([r["retrieval_text"] for r in missing])
                if (new_vectors.shape != (len(missing), embedder.dimension) or not np.isfinite(new_vectors).all()
                        or not np.allclose(np.linalg.norm(new_vectors, axis=1), 1, atol=1e-4)):
                    raise ValueError("Invalid, unnormalized or dimension-mismatched embeddings")
                flags = getattr(embedder, "last_truncation_flags", [False] * len(missing))
                collection.upsert(ids=[r["id"] for r in missing], embeddings=new_vectors.tolist(),
                                  documents=[json.dumps(r, ensure_ascii=False) for r in missing],
                                  metadatas=[{"source": r["source"], "source_id": r["id"], "embedding_truncated": flag}
                                             for r, flag in zip(missing, flags, strict=True)])
                vectors_by_id.update(zip([r["id"] for r in missing], new_vectors, strict=True))
                truncated_documents += sum(flags)
            vectors = np.asarray([vectors_by_id[r["id"]] for r in batch], dtype=np.float32)
            if (vectors.shape != (len(batch), embedder.dimension) or not np.isfinite(vectors).all()
                    or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
                raise ValueError("Invalid, unnormalized or dimension-mismatched embeddings")
            vector_digest.update(vectors.astype("<f4").tobytes())
        if collection.count() != len(records):
            raise ValueError("Chroma index count differs from corpus")
        manifest = {"fingerprint": identity, "configuration": config, "embedding_dimension": embedder.dimension,
                    "vectors_sha256": vector_digest.hexdigest(),
                    "embedding_device": str(getattr(getattr(embedder, "model", None), "device", device)),
                    "batch_size": batch_size, "embedding_truncated_documents": truncated_documents,
                    "truncation_policy": "BGE first 512 tokens only; stored Q&A/context remains complete"}
        write_json(manifest_path, manifest)
        return manifest
    finally:
        if owned:
            embedder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=C.PROJECT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=C.EMBEDDING_BATCH_SIZE)
    parser.add_argument("--rebuild", action="store_true", help="Replace only this derived RAG collection")
    parser.add_argument("--check", action="store_true", help="Validate existing corpus/index without loading embeddings")
    args = parser.parse_args()
    result = inspect_index(args.project_root.resolve())[2] if args.check else build_index(
        args.project_root.resolve(), device=args.device, batch_size=args.batch_size, rebuild=args.rebuild)
    print(json.dumps(result, indent=2))

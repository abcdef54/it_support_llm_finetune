"""Build and inspect the Chroma vectorstore index using LangChain."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from tqdm import tqdm

from data.utils import read_jsonl
from rag import config as C
from rag.embeddings import get_embeddings


def get_chroma_client(project_root: Path = C.PROJECT_ROOT) -> chromadb.PersistentClient:
    """Create a persistent Chroma client pointing to the index directory."""
    index_directory = Path(project_root) / C.INDEX_PATH
    return chromadb.PersistentClient(
        path=str(index_directory),
        settings=Settings(anonymized_telemetry=False),
    )


def calculate_fingerprint(data: Any) -> str:
    """Compute a deterministic SHA-256 fingerprint of any JSON-serializable structure."""
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# Alias for backwards compatibility with evaluation scripts
fingerprint = calculate_fingerprint


def inspect_index(project_root: Path = C.PROJECT_ROOT) -> tuple[chromadb.PersistentClient, Any, dict]:
    """Inspect the existing vectorstore collection and return client, collection, and manifest."""
    root_path = Path(project_root)
    client = get_chroma_client(root_path)

    existing_collections = {collection.name for collection in client.list_collections()}
    if C.COLLECTION_NAME not in existing_collections:
        raise FileNotFoundError(f"Chroma collection '{C.COLLECTION_NAME}' does not exist.")

    collection = client.get_collection(C.COLLECTION_NAME)
    manifest_path = root_path / C.INDEX_PATH / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    return client, collection, manifest


def load_corpus_records(corpus_path: Path) -> list[dict]:
    """Load knowledge corpus records from a JSONL file."""
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    return read_jsonl(corpus_path)


def build_vector_index(
    project_root: Path = C.PROJECT_ROOT,
    *,
    batch_size: int = C.EMBEDDING_BATCH_SIZE,
    device: str | None = None,
    rebuild: bool = False,
) -> dict:
    """Index knowledge records into Chroma using LangChain embeddings."""
    root_path = Path(project_root)
    corpus_path = root_path / C.CORPUS_PATH
    records = load_corpus_records(corpus_path)

    client = get_chroma_client(root_path)
    existing_collections = {c.name for c in client.list_collections()}

    if rebuild and C.COLLECTION_NAME in existing_collections:
        client.delete_collection(C.COLLECTION_NAME)

    embedding_model = get_embeddings(root_path, device=device, batch_size=batch_size)
    vectorstore = Chroma(
        client=client,
        collection_name=C.COLLECTION_NAME,
        embedding_function=embedding_model,
    )

    print(f"Indexing {len(records)} knowledge documents into collection '{C.COLLECTION_NAME}'...")
    for start_idx in tqdm(range(0, len(records), batch_size), desc="Indexing batches"):
        batch = records[start_idx : start_idx + batch_size]
        saved = vectorstore._collection.get(ids=[item["id"] for item in batch])
        already_indexed_ids = set(saved["ids"])

        unindexed_items = [item for item in batch if item["id"] not in already_indexed_ids]
        if unindexed_items:
            retrieval_passages = [item["retrieval_text"] for item in unindexed_items]
            embeddings = embedding_model.embed_documents(retrieval_passages)

            vectorstore._collection.upsert(
                ids=[item["id"] for item in unindexed_items],
                embeddings=embeddings,
                documents=[json.dumps(item, ensure_ascii=False) for item in unindexed_items],
                metadatas=[{"source": item["source"], "source_id": item["id"]} for item in unindexed_items],
            )

    manifest = {
        "fingerprint": calculate_fingerprint({"collection": C.COLLECTION_NAME, "total_records": len(records)}),
        "configuration": {
            "collection": C.COLLECTION_NAME,
            "documents": len(records),
            "embedding_model": C.EMBEDDING_MODEL,
        },
        "embedding_dimension": 768,
    }
    manifest_path = root_path / C.INDEX_PATH / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


# Alias for backwards compatibility
build_index = build_vector_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=C.PROJECT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=C.EMBEDDING_BATCH_SIZE)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector collection from scratch")
    parser.add_argument("--check", action="store_true", help="Inspect existing vectorstore without re-indexing")
    args = parser.parse_args()

    resolved_root = args.project_root.resolve()
    if args.check:
        _, collection, manifest = inspect_index(resolved_root)
        print(json.dumps({"collection": collection.name, "count": collection.count(), "manifest": manifest}, indent=2))
    else:
        result = build_vector_index(resolved_root, batch_size=args.batch_size, device=args.device, rebuild=args.rebuild)
        print(json.dumps(result, indent=2))


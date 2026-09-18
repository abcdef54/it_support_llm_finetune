"""Retrieve structured knowledge records without loading Qwen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag import config as C
from rag.embeddings import Embedder
from rag.index import inspect_index


class Retriever:
    def __init__(self, root=C.PROJECT_ROOT, *, device=None, embedder=None):
        self.client, self.collection, self.manifest = inspect_index(root)
        self.embedder = embedder or Embedder(root, device=device)
        if self.embedder.dimension != self.manifest["embedding_dimension"]:
            self.embedder.close()
            raise ValueError("Embedding dimension differs from the Chroma index")

    def retrieve(self, query, top_k=C.RAG_TOP_K, *, source=None):
        if not isinstance(query, str) or not query.strip() or top_k < 1:
            raise ValueError("A nonempty query and positive top_k are required")
        if source not in {None, "stackoverflow", "techqa"}:
            raise ValueError("Unknown source filter")
        count = self.manifest["configuration"]["source_counts"].get(source, self.collection.count())
        vectors = self.embedder.encode([query], query=True)
        if vectors.shape != (1, self.manifest["embedding_dimension"]):
            raise ValueError("Query embedding dimension mismatch")
        result = self.collection.query(query_embeddings=vectors.tolist(), n_results=min(top_k, count),
                                       where={"source": source} if source else None,
                                       include=["documents", "metadatas", "distances"])
        records = []
        for record_id, raw, metadata, distance in zip(result["ids"][0], result["documents"][0],
                                                      result["metadatas"][0], result["distances"][0], strict=True):
            record = json.loads(raw)
            if record["id"] != record_id or record["source"] != metadata["source"]:
                raise ValueError("Chroma document/source metadata mismatch")
            records.append({**record, "distance": float(distance), "score": 1.0 - float(distance)})
        return sorted(records, key=lambda r: (r["distance"], r["id"]))

    def close(self):
        self.embedder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=C.PROJECT_ROOT)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=C.RAG_TOP_K)
    parser.add_argument("--source", choices=("stackoverflow", "techqa"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    retriever = Retriever(args.project_root.resolve(), device=args.device)
    try:
        print(json.dumps({"query": args.query, "results": retriever.retrieve(args.query, args.top_k, source=args.source)}, indent=2))
    finally:
        retriever.close()

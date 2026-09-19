"""Retrieve structured knowledge records using LangChain Chroma."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_chroma import Chroma

from rag import config as C
from rag.embeddings import get_embeddings


class Retriever:
    """Retrieve structured knowledge records using LangChain Chroma."""

    def __init__(self, root=C.PROJECT_ROOT, *, device=None, embeddings=None, embedder=None):
        self.root = Path(root)
        manifest_path = self.root / C.INDEX_PATH / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(
            path=str(self.root / C.INDEX_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        self.embeddings = embeddings or (
            embedder.embeddings if hasattr(embedder, "embeddings") else get_embeddings(self.root, device=device)
        )
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=C.COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

    def retrieve(self, query: str, top_k: int = C.RAG_TOP_K, *, source: str | None = None) -> list[dict]:
        if not isinstance(query, str) or not query.strip() or top_k < 1:
            raise ValueError("A nonempty query and positive top_k are required")
        if source not in {None, "stackoverflow", "techqa"}:
            raise ValueError("Unknown source filter")

        filter_dict = {"source": source} if source else None
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=top_k, filter=filter_dict
        )
        records = []
        for doc, score in results:
            record = json.loads(doc.page_content)
            record["score"] = float(score)
            record["distance"] = 1.0 - float(score)
            records.append(record)
        return sorted(records, key=lambda r: (r["distance"], r["id"]))

    def as_retriever(self, **kwargs):
        """Expose standard LangChain retriever interface."""
        return self.vectorstore.as_retriever(**kwargs)

    def close(self):
        pass


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


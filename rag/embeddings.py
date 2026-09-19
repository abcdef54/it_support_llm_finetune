from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from rag import config as C


class BGEEmbeddings(HuggingFaceEmbeddings):
    """BGE embedding model with query instruction support."""

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(C.QUERY_INSTRUCTION + text)


def get_embeddings(
    root=C.PROJECT_ROOT,
    device: str | None = None,
    batch_size: int = C.EMBEDDING_BATCH_SIZE,
) -> BGEEmbeddings:
    """Create an accelerated BGE embedding instance with FP16 and batching."""
    import torch

    local = root / C.EMBEDDING_LOCAL
    model_name = str(local) if local.exists() else C.EMBEDDING_MODEL

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if resolved_device == "cuda" else torch.float32

    return BGEEmbeddings(
        model_name=model_name,
        model_kwargs={"device": resolved_device, "model_kwargs": {"torch_dtype": dtype}},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )


class Embedder:
    """LangChain-backed embedding wrapper for backward compatibility."""

    def __init__(self, root=C.PROJECT_ROOT, *, device=None, batch_size=C.EMBEDDING_BATCH_SIZE):
        self.embeddings = get_embeddings(root, device=device)
        self.dimension = 768
        self.batch_size = batch_size
        self.truncated_documents = 0
        self.truncated_queries = 0
        self.last_truncation_flags = []

    def encode(self, texts: list[str], *, query: bool = False):
        import numpy as np

        if query:
            vectors = np.array([self.embeddings.embed_query(t) for t in texts], dtype=np.float32)
        else:
            vectors = np.array(self.embeddings.embed_documents(texts), dtype=np.float32)
        self.last_truncation_flags = [False] * len(texts)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def close(self):
        pass


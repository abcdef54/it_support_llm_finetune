from __future__ import annotations

from rag import config as C


class Embedder:
    def __init__(self, root=C.PROJECT_ROOT, *, device=None, batch_size=C.EMBEDDING_BATCH_SIZE):
        import torch
        from sentence_transformers import SentenceTransformer

        if batch_size < 1:
            raise ValueError("Embedding batch size must be positive")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        local = root / C.EMBEDDING_LOCAL
        # HF local_dir download metadata records the exact snapshot commit.
        if local.exists():
            stamp = local / ".cache/huggingface/download/model.safetensors.metadata"
            if not stamp.exists() or stamp.read_text().splitlines()[0] != C.EMBEDDING_REVISION:
                raise ValueError("Local BGE snapshot does not match the pinned revision")
        self.model = SentenceTransformer(str(local) if local.exists() else C.EMBEDDING_MODEL,
                                         revision=None if local.exists() else C.EMBEDDING_REVISION,
                                         device=device, model_kwargs={"torch_dtype": dtype})
        self.model.max_seq_length = C.EMBEDDING_MAX_TOKENS
        self.batch_size = batch_size
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.truncated_documents = 0
        self.truncated_queries = 0

    def encode(self, texts, *, query=False):
        if query:
            texts = [C.QUERY_INSTRUCTION + text for text in texts]
        lengths = self.model.tokenizer(texts, truncation=False, padding=False, verbose=False)["input_ids"]
        self.last_truncation_flags = [len(ids) > C.EMBEDDING_MAX_TOKENS for ids in lengths]
        truncated = sum(self.last_truncation_flags)
        if query:
            self.truncated_queries += truncated
        else:
            self.truncated_documents += truncated
        import numpy as np
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def close(self):
        import gc
        import torch
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

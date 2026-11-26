
from __future__ import annotations

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class STEncoder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None,
                 max_seq_length: int = 256, fp16: bool = True, query_prefix: str = BGE_QUERY_PREFIX,
                 batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        # fp16 inference is fine on Turing (1660 Ti); it halves VRAM and is faster.
        if fp16 and str(self.model.device).startswith("cuda"):
            self.model.half()
        self.query_prefix = query_prefix
        self.batch_size = batch_size

    def _enc(self, texts: list[str], show: bool) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=show)

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self._enc(texts, show=len(texts) > 1000)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._enc([self.query_prefix + t for t in texts], show=False)

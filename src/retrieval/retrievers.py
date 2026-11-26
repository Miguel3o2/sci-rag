
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


class BM25Retriever:
    def __init__(self, texts: list[str]):
        import bm25s

        self._bm25s = bm25s
        self.n = len(texts)
        self.model = bm25s.BM25()
        self.model.index(bm25s.tokenize(texts, stopwords="en", show_progress=False),
                         show_progress=False)

    def search(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        k = min(k, self.n)
        qt = self._bm25s.tokenize(queries, stopwords="en", show_progress=False)
        idx, sc = self.model.retrieve(qt, k=k, show_progress=False)
        return [[(int(i), float(s)) for i, s in zip(ri, rs)] for ri, rs in zip(idx, sc)]


class DenseRetriever:
    def __init__(self, encoder, texts: list[str], cache_dir: str | Path = "data/index",
                 shard_size: int = 20_000, search_block: int = 50_000):
        self.encoder = encoder
        self.n = len(texts)
        self.search_block = search_block
        h = hashlib.md5()
        for t in texts:
            h.update(t.encode("utf-8"))
        slug = re.sub(r"[^A-Za-z0-9]+", "_", encoder.name)
        self.dir = Path(cache_dir) / f"{slug}__{h.hexdigest()[:10]}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._build(texts, shard_size)
        shards = sorted(self.dir.glob("shard_*.npy"))
        self.shards = [np.load(p, mmap_mode="r") for p in shards]
        assert sum(len(s) for s in self.shards) == self.n, "cache/shard mismatch; delete " + str(self.dir)

    def _build(self, texts: list[str], shard_size: int) -> None:
        # Resumable: finished shards are skipped, so an overnight job can be interrupted.
        for si, start in enumerate(range(0, len(texts), shard_size)):
            path = self.dir / f"shard_{si:05d}.npy"
            if path.exists():
                continue
            emb = self.encoder.encode_docs(texts[start:start + shard_size])
            np.save(path.with_suffix(".tmp.npy"), np.asarray(emb, dtype=np.float16))
            path.with_suffix(".tmp.npy").replace(path)
            print(f"embedded {min(start + shard_size, len(texts))}/{len(texts)}", flush=True)

    def search(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        q = np.asarray(self.encoder.encode_queries(queries), dtype=np.float32)
        k = min(k, self.n)
        cand_s, cand_i = [], []
        offset = 0
        for shard in self.shards:
            for b in range(0, len(shard), self.search_block):
                blk = np.asarray(shard[b:b + self.search_block], dtype=np.float32)
                s = q @ blk.T
                kk = min(k, s.shape[1])
                part = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
                cand_s.append(np.take_along_axis(s, part, axis=1))
                cand_i.append(part + offset + b)
            offset += len(shard)
        S = np.concatenate(cand_s, axis=1)
        I = np.concatenate(cand_i, axis=1)
        order = np.argsort(-S, axis=1)[:, :k]
        S = np.take_along_axis(S, order, axis=1)
        I = np.take_along_axis(I, order, axis=1)
        return [[(int(i), float(s)) for i, s in zip(ri, rs)] for ri, rs in zip(I, S)]


class HybridRetriever:
    """Reciprocal rank fusion over any number of retrievers."""

    def __init__(self, retrievers: list, rrf_k: int = 60, depth: int = 100):
        self.retrievers, self.rrf_k, self.depth = retrievers, rrf_k, depth

    def search(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        runs = [r.search(queries, self.depth) for r in self.retrievers]
        out = []
        for qi in range(len(queries)):
            agg: dict[int, float] = defaultdict(float)
            for run in runs:
                for rank, (idx, _) in enumerate(run[qi]):
                    agg[idx] += 1.0 / (self.rrf_k + rank + 1)
            out.append(sorted(agg.items(), key=lambda x: -x[1])[:k])
        return out

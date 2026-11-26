
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.eval.metrics import hit_at_k, ndcg_at_k, reciprocal_rank

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tok_set(s: str) -> set[str]:
    return set(TOKEN_RE.findall(s.lower()))


def build_gold(texts: list[str], pmcids: list[str], questions: list[dict],
               min_cov: float = 0.8) -> dict[str, set[int]]:
    by_doc: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pmcids):
        by_doc[p].append(i)
    gold: dict[str, set[int]] = {}
    for q in questions:
        ev = tok_set(q["evidence"])
        rel: set[int] = set()
        if len(ev) >= 5:
            for i in by_doc.get(q["pmcid"], []):
                if len(ev & tok_set(texts[i])) / len(ev) >= min_cov:
                    rel.add(i)
        gold[q["qid"]] = rel
    return gold


def evaluate(results: list[list[tuple[int, float]]], questions: list[dict],
             gold: dict[str, set[int]], ks=(1, 5, 10, 20)) -> tuple[dict, list[dict]]:
    per_q, agg = [], defaultdict(list)
    for q, ranked in zip(questions, results):
        g = gold[q["qid"]]
        if not g:
            per_q.append({"qid": q["qid"], "unmatched": True})
            continue
        rels = [i in g for i, _ in ranked]
        row = {"qid": q["qid"], "qtype": q.get("qtype", ""), "n_relevant": len(g),
               "first_rel_rank": next((r for r, ok in enumerate(rels, 1) if ok), None)}
        for k in ks:
            row[f"hit@{k}"] = hit_at_k(rels, k)
            agg[f"hit@{k}"].append(row[f"hit@{k}"])
        row["mrr@10"] = reciprocal_rank(rels, 10)
        row["ndcg@10"] = ndcg_at_k(rels, 10, len(g))
        agg["mrr@10"].append(row["mrr@10"])
        agg["ndcg@10"].append(row["ndcg@10"])
        per_q.append(row)
    metrics = {k: float(np.mean(v)) for k, v in agg.items()}
    metrics["n_questions"] = len(questions)
    metrics["n_unmatched"] = sum(1 for r in per_q if r.get("unmatched"))
    return metrics, per_q


def make_retriever(args, texts: list[str]):
    from src.retrieval.retrievers import BM25Retriever, DenseRetriever, HybridRetriever

    if args.retriever == "bm25":
        return BM25Retriever(texts)
    from src.retrieval.encoders import STEncoder
    enc = STEncoder(args.model, query_prefix=args.query_prefix)
    dense = DenseRetriever(enc, texts, cache_dir=args.cache_dir)
    if args.retriever == "dense":
        return dense
    return HybridRetriever([BM25Retriever(texts), dense])


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="bm25")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--query-prefix", default="Represent this sentence for searching relevant passages: ")
    ap.add_argument("--cache-dir", default="data/index")
    ap.add_argument("--no-header", action="store_true", help="index chunk text without title/section header")
    ap.add_argument("--min-cov", type=float, default=0.8)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    chunks = pd.read_parquet(args.chunks)
    raw = chunks["text"].tolist()
    texts = raw if args.no_header else (chunks["header"] + "\n" + chunks["text"]).tolist()
    questions = [json.loads(l) for l in open(args.eval, encoding="utf-8") if l.strip()]

    # Relevance is always judged on raw text, never on header-augmented text.
    gold = build_gold(raw, chunks["pmcid"].tolist(), questions, args.min_cov)

    retriever = make_retriever(args, texts)
    t0 = time.perf_counter()
    results = retriever.search([q["question"] for q in questions], args.depth)
    ms_per_q = 1000 * (time.perf_counter() - t0) / max(len(questions), 1)

    metrics, per_q = evaluate(results, questions, gold)
    metrics["search_ms_per_query"] = ms_per_q
    print(f"{args.retriever} | {Path(args.chunks).stem} | header={'off' if args.no_header else 'on'}")
    for k, v in metrics.items():
        print(f"  {k:>20}: {v:.4f}" if isinstance(v, float) else f"  {k:>20}: {v}")
    if metrics["n_unmatched"]:
        print(f"  WARNING: {metrics['n_unmatched']} questions have no chunk covering their evidence "
              f"(bad label, or evidence fell in a dropped section / sub-{20}-word chunk)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "metrics": metrics, "per_question": per_q}, f, indent=2)


if __name__ == "__main__":
    main()

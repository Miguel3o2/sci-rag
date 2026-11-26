
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
SKIP_SECTION_RE = re.compile(
    r"acknowledg|funding|conflict|competing|author contrib|data availab|ethic|"
    r"abbreviation|supplement|references|disclosure", re.I)


def sentences(text: str) -> list[str]:
    return [s for s in SENT_RE.split(text) if s]


def _words(s: str) -> int:
    return len(s.split())


def kept_sections(article: dict) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    if article.get("abstract"):
        out.append(("Abstract", article["abstract"]))
    for s in article["sections"]:
        if SKIP_SECTION_RE.search(s["title"]):
            continue
        out.append((s["title"] or "Body", s["paragraphs"]))
    return out


def _pack(units: list[str], max_words: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for u in units:
        w = _words(u)
        if cur and n + w > max_words:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(u)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _row(article: dict, strategy: str, section: str, text: str) -> dict:
    header = f"{article['title']} | {section}" if section else article["title"]
    return {"pmcid": article["pmcid"], "strategy": strategy, "section": section,
            "header": header, "text": text, "n_words": _words(text)}


def _number(rows: list[dict]) -> list[dict]:
    for i, r in enumerate(rows):
        r["chunk_id"] = f"{r['pmcid']}:{r['strategy']}:{i}"
    return rows


def chunk_fixed(article: dict, size: int = 200, overlap: int = 40, min_words: int = 20) -> list[dict]:
    words: list[str] = []
    for _, paras in kept_sections(article):
        for p in paras:
            words.extend(p.split())
    step = size - overlap
    rows = []
    for start in range(0, max(len(words), 1), step):
        w = words[start:start + size]
        if len(w) >= min_words:
            rows.append(_row(article, "fixed", "", " ".join(w)))
        if start + size >= len(words):
            break
    return _number(rows)


def chunk_section(article: dict, max_words: int = 200, min_words: int = 20) -> list[dict]:
    rows = []
    for title, paras in kept_sections(article):
        units: list[str] = []
        for p in paras:
            units.extend(sentences(p) if _words(p) > max_words else [p])
        packs = _pack(units, max_words)
        if len(packs) > 1 and _words(packs[-1]) < min_words:
            tail = packs.pop()
            packs[-1] += " " + tail
        for t in packs:
            if _words(t) >= min_words:
                rows.append(_row(article, "section", title, t))
    return _number(rows)


def chunk_semantic(article: dict, encoder, max_words: int = 250, min_words: int = 20,
                   percentile: float = 15.0) -> list[dict]:
    """Break where adjacent-sentence cosine similarity falls below a per-article percentile.

    `encoder` needs .encode_docs(list[str]) -> normalized float array.
    Ignores section boundaries on purpose (that is the contrast with `section`).
    """
    sents: list[tuple[str, str]] = []
    for title, paras in kept_sections(article):
        for p in paras:
            sents.extend((title, s) for s in sentences(p))
    if len(sents) < 2:
        return []
    emb = np.asarray(encoder.encode_docs([s for _, s in sents]), dtype=np.float32)
    sims = (emb[:-1] * emb[1:]).sum(axis=1)
    thr = float(np.percentile(sims, percentile))

    rows: list[dict] = []
    cur: list[str] = []
    n = 0
    sec = ""

    def flush() -> None:
        nonlocal cur, n
        if cur and n >= min_words:
            rows.append(_row(article, "semantic", sec, " ".join(cur)))
        cur, n = [], 0

    for i, (title, s) in enumerate(sents):
        w = _words(s)
        if cur and (n + w > max_words or sims[i - 1] < thr):
            flush()
        if not cur:
            sec = title
        cur.append(s)
        n += w
    flush()
    return _number(rows)


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", required=True)
    ap.add_argument("--strategy", choices=["fixed", "section", "semantic"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-words", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="semantic only")
    ap.add_argument("--max-articles", type=int, default=None)
    args = ap.parse_args()

    encoder = None
    if args.strategy == "semantic":
        from src.retrieval.encoders import STEncoder
        encoder = STEncoder(args.model, max_seq_length=128)

    rows: list[dict] = []
    with open(args.articles, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_articles and i >= args.max_articles:
                break
            art = json.loads(line)
            if args.strategy == "fixed":
                rows += chunk_fixed(art, args.max_words, args.overlap)
            elif args.strategy == "section":
                rows += chunk_section(art, args.max_words)
            else:
                rows += chunk_semantic(art, encoder, args.max_words)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"{len(df)} chunks from {df.pmcid.nunique()} articles; "
          f"mean words {df.n_words.mean():.0f} -> {args.out}")


if __name__ == "__main__":
    main()

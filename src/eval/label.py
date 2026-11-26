
from __future__ import annotations

import argparse
import json
import random
import re
import textwrap
from pathlib import Path

from src.ingest.chunk import sentences

QTYPES = ["factoid", "numeric", "yesno", "list", "mechanism", "comparison", "other"]


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", default="data/eval/questions.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-words", type=int, default=80)
    args = ap.parse_args()

    df = pd.read_parquet(args.chunks)
    df = df[(df.n_words >= args.min_words) & (df.section != "Abstract")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()] if out.exists() else []
    seen_chunks = {d.get("chunk_id") for d in done}
    n = len(done)

    order = list(df.index)
    random.Random(args.seed).shuffle(order)
    for i in order:
        row = df.loc[i]
        if row.chunk_id in seen_chunks:
            continue
        sents = sentences(row.text)
        print("\n" + "=" * 90)
        print(f"{row.pmcid} | {row.section}\n")
        for k, s in enumerate(sents, 1):
            print(textwrap.fill(f"[{k}] {s}", 100, subsequent_indent="     "))
        cmd = input("\n[Enter]=write question  s=skip  q=quit > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            continue
        question = input("question (self-contained, paraphrased): ").strip()
        if not question:
            print("  empty question, skipping")
            continue
        while True:
            pick = input(f"evidence sentence number(s), e.g. 3 or 3,4 [1-{len(sents)}]: ")
            try:
                idx = sorted({int(x) for x in re.split(r"[,\s]+", pick.strip()) if x})
            except ValueError:
                idx = []
            if not idx or not all(1 <= k <= len(sents) for k in idx):
                print("  -> enter valid sentence numbers")
                continue
            if idx[-1] - idx[0] + 1 != len(idx):
                print("  -> pick adjacent sentences only (evidence must sit inside one chunk)")
                continue
            evidence = " ".join(sents[k - 1] for k in idx)
            if len(evidence.split()) < 5:
                print("  -> evidence too short; include another sentence")
                continue
            break
        answer = input("short answer: ").strip()
        qtype = input(f"type {QTYPES} [factoid]: ").strip() or "factoid"
        n += 1
        rec = {"qid": f"q{n:04d}", "question": question, "pmcid": row.pmcid,
               "evidence": evidence, "answer": answer, "qtype": qtype, "chunk_id": row.chunk_id}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"saved {rec['qid']} ({n} total)")


if __name__ == "__main__":
    main()

import json

import numpy as np
import pytest
from lxml import etree

from src.eval.harness import build_gold, evaluate
from src.eval.metrics import hit_at_k, ndcg_at_k, reciprocal_rank
from src.ingest.chunk import chunk_fixed, chunk_section, chunk_semantic
from src.ingest.parse_pmc import parse_article

FILLER = ("The cohort was followed for twelve months and outcomes were recorded at each visit. " * 12)


def jats(pmc: str, journal: str, topic_sentence: str) -> bytes:
    return f"""<article><front>
      <journal-meta><journal-title-group><journal-title>{journal}</journal-title></journal-title-group></journal-meta>
      <article-meta><article-id pub-id-type="pmc">{pmc}</article-id>
        <title-group><article-title>Study of {pmc}</article-title></title-group>
        <abstract><p>Short abstract about {pmc} with enough words to count as a paragraph of text here.</p></abstract>
      </article-meta></front>
      <body>
        <sec><title>Introduction</title><p>{FILLER} Prior work<xref ref-type="bibr" rid="b1">[1]</xref> exists.</p></sec>
        <sec><title>Results</title>
          <p>{topic_sentence} {FILLER}</p>
          <fig id="f1"><caption><p>FIGURE CAPTION NOISE</p></caption></fig>
          <sec><title>Subgroup</title><p>{FILLER}</p></sec>
        </sec>
        <sec><title>Funding</title><p>{FILLER}</p></sec>
      </body></article>""".encode()


@pytest.fixture
def articles():
    specs = [
        ("1001", "Cancer Research", "Treatment with zorvelimab reduced tumor volume by 41 percent in murine xenografts."),
        ("1002", "Cancer Research", "Mutations in the QRX7 promoter were associated with resistance to platinum therapy."),
        ("1003", "Journal of Botany", "Leaf stomatal density increased under elevated carbon dioxide in wheat."),
    ]
    out = []
    for pmc, j, s in specs:
        art = parse_article(etree.fromstring(jats(pmc, j, s)), min_body_words=100)
        assert art is not None
        out.append(art)
    return out


def test_parse_cleans_and_flattens(articles):
    a = articles[0]
    assert a["pmcid"] == "PMC1001"
    titles = [s["title"] for s in a["sections"]]
    assert "Results > Subgroup" in titles
    body = " ".join(p for s in a["sections"] for p in s["paragraphs"])
    assert "[1]" not in body and "CAPTION NOISE" not in body


def test_journal_filter():
    root = etree.fromstring(jats("9", "Journal of Botany", "x"))
    assert parse_article(root, journals=["cancer"], min_body_words=100) is None
    assert parse_article(root, journals=["botany"], min_body_words=100) is not None


def test_chunkers_skip_funding_and_respect_sections(articles):
    a = articles[0]
    sec = chunk_section(a, max_words=150)
    assert all("Funding" not in c["section"] for c in sec)
    assert all(c["n_words"] <= 160 for c in sec)
    fixed = chunk_fixed(a, size=150, overlap=30)
    assert len(fixed) > 1 and all(c["n_words"] <= 150 for c in fixed)
    assert len({c["chunk_id"] for c in sec + fixed}) == len(sec) + len(fixed)


class HashEncoder:
    """Deterministic bag-of-words encoder so tests need no model download."""
    name = "hash-enc"

    def __init__(self, dim=512):
        self.dim = dim

    def _enc(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in t.lower().split():
                out[r, hash(w) % self.dim] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    encode_docs = encode_queries = _enc


def test_semantic_chunker_runs(articles):
    rows = chunk_semantic(articles[0], HashEncoder(), max_words=120)
    assert rows and all(r["strategy"] == "semantic" for r in rows)


def _setup(articles):
    rows = []
    for a in articles:
        rows += chunk_section(a, max_words=150)
    texts = [r["text"] for r in rows]
    pm = [r["pmcid"] for r in rows]
    qs = [
        {"qid": "a", "question": "How much did zorvelimab shrink tumors in mice?", "pmcid": "PMC1001",
         "evidence": "Treatment with zorvelimab reduced tumor volume by 41 percent in murine xenografts."},
        {"qid": "b", "question": "Which promoter mutations cause platinum resistance?", "pmcid": "PMC1002",
         "evidence": "Mutations in the QRX7 promoter were associated with resistance to platinum therapy."},
        {"qid": "c", "question": "orphan", "pmcid": "PMC1002", "evidence": "this sentence appears nowhere in the paper at all"},
    ]
    return rows, texts, pm, qs


def test_gold_and_bm25(articles):
    from src.retrieval.retrievers import BM25Retriever

    rows, texts, pm, qs = _setup(articles)
    gold = build_gold(texts, pm, qs)
    assert len(gold["a"]) == 1 and len(gold["b"]) == 1 and gold["c"] == set()
    res = BM25Retriever(texts).search([q["question"] for q in qs], 10)
    m, _ = evaluate(res, qs, gold)
    assert m["n_unmatched"] == 1 and m["hit@5"] == 1.0 and m["mrr@10"] == 1.0


def test_dense_cache_and_hybrid(articles, tmp_path):
    from src.retrieval.retrievers import BM25Retriever, DenseRetriever, HybridRetriever

    rows, texts, pm, qs = _setup(articles)
    gold = build_gold(texts, pm, qs)
    enc = HashEncoder()
    dense = DenseRetriever(enc, texts, cache_dir=tmp_path, shard_size=3, search_block=4)  # force multi-shard/block
    qtexts = [q["question"] for q in qs]
    m, _ = evaluate(dense.search(qtexts, 10), qs, gold)
    assert m["hit@10"] == 1.0
    n_files = len(list(dense.dir.glob("shard_*.npy")))
    DenseRetriever(enc, texts, cache_dir=tmp_path, shard_size=3)  # reuses cache, no re-embedding
    assert len(list(dense.dir.glob("shard_*.npy"))) == n_files
    hyb = HybridRetriever([BM25Retriever(texts), dense])
    m, _ = evaluate(hyb.search(qtexts, 10), qs, gold)
    assert m["hit@10"] == 1.0


def test_metrics():
    assert hit_at_k([False, True], 1) == 0.0 and hit_at_k([False, True], 2) == 1.0
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert ndcg_at_k([True, False], 10, 1) == 1.0
    assert ndcg_at_k([False, True], 10, 1) == pytest.approx(1 / np.log2(3))

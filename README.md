# sci-rag

Domain-specific scientific QA on a 6GB consumer GPU: hybrid retrieval, fine-tuned
embedder + reranker, and a RAFT-style QLoRA generator. (Results table goes here first.)

## Quickstart (PowerShell: keep each command on ONE line)

```
python -m pytest -q

# 1. Download open-access full text (resumable; ~20-40 min for 3000 articles)
python -m src.ingest.fetch_europepmc --journal "BMC Cancer" --year-from 2018 --max-articles 3000 --contact you@example.com

# 2. Parse to articles.jsonl, then build chunk sets
python -m src.ingest.parse_pmc data/raw --out data/articles.jsonl
python -m src.ingest.chunk --articles data/articles.jsonl --strategy section --out data/chunks/section.parquet
python -m src.ingest.chunk --articles data/articles.jsonl --strategy fixed --out data/chunks/fixed.parquet

# 3. Label questions (resumable, 20 per sitting)
python -m src.eval.label --chunks data/chunks/section.parquet --out data/eval/questions.jsonl

# 4. Evaluate
python -m src.eval.harness --chunks data/chunks/section.parquet --eval data/eval/questions.jsonl --retriever bm25 --out experiments/section_bm25.json
```

# Project 2 — Re-Ranking Pipeline Tasks

## Goal
Extend the existing BM25-only retrieval into a full two-stage retrieve-then-rerank pipeline, then pass the top results to an LLM for answer generation.

## Current State
- `build_cache.py` — fixed-size chunking (800 chars, 200 overlap) → `corpus.json` (3,241 chunks from 37 PDFs)
- `retrieval.py` — shared BM25 utilities (load_corpus, build_bm25, tokenize, search, is_hit)
- `eval_keyword.py` — BM25-only eval, `--retrieval_k` (default 100), `--topk` (default 5)
- `rerank.py` — ZeroEntropy API reranker (`zerank-2`), `MAX_CANDIDATES=50`, `TOP_K=10`
- `eval_rerank.py` — two-stage eval: BM25 → rerank, 17/17 (100%) easy, 9/9 (100%) hard
- `answer.py` — full pipeline: BM25 → rerank → expand context (prev/next page) → LLM

---

## Tasks

### ~~1. Expand BM25 retrieval to top-100~~ ✅
### ~~2. Build `rerank.py` — Cross-encoder re-ranker (Stage 2)~~ ✅
### ~~3. Build `eval_rerank.py` — Two-stage eval script~~ ✅
### ~~4. Add cross-encoder dependency to `pyproject.toml`~~ ✅ (switched to `zeroentropy` API)
### ~~5. Build `answer.py` — LLM answer generation (Stage 3)~~ ✅
### ~~6. Eval comparison~~ ✅
- Run both `eval_keyword.py` and `eval_rerank.py` against `evals/easy.csv` and `evals/hard.csv`
- Compare accuracy: BM25-only vs BM25 + ZeroEntropy reranker
- Results saved to `evals/results_{file}_rerank.json`
